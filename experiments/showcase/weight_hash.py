"""SHA-256 of stock-emitted BF16 state tensors in canonical lexical name order.

Hash input is the concatenation of each named state_dict tensor's contiguous
little-endian BF16 bytes, including tied lm_head and zero padding-layer tensors.
Names, shapes, per-tensor digests and source bindings are retained separately.
No ELF, snapshot, program, reference or guest data enters emission.
"""
import argparse
import gc
import hashlib
import inspect
import json
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original',action='store_true')
    p.add_argument('--out',type=Path,default=Path(__file__).parent/'evidence/weights.json')
    a=p.parse_args()
    import torch
    torch.set_num_threads(4)
    sys.path.insert(0,str(ROOT/'experiments/neural-accelerator'))
    from prepare import prepare
    with tempfile.TemporaryDirectory(prefix='showcase-hash-',dir=ROOT/'.cache') as tmp:
        tmp=Path(tmp);manifest=prepare(tmp/'package')
        from chipformer.q35 import emit
        from chipformer.q35.tiny import BASE,CODE_GAIN,pick_layout
        from chipformer.rv import spec
        from chipformer.rv.tiny import embedding_matrix
        from transformers import Qwen3_5ForCausalLM,Qwen3_5TextConfig
        if a.original:
            from chipformer.rv import core
            circuit=core.build()
        else:
            from constructor import build
            circuit=build(4)
        layout=pick_layout(len(circuit.dims));count=circuit.n_layers()
        vocabulary=-(-(spec.VOCAB+1)//64)*64
        config=Qwen3_5TextConfig(**dict(BASE,hidden_size=layout.hidden,vocab_size=vocabulary,
            num_hidden_layers=count+1,layer_types=['linear_attention']*count+['full_attention'],
            linear_num_key_heads=circuit.target.key_heads,linear_num_value_heads=circuit.target.value_heads,
            intermediate_size=circuit.target.intermediate,eos_token_id=spec.tok('HALT'),
            pad_token_id=spec.VOCAB,bos_token_id=spec.VOCAB))
        with torch.device('meta'):model=Qwen3_5ForCausalLM(config)
        shapes={k:list(v.shape) for k,v in model.state_dict().items()}
        del model
        source=inspect.getsource(emit.emit_layers)
        for before,after in [('def emit_layers(','def iter_layers('),
            ('            out[p + k] = t','        yield {p + k: t for k, t in layer.items()}'),('    return out','')]:
            assert source.count(before)==1;source=source.replace(before,after)
        namespace=dict(emit.__dict__)
        exec(compile(source,'<stock streaming emitter>','exec'),namespace)
        entries={}
        def store(name,value):
            value=value.to(torch.bfloat16).contiguous()
            assert list(value.shape)==shapes[name]
            raw=value.view(torch.uint16).numpy().astype('<u2',copy=False).tobytes()
            path=tmp/(name+'.bf16');path.write_bytes(raw)
            entries[name]=dict(name=name,shape=shapes[name],bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        for weights in namespace['iter_layers'](circuit,layout,config.to_dict(),0,'model.',None,set(circuit.keep)):
            for name,value in weights.items():store(name,value)
            del weights;gc.collect()
        norm=torch.full((layout.hidden,),-1.)
        for dim in circuit.keep:norm[layout.dim(dim)]=CODE_GAIN
        emit.check_bf16('final norm',norm)
        store('model.norm.weight',norm)
        embedding=embedding_matrix(circuit,layout,vocabulary)
        store('model.embed_tokens.weight',embedding)
        # Stock ties these weights after loading.
        import shutil
        shutil.copyfile(tmp/'model.embed_tokens.weight.bf16',tmp/'lm_head.weight.bf16')
        entries['lm_head.weight']=dict(entries['model.embed_tokens.weight'],name='lm_head.weight')
        del embedding;gc.collect()
        for name in sorted(set(shapes)-set(entries)):
            assert name.startswith(f'model.layers.{count}.')
            store(name,torch.zeros(shapes[name],dtype=torch.bfloat16))
        digest=hashlib.sha256()
        for name in sorted(entries):
            with (tmp/(name+'.bf16')).open('rb') as f:
                while raw:=f.read(8*1024*1024):digest.update(raw)
        report=dict(sha256=digest.hexdigest(),circuit='original full-map' if a.original else 'unified W=4 full-map',
            canonical_order='lexicographic state_dict name, UTF-8 Python ordering',
            hash_input='concatenated contiguous little-endian BF16 tensor bytes; tied lm_head included',
            boot_snapshot_used=False,bytes=sum(x['bytes'] for x in entries.values()),
            tensors=[entries[n] for n in sorted(entries)],config=config.to_dict(),
            constructor=getattr(circuit,'source_manifest',None),prepare_manifest=manifest,
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            stock_builder_sha256=hashlib.sha256((ROOT/'experiments/neural-accelerator/stock.py').read_bytes()).hexdigest())
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(report,indent=2)+'\n')
        print(digest.hexdigest(),flush=True)


if __name__=='__main__':main()
