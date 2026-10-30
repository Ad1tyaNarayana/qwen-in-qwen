"""Publish the retained final GPU Qwen state after checking its exact artifact bindings."""
import json
import sys
from types import SimpleNamespace
import numpy as np
from common import HERE, ROOT, compiler, sha, write_json
from record import neural_snapshot, program_info, decode_trace, bundle, reference_checker


def main():
    import torch
    torch.set_num_threads(2)
    c=compiler()
    sys.path.insert(0,str(ROOT/'experiments/showcase'))
    from make_manifests import drom_hash
    from chipformer.q35.ir import Head
    from chipformer.rv import spec
    inputs=ROOT/'.cache/tiny-qwen35/trained-program'
    actual=ROOT/'.cache/tiny-qwen35/actual-covariance-20261030/main'
    path=ROOT/'.cache/explorer/tiny-qwen35-checkpoint/main/checkpoint.pt'
    metadata=json.loads((actual.parent/'checkpoint-metadata.json').read_text())
    report=json.loads((actual/'full-report.json').read_text())
    ref=json.loads((inputs/'reference.json').read_text())
    assert sha(path)==metadata['checkpoint_sha256']
    saved=torch.load(path,map_location='cpu',weights_only=True)
    assert saved['offset']==metadata['offset']==ref['tokens']==35836005
    assert saved['binding']==metadata['binding']
    assert saved['binding']['elf']==sha(inputs/'engine.elf')==ref['elf_sha256']
    assert saved['binding']['reference']==sha(actual/'actual.tokens.u32')==sha(inputs/'reference.tokens.u32')==ref['sha256']
    generated='5e156bdb8fa6c543d96f0760a1a4d18dfea4cba3e558673a909bed1be0fde7da'
    assert report['constructor']['generated_sha256']==generated
    steps=[]
    for L in range(c.n_layers()):
        heads=[h for h in c.nodes if isinstance(h,Head) and h.level==2*L+1]
        if heads: steps.append(dict(kind='head',heads=heads,**saved['heads'][len(steps)]))
    assert len(steps)==len(saved['heads'])
    s=neural_snapshot(SimpleNamespace(ex=SimpleNamespace(steps=steps)),c,saved['offset'])
    assert bytes.fromhex(s['ram_hex'])==(actual/'actual-ram.bin').read_bytes()==(inputs/'reference-ram.bin').read_bytes()
    assert s['registers']==json.loads((actual/'actual-registers.json').read_text())
    assert all(r==ref['final_registers'] for r in s['registers'].values())
    reference_checker(inputs)(s,saved['offset']-1)
    tokens=np.memmap(actual/'actual.tokens.u32',dtype='<u4',mode='r')
    assert tokens[-1]==spec.tok('HALT')
    s['pending_token']=int(tokens[-1]);s['state_boundary']='Actual final GPU checkpoint. Last generated HALT awaits feedback.'
    s['verification']=dict(passed=True,reference_consumed_tokens=len(tokens)-1,ram_bytes=65536,register_values=96,pc=True,anchor=True,count=True,
                           scope='Every RAM byte, register, PC, anchor and count independently replayed to consumed-feedback index. Full generated trace SHA matches independent trace.')
    program=program_info(inputs)
    start=len(tokens)-4096
    pairs=decode_trace(tokens,spec,program['entry'],start=start)
    assert int(pairs[-1,1])==s['pc']
    out=HERE/'data/recordings/tiny-qwen35'
    binding=dict(checkpoint_sha256=sha(path),actual_trace_sha256=ref['sha256'],elf_sha256=ref['elf_sha256'],
                 generated_sha256=generated,reference_used='comparison only',source='Retained final actual autonomous GPU checkpoint; decoded on CPU')
    entry=bundle(out,'tiny-qwen35-final',pairs,start,program,[s],inputs,binding)
    record=json.loads((out/'tiny-qwen35-final.json').read_text())
    byte_base,_=spec.FAMILIES['BYTE']
    prior=tokens[:start];positions=np.flatnonzero((prior>=byte_base)&(prior<byte_base+256))
    record['prefix_byte_events']=[[int(i)+1,int(prior[i])-byte_base] for i in positions]
    write_json(out/'tiny-qwen35-final.json',record)
    entry['url']='tiny-qwen35/'+entry['url']
    catalog=json.loads((HERE/'data/recordings/index.json').read_text())
    catalog['recordings']=[r for r in catalog['recordings'] if r['id']!=entry['id']]+[entry]
    write_json(HERE/'data/recordings/index.json',catalog)
    write_json(HERE/'data/tiny-qwen35-validation.json',dict(passed=True,**binding,verification=s['verification'],snapshots=1))
    manifest=dict(demo='tiny-qwen35',status='neural-verified',network='unified W=4 full-map; compiled outer network weights, not pretrained',
                  network_weight_sha256='0e515bbb3109997a130c903deb40dce3447f8991e15b6637e0071c71f8b6e07d',
                  weight_hash_scope='Shared stock BF16 tensor hash bound to constructor generated_sha256; the outer CPU weights are compiled, not pretrained.',
                  generated_sha256=generated,constructor_binding=report['constructor'],elf_sha256=ref['elf_sha256'],
                  drom_image_sha256=drom_hash(inputs/'engine.elf'),
                  instruction_count=25121038,trace_tokens=len(tokens),mismatches=0,outputs='Once upon a time, there',
                  output_source='BYTE records from actual self-fed GPU trace',stock_hf_prefix_tokens_checked=16384,
                  generation_seconds=report['compute_seconds'],timing_scope='Retained generation calls; excludes setup, IO, comparisons, waits, power outage and discarded replayed work',
                  quality_limitation='291/327 float agreement on six inspected prompts. These prompts were used during quantization investigation and are not an unseen benchmark.',
                  explorer_recording='tiny-qwen35/tiny-qwen35-final.json',
                  evidence_paths=['experiments/tiny-qwen35/evidence/neural-validation.json','experiments/tiny-qwen35/evidence/covariance-quality.json','experiments/explorer/data/tiny-qwen35-validation.json','experiments/showcase/evidence/weights.json'])
    write_json(ROOT/'experiments/showcase/manifest/tiny-qwen35.json',manifest)
    print(json.dumps(entry))

if __name__=='__main__':main()
