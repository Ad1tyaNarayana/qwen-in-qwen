#!/usr/bin/env python3
"""Plain stock generate for a guest. No reference in the process, no streamer, no stopping criteria.
Hashes the BF16 weights from the loaded model with its own code; records how forward/generate resolve to
stock source. Output compared offline with rvi.py."""
import argparse,hashlib,inspect,json,sys,time,subprocess,threading,importlib.util
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--repo',type=Path,required=True);p.add_argument('--upstream',type=Path,required=True)
p.add_argument('--out',type=Path,required=True);p.add_argument('--guest',type=Path,required=True)
p.add_argument('--default-flags',action='store_true');p.add_argument('--tokens',type=int,default=16384);p.add_argument('--seed-records',type=int,default=0)
a=p.parse_args();a.repo=a.repo.resolve();a.out.mkdir(parents=True,exist_ok=False)
sys.path.insert(0,str(a.repo/'experiments/neural-accelerator'))
import torch
torch.set_num_threads(4)
if not a.default_flags:
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
total=torch.cuda.get_device_properties(0).total_memory
torch.cuda.set_per_process_memory_fraction(15*2**30/total)
s=importlib.util.spec_from_file_location('mp',a.repo/'experiments/memory-map/prepare.py');mp=importlib.util.module_from_spec(s);s.loader.exec_module(mp)
mp.prepare(a.upstream,a.out/'scratch',64,4096);sys.path.insert(0,str(a.out/'scratch'))
from chipformer.rv import elf,spec
from constructor import build
from stock import build_model,boot_cache
import reference,transformers
from transformers import Qwen3_5ForCausalLM
stop=threading.Event();lo=[total]
def mon():
    while not stop.is_set():
        try:
            lo[0]=min(lo[0],int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())*2**20)
            if lo[0]<512*2**20: print('below guard',lo,flush=True);import os;os._exit(3)
        except Exception as e: print(e,flush=True)
        stop.wait(5)
threading.Thread(target=mon,daemon=True).start()
def guard(req=0):
    torch.cuda.empty_cache();f,_=torch.cuda.mem_get_info()
    assert f-req>=512*2**20,(f,req)
c=build(4);model,meta=build_model(c,'cuda',memory_guard=guard)
h=hashlib.sha256();n=0
for k,v in sorted(model.state_dict().items()):
    f=v.detach().view(-1);assert f.dtype==torch.bfloat16;n+=f.numel()*2
    for q in f.split(4*1024*1024): h.update(q.cpu().contiguous().view(torch.uint16).numpy().astype('<u2',copy=False).tobytes())
mod=sys.modules[Qwen3_5ForCausalLM.__module__]
rep=dict(weight_sha256=h.hexdigest(),weight_bytes=n,constructor=c.source_manifest,transformers=transformers.__version__,
  modeling_file=mod.__file__,modeling_sha256=hashlib.sha256(open(mod.__file__,'rb').read()).hexdigest(),
  type_exact=type(model) is Qwen3_5ForCausalLM,forward_qualname=type(model).forward.__qualname__,forward_file=inspect.getsourcefile(type(model).forward),
  instance_overrides=sorted(k for k in model.__dict__ if k in('forward','generate','prepare_inputs_for_generation')),
  layer_types=sorted({type(l).__name__ for l in model.model.layers}),attn_types=sorted({type(m).__name__ for m in model.modules()}),
  hooks=sum(len(m._forward_hooks)+len(m._forward_pre_hooks) for m in model.modules()))
(a.out/'report.json').write_text(json.dumps(rep,indent=1))
print(rep,flush=True)
m=reference.Machine.from_program(elf.program(a.guest/'engine.elf'));m.maximum=4
seed=0
while seed<a.seed_records and not m.halted:
    seed+=len(m.record())
rep['seed_token_offset']=seed
t=time.perf_counter()
with torch.inference_mode():
    cache=boot_cache(model,c,m)
    rep['cache_type']=type(cache).__name__;rep['cache_module']=type(cache).__module__
    out=model.generate(input_ids=torch.tensor([[spec.VOCAB,spec.tok('GO')]],device='cuda'),past_key_values=cache,max_new_tokens=a.tokens,do_sample=False,eos_token_id=spec.tok('HALT'),pad_token_id=spec.VOCAB)
el=time.perf_counter()-t
g=out[0,2:].cpu().numpy().astype('<u4');(a.out/'tokens.u32').write_bytes(g.tobytes())
stop.set()
rep.update(generated=len(g),seconds=el,tps=len(g)/el,trace_sha256=hashlib.sha256(g.tobytes()).hexdigest(),min_free_bytes=lo[0],peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),done=True)
(a.out/'report.json').write_text(json.dumps(rep,indent=1))
print('done',len(g),el,flush=True)
