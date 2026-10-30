#!/usr/bin/env python3
"""Audit worker. No reference tokens anywhere in this file. --mode fused|unfused runs the actual project
executor over many guests in ONE process (same network instance) with a generic HALT stop; hashes every
non-state executor tensor before and after; dumps tokens, RAM, registers, full-state digest."""
import argparse,hashlib,json,sys,time,types
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser()
p.add_argument('--repo',type=Path,required=True);p.add_argument('--upstream',type=Path,required=True)
p.add_argument('--out',type=Path,required=True);p.add_argument('--guests',type=Path,nargs='+',required=True)
p.add_argument('--mode',default='fused');p.add_argument('--limit',type=int,default=2000000)
a=p.parse_args();a.repo=a.repo.resolve()
# A failed import can leave an empty output directory. Never overwrite results.
if a.out.exists():
    assert not any(a.out.iterdir()), f'Output is not empty: {a.out}'
a.out.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(a.repo/'experiments/showcase'))
sys.path.insert(0,str(a.repo/'experiments/tiny-llm-ndot-fused'))
sys.path.insert(0,str(a.repo/'experiments/tiny-llm-ndot'))
import torch
def thash(h,t):
    h.update(str((t.dtype,tuple(t.shape))).encode());h.update(t.detach().cpu().numpy().tobytes())
def walk(h,v):
    if torch.is_tensor(v): thash(h,v)
    elif isinstance(v,(list,tuple)):
        for x in v: walk(h,x)
    elif isinstance(v,dict):
        for k in sorted(v,key=str): walk(h,v[k])
def weights_hash(fr,fused):
    h=hashlib.sha256();n=0
    for st in fr.ex.steps:
        for k in sorted(st):
            if k in('S','hist'): continue
            if k in('heads',): continue
            walk(h,st[k])
    for k in ('fam_idx','pay_idx','fam_w','pay_w','base','bits','ids','small','row0','table'): thash(h,getattr(fr,k))
    if fused and fused.get('batches'):
        for b in fused['batches'].values(): walk(h,b['packed'])
    return h.hexdigest()
def state_digest(fr):
    h=hashlib.sha256()
    for st in fr.ex.steps:
        if st['kind']=='head':
            thash(h,st['S']);thash(h,st['hist'])
    thash(h,fr.x);thash(h,fr.tok);return h.hexdigest()
args=types.SimpleNamespace(repo=a.repo,upstream=a.upstream.resolve(),out=a.out/'runtime',inputs=a.guests[0],original_circuit=False)
if a.mode=='fused':
    import gpu_worker
    t=time.perf_counter();fr,fused,spec,meta=gpu_worker.build(args,{})
else:
    import worker
    t=time.perf_counter();fr,fused,spec,meta=worker.build(args,{})
report=dict(mode=a.mode,build_seconds=time.perf_counter()-t,meta={k:v for k,v in meta.items() if k in('mode','constructor','network_weight_sha256','torch','triton','gpu')},guests={})
from runtime import reset
import reference,worker as W
from chipformer.rv import elf
sys.path.insert(0,str(a.repo/'experiments/dmc-cuda/source/helpers'))
from state import read_ram
w0=weights_hash(fr,fused if a.mode=='fused' else None);report['nonstate_tensors_sha256_start']=w0
HALT=spec.tok('HALT')
def boot(g):
    m=reference.Machine.from_program(elf.program(g/'engine.elf'));m.maximum=4
    reset(fr,m,fused if a.mode=='fused' else None)
    if a.mode!='fused':
        pass
    return m
with torch.no_grad():
    for g in a.guests:
        boot(g);toks=[];stop=None;t=time.perf_counter()
        while len(toks)<a.limit and stop is None:
            c=fr.run(2048)[0];w=np.flatnonzero(c==HALT)
            if len(w): stop=len(toks)+int(w[0])+1
            toks.extend(int(x) for x in c)
        first=time.perf_counter()-t
        if stop is None: report['guests'][g.name]=dict(halted=False,tokens=len(toks));continue
        pass1=toks[:stop]
        boot(g);got=[];t=time.perf_counter()
        while len(got)<stop:
            got.extend(int(x) for x in fr.run(min(2048,stop-len(got)))[0])
        second=time.perf_counter()-t
        assert got==pass1
        arr=np.asarray(got,dtype='<u4');(a.out/f'{g.name}.tokens.u32').write_bytes(arr.tobytes())
        ram=read_ram(fr.ex,spec.RAM_BASE,65536);(a.out/f'{g.name}.ram.bin').write_bytes(ram)
        try: regs=W.read_registers(fr,fused['circuit'])
        except AssertionError as e: regs=dict(error=repr(e)[:300])
        report['guests'][g.name]=dict(halted=True,tokens=stop,seconds_first=first,seconds_second=second,
            trace_sha256=hashlib.sha256(arr.tobytes()).hexdigest(),ram_sha256=hashlib.sha256(ram).hexdigest(),
            registers=regs,state_sha256=state_digest(fr),elf_sha256=hashlib.sha256((g/'engine.elf').read_bytes()).hexdigest())
        report['nonstate_tensors_sha256_end']=weights_hash(fr,fused if a.mode=='fused' else None)
        (a.out/'report.json').write_text(json.dumps(report,indent=1))
        print(g.name,stop,first,flush=True)
report['done']=True;(a.out/'report.json').write_text(json.dumps(report,indent=1))
