#!/usr/bin/env python3
"""Attack 5: corrupt project checkpoints and feed them through the project's own restore + checks."""
import argparse,copy,json,sys,types,shutil
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser()
for n in('repo','upstream','out','inputs'): p.add_argument('--'+n,type=Path,required=True)
a=p.parse_args();a.repo=a.repo.resolve();a.out.mkdir(parents=True,exist_ok=False)
sys.path[:0]=[str(a.repo/'experiments/showcase'),str(a.repo/'experiments/tiny-llm-ndot-fused'),str(a.repo/'experiments/tiny-llm-ndot')]
import torch,gpu_worker,worker as W
args=types.SimpleNamespace(repo=a.repo,upstream=a.upstream.resolve(),out=a.out/'runtime',inputs=a.inputs,original_circuit=False)
meta=json.loads((a.inputs/'reference.json').read_text())
fr,fused,spec,md=gpu_worker.build(args,meta)
sys.path.insert(0,str(a.repo/'experiments/dmc-cuda/source/helpers'))
from state import read_ram
exp=np.fromfile(a.inputs/'reference.tokens.u32',dtype='<u4');ram0=(a.inputs/'reference-ram.bin').read_bytes()
binding=dict(elf=meta['elf_sha256'],reference=meta['sha256'],scripts=md['scripts'],mode=md['mode'])
N=8192
with torch.no_grad():
    g=fr.run(N)[0];assert (g==exp[:N]).all()
    ck=a.out/'ck.pt';W.save(fr,fused,ck,binding,N,0.,0.)
    base=torch.load(ck,map_location='cpu',weights_only=True)
    heads=[st for st in fr.ex.steps if st['kind']=='head']
    names=[[h.name for h in st['heads']] for st in heads]
    (a.out/'head-names.json').write_text(json.dumps(names))
    def idx(prefix): return next(i for i,n in enumerate(names) if any(x.startswith(prefix) for x in n))
    def flip(t,pos): 
        v=t.reshape(-1);i=pos if pos>=0 else int(torch.nonzero(v)[len(torch.nonzero(v))//2]);w=v.numpy().view('u4');w[i]^=1<<3;return i
    def live_reg(s):
        i=idx('regs.rs1');return flip(s['heads'][i]['S'],-1)
    def far_drom(s):
        i=idx('drom');t=s['heads'][i]['S'];return flip(t,t.numel()-5)
    def far_ram(s):
        i=idx('ram');t=s['heads'][i]['S'];return flip(t,t.numel()-5)
    def tok_s(s): s['tok'][0]^=1;return 0
    def x_s(s): return flip(s['x'],-1)
    def bind(s): s['binding']['elf']='0'*64;return 0
    def hist(s):
        for i in range(len(s['heads'])):
            t=s['heads'][i]['hist']
            if torch.count_nonzero(t): return flip(t,-1)
    def none(s): return 0
    res={}
    for name,f in [('none',none),('binding',bind),('live_register_S',live_reg),('next_embedding_x',x_s),('token_word',tok_s),('lag_history',hist),('unused_drom_row',far_drom),('unused_ram_row',far_ram)]:
        s=copy.deepcopy(base);f(s);pt=a.out/f'{name}.pt';torch.save(s,pt)
        try:
            W.restore(fr,fused,pt,binding)
        except AssertionError as e:
            res[name]='rejected at restore (binding/shape)';continue
        out=[];got=N
        while got<len(exp):
            n=min(2048,len(exp)-got);out.extend(int(x) for x in fr.run(n)[0]);got+=n
        o=np.asarray(out,dtype='<u4');w=exp[N:];k=np.flatnonzero(o!=w)
        ram=read_ram(fr.ex,spec.RAM_BASE,65536) if True else None
        res[name]=dict(token_mismatches=int(len(k)),first=int(k[0]) if len(k) else None,final_ram_equal=ram==ram0)
        print(name,res[name],flush=True);(a.out/'report.json').write_text(json.dumps(res,indent=1))
    (a.out/'report.json').write_text(json.dumps(res,indent=1));print(res,flush=True)
