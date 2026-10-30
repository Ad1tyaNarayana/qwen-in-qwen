#!/usr/bin/env python3
"""C4 at full depth: unfused unified executor over the whole Doom frame, then literal comparison of ALL recurrent
state against the fused run's retained final GPU checkpoint."""
import argparse,hashlib,json,sys,time,types
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser()
for n in('repo','upstream','out','inputs','checkpoint'): p.add_argument('--'+n,type=Path,required=True)
p.add_argument('--tokens',type=int,required=True)
a=p.parse_args();a.repo=a.repo.resolve();a.out.mkdir(parents=True,exist_ok=False)
sys.path[:0]=[str(a.repo/'experiments/tiny-llm-ndot-fused'),str(a.repo/'experiments/tiny-llm-ndot')]
import torch,worker
args=types.SimpleNamespace(repo=a.repo,upstream=a.upstream.resolve(),out=a.out/'runtime',inputs=a.inputs,original_circuit=False)
fr,fused,spec,meta=worker.build(args,{})
assert 'unfused' in meta['mode'],meta['mode']
h=hashlib.sha256();done=0;t=time.perf_counter();rep=dict(mode=meta['mode'],tokens=a.tokens)
with torch.no_grad():
    while done<a.tokens:
        n=min(8192,a.tokens-done);c=fr.run(n)[0].astype('<u4');h.update(c.tobytes());done+=n
        if done%(1<<18)<8192:
            print(done,time.perf_counter()-t,flush=True);(a.out/'progress.txt').write_text(str(done))
rep.update(seconds=time.perf_counter()-t,trace_sha256=h.hexdigest(),final_token=int(fr.tok[0].item()))
ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
rep['checkpoint_offset']=ck['metadata']['offset']
heads=[st for st in fr.ex.steps if st['kind']=='head'];assert len(heads)==len(ck['heads'])
bad=dict(S=0,hist=0);words=dict(S=0,hist=0);first=[]
for i,(st,sv) in enumerate(zip(heads,ck['heads'])):
    for k in('S','hist'):
        A=st[k].detach().cpu();B=sv[k]
        assert A.shape==B.shape and A.dtype==B.dtype
        ai=A.view(-1).numpy().view('u4');bi=B.reshape(-1).numpy().view('u4')
        d=int((ai!=bi).sum());bad[k]+=d;words[k]+=len(ai)
        if d and len(first)<8:first.append((i,k,d,[h_.name for h_ in st['heads']][:2]))
rep.update(words_compared=words,differing_words=bad,first_differences=first,
  x_equal=bool((fr.x.detach().cpu().numpy().view('u4')==ck['x'].numpy().view('u4')).all()),tok_equal=bool((fr.tok.cpu()==ck['tok']).all()),
  scratch_y_zero=not torch.count_nonzero(fr.ex.y).item())
(a.out/'report.json').write_text(json.dumps(rep,indent=1));print(rep,flush=True)
