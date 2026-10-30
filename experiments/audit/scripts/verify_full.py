#!/usr/bin/env python3
"""Independent full-program check: rvi.py vs actual GPU trace/RAM/registers/output, no project code imported."""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import rvi
name,elf,trace=sys.argv[1:4];ram=sys.argv[4] if len(sys.argv)>4 else None;regs=sys.argv[5] if len(sys.argv)>5 else None
t=time.time();c=rvi.CPU(elf).run(10**9);el=time.time()-t
mine=np.frombuffer(c.tokens,dtype='<u4')
a=np.fromfile(trace,dtype='<u4')
q=(a>=rvi.BASE['QS'])&(a<rvi.BASE['QD']+128)
s=a[~q]
r=dict(name=name,elf_sha256=hashlib.sha256(open(elf,'rb').read()).hexdigest(),seconds=el,instructions=c.steps,ndots=c.ndots,
  rvi_tokens=len(mine),gpu_tokens=len(a),gpu_qs_qd_tokens=int(q.sum()),equal_after_strip=bool(len(s)==len(mine) and np.array_equal(s,mine)),
  first_diff=None,exit=getattr(c,'exit',None),fault=c.fault,halted=c.halted,
  rvi_ram_sha256=hashlib.sha256(c.ram).hexdigest(),output_tail=c.out[-120:].decode('latin1'),output_sha256=hashlib.sha256(c.out).hexdigest())
if not r['equal_after_strip']:
    n=min(len(s),len(mine));d=np.flatnonzero(s[:n]!=mine[:n]);r['first_diff']=int(d[0]) if len(d) else n
if ram: r['ram_equal_gpu']=open(ram,'rb').read()==bytes(c.ram)
if regs:
    g=json.load(open(regs));r['regs_equal_gpu']=all(g[k]==c.x for k in g)
print(json.dumps(r,indent=1));Path(f'experiments/audit/results/full-{name}.json').write_text(json.dumps(r,indent=1))
