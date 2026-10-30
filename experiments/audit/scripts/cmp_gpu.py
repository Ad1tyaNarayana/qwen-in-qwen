#!/usr/bin/env python3
"""Compare a GPU audit dir with rvi.py, guest by guest (tokens less QS/QD, final RAM, registers, output)."""
import json,sys,hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0,'experiments/audit/scripts')
import rvi
d=Path(sys.argv[1]);rep=json.load(open(d/'report.json'));out={}
print('meta',{k:v for k,v in rep['meta'].items() if k!='constructor'},rep['meta']['constructor']['generated_sha256'] if rep['meta'].get('constructor') else None)
print('nonstate start/end',rep['nonstate_tensors_sha256_start'],rep.get('nonstate_tensors_sha256_end'),rep.get('done'))
for n,g in rep['guests'].items():
    c=rvi.CPU(f'.cache/audit/guests/{n}/engine.elf').run(10**8)
    a=np.fromfile(d/f'{n}.tokens.u32',dtype='<u4');q=(a>=rvi.BASE['QS'])&(a<rvi.BASE['QD']+128);s=a[~q]
    mine=np.frombuffer(c.tokens,dtype='<u4')
    r=dict(gpu_tokens=len(a),rvi_tokens=len(mine),equal=bool(len(s)==len(mine) and (s==mine).all()),ram_equal=(d/f'{n}.ram.bin').read_bytes()==bytes(c.ram),
      regs=None,fault=c.fault,exit=getattr(c,'exit',None),out=c.out[-24:].decode('latin1'),state=g['state_sha256'][:12],elf=g['elf_sha256'][:12],trace=g['trace_sha256'][:12])
    gr=g['registers']
    if 'error' in gr: r['regs']=gr['error'][:120]
    else:
        x=list(c.x);x[2]=x[2]
        diff=[i for i in range(32) if gr['regs.rs1'][i]!=x[i]]
        r['regs']=diff
    out[n]=r;print(n,json.dumps(r))
json.dump(dict(meta=rep['meta'],start=rep['nonstate_tensors_sha256_start'],end=rep.get('nonstate_tensors_sha256_end'),guests=out),open(sys.argv[2],'w'),indent=1)
