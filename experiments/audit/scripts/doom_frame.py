#!/usr/bin/env python3
"""Attack 2: recompute the Doom frame from rvi.py's RAM only; then mutate the game data and see the frame change."""
import hashlib,json,struct,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,'experiments/audit/scripts')
import rvi
E0=sys.argv[1];M=json.load(open(sys.argv[2]));A=M['metadata_addresses']
def frame(raw):
    c=rvi.CPU(raw);sur=np.zeros((240,320),dtype='<u2');cov=np.zeros((240,320),bool);n=0;mark=rvi.BASE['BYTE']+256;seen=0
    while not c.halted and c.steps<6_000_000:
        c.step()
        t=c.tokens
        while seen<len(t):
            if t[seen]==mark:
                r=c.ram;g=lambda k:struct.unpack_from('<I',r,A['qp_strip_'+k]-rvi.RAMB)[0]
                d={k:g(k) for k in('ptr','x','y','h','n','flags')}
                s,cnt=d['ptr']-rvi.RAMB,d['n']*d['h'];mul=2 if d['flags']&1 else 1
                pal=np.frombuffer(r,dtype='<u2',count=256,offset=A['pump_lut']-rvi.RAMB)
                v=pal[np.frombuffer(r,dtype='u1',count=cnt,offset=s)].reshape(d['n'],d['h']).T
                sur[d['y']:d['y']+d['h'],d['x']:d['x']+d['n']*mul]=np.repeat(v,mul,axis=1);cov[d['y']:d['y']+d['h'],d['x']:d['x']+d['n']*mul]=True;n+=1
            seen+=1
        if len(t)>1<<16: del t[:seen];seen=0
    return dict(strips=n,covered=int(cov.sum()),frame_sha256=hashlib.sha256(sur.tobytes()).hexdigest(),halted=c.halted,fault=c.fault,steps=c.steps,unique=int(len(np.unique(sur))))
raw=open(E0,'rb').read();res=dict(base=frame(raw))
print(res,flush=True)
# NOR data: first PT_LOAD at DROM above 0x02000000 + rodata; flip bits every 251 bytes across the back half
ph=struct.unpack_from('<I',raw,28)[0];ps,pn=struct.unpack_from('<HH',raw,42)
seg=max((struct.unpack_from('<8I',raw,ph+i*ps) for i in range(pn)),key=lambda e:e[4] if 0x02000000<=e[2]<0x02400000 else 0)
b=bytearray(raw);off,size=seg[1],seg[4]
for z in range(off+size//2,off+size,251): b[z]^=0x04
res['mutated_game_data']=frame(bytes(b));print(res['mutated_game_data'],flush=True)
for frac,step in ((0.97,4099),(0.9,8191),(0.8,16381),(0.6,16381)):
    b=bytearray(raw)
    for z in range(off+int(size*frac),off+size,step): b[z]^=0x04
    k=f'mutated_from_{frac}_step_{step}';res[k]=frame(bytes(b));print(k,res[k],flush=True)
json.dump(res,open('experiments/audit/results/doom-frame.json','w'),indent=1)
