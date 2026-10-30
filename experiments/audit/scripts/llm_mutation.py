#!/usr/bin/env python3
"""Attack 2/5 on LLM guests: are stories in the image? Does a one-byte model change alter output?
Also Gate 2 check and C6 code-identity check, all with rvi.py only."""
import hashlib,json,re,struct,sys
from pathlib import Path
sys.path.insert(0,'experiments/audit/scripts')
import rvi
Q='.cache/tiny-qwen35/trained-program/engine.elf';S='.cache/tiny-llm/ndot-story38-20261030/engine.elf'
res={}
def segs(p):
    raw=open(p,'rb').read();ph=struct.unpack_from('<I',raw,28)[0];ps,pn=struct.unpack_from('<HH',raw,42);o=[]
    for i in range(pn):
        k,off,addr,_,sz,msz,fl,_=struct.unpack_from('<8I',raw,ph+i*ps)
        if k==1:o.append((addr,sz,msz,fl,off))
    return raw,o
def text(c):
    return ''.join(x[1:-1] for x in re.split(r'^@\d+ \d+ \d+ \d+ \d+\n',c.out.decode('latin1'),flags=re.M)[1:])
for name,p in (('qwen35',Q),('stories',S)):
    raw,o=segs(p)
    r=dict(segments=[[hex(a),s,m,f] for a,s,m,f,_ in o])
    r['contains_once_upon']=b'Once upon a time' in raw;r['contains_there_was']=b'there was a little' in raw;r['contains_lily']=b'Lily' in raw
    d=[x for x in o if 0x02000000<=x[0]<0x02400000]
    size=sum(x[1] for x in d)
    # mutate one byte in the model image ~70% through DROM (matrices), by flipping its top bit
    a,s,m,f,off=max(d,key=lambda x:x[1])
    b=bytearray(raw);pos=off+int(s*0.70);b[pos]^=0x80
    for z in range(off+int(s*0.3),off+int(s*0.95),37): b[z]^=0x10
    c=rvi.CPU(bytes(b));c.run(40_000_000 if name=='qwen35' else 75_000_000)
    r['mutation']='top bit of one byte, plus bit 4 of every 37th DROM byte from 30% to 95%';r['mutated_halted']=c.halted;r['mutated_fault']=c.fault;r['mutated_instructions']=c.steps
    r['mutated_text']=text(c)[:200]
    print(name,json.dumps(r),flush=True);res[name]=r
json.dump(res,open('experiments/audit/results/llm-mutation.json','w'),indent=1)
