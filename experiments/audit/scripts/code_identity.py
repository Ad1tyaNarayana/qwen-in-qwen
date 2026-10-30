#!/usr/bin/env python3
import hashlib,struct,glob,json
def ex(p):
    raw=open(p,'rb').read();ph=struct.unpack_from('<I',raw,28)[0];ps,pn=struct.unpack_from('<HH',raw,42);o=[]
    for i in range(pn):
        k,off,addr,_,sz,msz,fl,_=struct.unpack_from('<8I',raw,ph+i*ps)
        if k==1 and fl&1:o.append((addr,raw[off:off+sz]))
    return hashlib.sha256(repr([(a,b) for a,b in o]).encode()).hexdigest(),sum(len(b) for a,b in o)
for p in sorted(glob.glob('.cache/tiny-qwen35/final-safe/*/engine.elf')+['.cache/tiny-qwen35/trained-program/engine.elf']+glob.glob('.cache/tiny-qwen35/final-untied*/engine.elf')+['.cache/tiny-llm/ndot-story38-20261030/engine.elf']):
    print(ex(p),p)
