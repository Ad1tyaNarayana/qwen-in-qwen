#!/usr/bin/env python3
"""Attack 2: is any demo's output (or a long substring / digest) stored in its ELF, in raw form?"""
import json,re,sys,hashlib
from pathlib import Path
R=Path('.').resolve()
def sub_hits(out,img,n):
    hits=0;spots=[]
    for i in range(0,len(out)-n+1):
        s=out[i:i+n]
        if s in img: hits+=1;spots.append(s)
    return hits,spots[:3]
res={}
for name,d in [('sha256','.cache/showcase/sha256'),('life','.cache/showcase/life'),('tictactoe','.cache/showcase/tictactoe'),('mnist','.cache/showcase/mnist'),('mnist-scalar','.cache/showcase/mnist-scalar')]:
    d=R/d
    elf=(d/'engine.elf').read_bytes();out=(d/'reference-output.bin').read_bytes()
    res[name]=dict(elf_bytes=len(elf),output_bytes=len(out),full_output_in_elf=out in elf,hits12=sub_hits(out,elf,12))
    # digests binary form
    for m in re.findall(rb'\b[0-9a-f]{64}\b',out):
        res[name].setdefault('digest_binary_in_elf',[]).append(bytes.fromhex(m.decode()) in elf)
print(json.dumps(res,indent=1,default=str))
