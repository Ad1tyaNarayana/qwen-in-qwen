#!/usr/bin/env python3
"""Build fault guests and mutated guests; every ELF goes in its own fixture dir with an engine.elf."""
import shutil,struct,subprocess,sys,hashlib
from pathlib import Path
A=Path(__file__).resolve().parents[1];R=A.parents[1];C=R/'.cache/audit/guests'
sys.path.insert(0,str(A/'scripts'))
import rvi
zig=next((R/'.venv/lib').glob('python*/site-packages/ziglang/zig'))
def put(n,b):
    (C/n).mkdir(parents=True,exist_ok=True);(C/n/'engine.elf').write_bytes(b);return C/n
for i in range(1,8):
    (C/f'fault{i}').mkdir(parents=True,exist_ok=True)
    subprocess.run([str(zig),'cc','-target','riscv32-freestanding','-mcpu=generic_rv32+m','-mabi=ilp32',f'-DCASE={i}','-nostdlib','-static','-Wl,-T,'+str(A/'fault/link.ld'),str(A/'fault/f.S'),'-o',str(C/f'fault{i}/engine.elf')],check=True)
fresh=(R/'.cache/audit/fresh/engine.elf').read_bytes()
put('fresh',fresh)
# mutate one bit of the DROM blob at the sweep's t==3 span (last four blob bytes).
blob=(R/'.cache/audit/fresh/blob.bin').read_bytes()
i=fresh.rfind(blob[-64:]);assert i>0
m=bytearray(fresh);m[i+64-2]^=0x08;put('fresh-mut',bytes(m))
sha=(R/'.cache/showcase/sha256/engine.elf').read_bytes();put('sha256',sha)
i=sha.find(b'abc\0');assert i>0 and sha.count(b'abc\0')==1
m=bytearray(sha);m[i+1]^=0x01;put('sha256-mut',bytes(m))  # 'abc' -> 'acc'
for n in sorted(p.name for p in C.iterdir()):
    c=rvi.CPU(str(C/n/'engine.elf')).run()
    print(n,c.steps,len(c.tokens),c.ndots,c.fault,repr(c.out[:90]),getattr(c,'exit',None),hashlib.sha256(c.ram).hexdigest()[:12])
