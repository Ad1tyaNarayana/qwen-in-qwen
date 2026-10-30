#!/usr/bin/env python3
import hashlib,subprocess,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[2]
out=R/'.cache/audit'/('fresh-safe' if '--safe' in sys.argv else 'fresh');out.mkdir(parents=True,exist_ok=True)
b=bytearray();i=0
while len(b)<4000000:
    b+=hashlib.sha256(b'audit-blob-%d'%i).digest();i+=1
(out/'blob.bin').write_bytes(b[:4000000])
zig=next((R/'.venv/lib').glob('python*/site-packages/ziglang/zig'))
subprocess.run([str(zig),'cc','-target','riscv32-freestanding','-mcpu=generic_rv32+m','-mabi=ilp32','-O2','-fno-unroll-loops','-fno-vectorize','-fno-slp-vectorize','-fwrapv','-std=c11','-ffreestanding','-fno-builtin','-nostdlib','-static','-ffunction-sections','-fdata-sections','-Wl,--gc-sections','-Wl,-T,'+str(H/'link.ld'),'-I'+str(R/'experiments/neural-accelerator'),'-I'+str(out),'-Wa,-I'+str(out),str(H/('main_safe.c' if '--safe' in sys.argv else 'main.c')),str(H/'start.S'),str(H/'blob.S'),'-o',str(out/'engine.elf')],check=True,cwd=out)
print(hashlib.sha256((out/'engine.elf').read_bytes()).hexdigest())
