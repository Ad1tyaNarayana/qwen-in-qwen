#!/usr/bin/env python3
"""Random RV32IM + ndot guests in straight-line form with forward branches. Seeds chosen by the auditor."""
import random,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parents[3];A=R/'experiments/audit'
zig=next((R/'.venv/lib').glob('python*/site-packages/ziglang/zig'))
RT='add sub sll slt sltu xor srl sra or and mul mulh mulhsu mulhu div divu rem remu'.split()
IT='addi slti sltiu xori ori andi'.split()
def reg(r,z=True): return 'x0' if z and r.random()<.04 else f'x{r.randint(1,25)}'
def special(r): return r.choice([0,1,-1,2,0x7fffffff,-0x80000000,0x80000000-2,0xffff,0x10000,-0x8000,0x7fff,0x3fff,0x4000,-0x4000,-0x4001,0x1fffffff,0x20000000,-0x20000000,-0x20000001,r.getrandbits(32)])
def gen(seed,n):
    r=random.Random(seed);o=['.section .text.init,"ax",@progbits','.global _start','_start:','.option norvc','.option norelax']
    o+=['li x31, 0x01004000','li x30, 0x02000000']
    for i in range(1,26): o.append(f'li x{i}, {special(r)&0xffffffff}')
    l=0
    for k in range(n):
        c=r.random()
        if c<.35: o.append(f'{r.choice(RT)} {reg(r)}, {reg(r)}, {reg(r)}')
        elif c<.50: o.append(f'{r.choice(IT)} {reg(r)}, {reg(r)}, {r.randint(-2048,2047)}')
        elif c<.56: o.append(f'{r.choice(["slli","srli","srai"])} {reg(r)}, {reg(r)}, {r.randint(0,31)}')
        elif c<.60: o.append(f'{r.choice(["lui","auipc"])} {reg(r)}, {r.getrandbits(20)}')
        elif c<.72:
            op,sz=r.choice([('lb',1),('lbu',1),('lh',2),('lhu',2),('lw',4)])
            base=r.choice(['x31','x30']);imm=r.randrange(-2000 if base=='x31' else 0,2000)//sz*sz
            o.append(f'{op} {reg(r)}, {imm}(x{base[1:]})')
        elif c<.82:
            op,sz=r.choice([('sb',1),('sh',2),('sw',4)]);o.append(f'{op} {reg(r)}, {r.randrange(-2000,2000)//sz*sz}(x31)')
        elif c<.90:
            o+= [f'andi x28, {reg(r)}, 0x7f8',f'slli x28, x28, 3','add x28, x28, x31']
            o+= [f'andi x29, {reg(r)}, 0x7fc',f'slli x29, x29, {r.randint(0,9)}','add x29, x29, x30']
            rd=reg(r);o.append(f'.insn r 0x0b, 0, 0x20, {rd}, x29, x28')
        elif c<.96:
            l+=1;o.append(f'{r.choice(["beq","bne","blt","bge","bltu","bgeu"])} {reg(r)}, {reg(r)}, L{l}');o.append(f'{r.choice(RT)} {reg(r)}, {reg(r)}, {reg(r)}');o.append(f'L{l}:')
        elif c<.98:
            l+=1;o.append(f'jal {reg(r)}, L{l}');o.append(f'addi {reg(r)}, x0, 1');o.append(f'L{l}:')
        else:
            o+=['auipc x27, 0',f'jalr {reg(r)}, x27, 13','addi x26, x26, 1']
    o+=['li a7, 1']+[f'mv a0, x{i}\necall\nsrli a0, x{i}, 8\necall' for i in range(1,8)]+['li a0, 0','li a7, 93','ecall']
    o+=['.section .rodata','d: .incbin "fuzz-blob.bin"','.section .data','.balign 4','b:']
    o.append('.fill 7500, 4, 0x5a17c3e9')
    return '\n'.join(o)+'\n'
C=R/'.cache/audit/guests'
b=random.Random(77);(R/'.cache/audit/fuzz-blob.bin').write_bytes(bytes(b.getrandbits(8) for _ in range(2**21)))
for s,n in ((1101,30000),(2202,30000),(3303,30000)):
    d=C/f'fuzz{s}';d.mkdir(parents=True,exist_ok=True);(d/'f.S').write_text(gen(s,n))
    subprocess.run([str(zig),'cc','-target','riscv32-freestanding','-mcpu=generic_rv32+m','-mabi=ilp32','-nostdlib','-static','-Wl,-T,'+str(A/'fresh/link.ld'),'-Wa,-I'+str(R/'.cache/audit'),str(d/'f.S'),'-o',str(d/'engine.elf')],check=True,cwd=R/'.cache/audit')
