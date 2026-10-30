"""Build a guest and native executable with the shared fixed memory map."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'experiments/dmc-rv32'))
from elf_info import inspect


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(demo, scalar=False):
    name = demo + ('-scalar' if scalar else '')
    out = ROOT / '.cache/showcase' / name
    out.mkdir(parents=True, exist_ok=True)
    source = HERE / demo / 'main.c'
    if demo == 'tictactoe':
        moves = json.loads((HERE / demo / 'moves.json').read_text())
        if not moves or any(type(x) is not int or not 0 <= x < 9 for x in moves):
            raise ValueError('Moves must be square indices 0..8')
        (out / 'moves.h').write_text(f'#define MOVE_COUNT {len(moves)}\nstatic const unsigned char moves[]={{'+','.join(map(str,moves))+'};\n')
    if demo == 'mnist':
        import shutil
        shutil.copyfile(ROOT / '.cache/showcase/mnist-model/model.h', out / 'model.h')
    zig = next((ROOT / '.venv/lib').glob('python*/site-packages/ziglang/zig'))
    common = ['-O2','-fno-unroll-loops','-fno-vectorize','-fno-slp-vectorize','-fwrapv','-std=c11',
              '-I'+str(HERE),'-I'+str(out),'-I'+str(ROOT/'experiments/neural-accelerator')]
    if scalar:
        common += ['-DSCALAR']
    commands = [ ['clang',*common,'-DNATIVE',str(source),'-o',str(out/'native')],
        [str(zig),'cc','-target','riscv32-freestanding','-mcpu=generic_rv32+m','-mabi=ilp32',
         *common,'-ffreestanding','-fno-builtin','-nostdlib','-static','-ffunction-sections','-fdata-sections',
         '-Wl,--gc-sections','-Wl,-T,'+str(out/'link.ld'),str(source),
         str(ROOT/'experiments/tiny-llm-rv32/start.S'),'-o',str(out/'engine.elf')] ]
    (out/'link.ld').write_text('STACK_BYTES = 8192;\nRAM_MARGIN = 1024;\n'+(ROOT/'experiments/tiny-llm-rv32/link.ld').read_text())
    if demo == 'mnist':
        runtime=out/'compiler_rt.a'
        subprocess.run([str(zig),'build-lib',str(zig.parent/'lib/compiler_rt.zig'),'-target','riscv32-freestanding','-mcpu','generic_rv32+m','-O','ReleaseSmall','-ffunction-sections','-fdata-sections','-femit-bin='+str(runtime)],check=True)
        commands[-1].insert(-2,str(runtime))
    for cmd in commands:
        subprocess.run(cmd,check=True)
    raw = (out/'engine.elf').read_bytes()
    phoff = struct.unpack_from('<I',raw,28)[0]
    phsize, phnum = struct.unpack_from('<HH',raw,42)
    drom = bytearray(4*1024*1024)
    for i in range(phnum):
        kind,off,addr,_,size,memsize,_,_ = struct.unpack_from('<8I',raw,phoff+i*phsize)
        if kind==1 and 0x02000000 <= addr < 0x02400000:
            assert addr+memsize <= 0x02400000
            drom[addr-0x02000000:addr-0x02000000+size] = raw[off:off+size]
    (out/'drom.bin').write_bytes(drom)
    meta = dict(demo=name,elf=inspect(out/'engine.elf'),elf_sha256=sha(out/'engine.elf'),
        drom_sha256=sha(out/'drom.bin'),source_sha256=sha(source),commands=commands,
        memory_map=dict(ram=65536,drom=4194304,code=262144),prompt_tokens=[],steps=1)
    (out/'build.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(out)
    return out


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('demo',choices=['sha256','life','tictactoe','mnist'])
    p.add_argument('--scalar',action='store_true')
    a=p.parse_args()
    build(a.demo,a.scalar)
