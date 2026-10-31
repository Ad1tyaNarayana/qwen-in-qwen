"""Compile one model-independent runtime; link an image without code generation."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0,str(ROOT/'experiments/dmc-rv32'))
from elf_info import inspect


def bindings():
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [HERE/'runtime.c',HERE/'runtime.h',ROOT/'experiments/neural-accelerator/ndot.h']}


def build(out, image):
    out = out.resolve()
    out.mkdir(parents=True,exist_ok=True)
    zig = Path(__import__('ziglang').__file__).parent / ('zig.exe' if sys.platform == 'win32' else 'zig')
    flags = ['-O2','-fno-unroll-loops','-fno-vectorize','-fno-slp-vectorize','-fwrapv','-std=c11',
             '-I'+str(ROOT/'experiments/neural-accelerator')]
    subprocess.run(['clang',*flags,'-DNATIVE',str(HERE/'runtime.c'),'-o',str(out/'native')],check=True)
    target = [str(zig),'cc','-target','riscv32-freestanding','-mcpu=generic_rv32+m','-mabi=ilp32']
    compile_flags = [*flags,'-ffreestanding','-fno-builtin','-ffunction-sections','-fdata-sections']
    subprocess.run([*target,*compile_flags,'-c',str(HERE/'runtime.c'),'-o',str(out/'runtime.o')],check=True)
    subprocess.run([str(zig),'build-lib',str(zig.parent/'lib/compiler_rt.zig'),'-target','riscv32-freestanding',
                    '-mcpu','generic_rv32+m','-O','ReleaseSmall','-ffunction-sections','-fdata-sections',
                    '-femit-bin='+str(out/'compiler_rt.a')],check=True)
    linker = out/'link.ld'
    linker.write_text('STACK_BYTES = 1024;\nRAM_MARGIN = 1024;\n'+
                      (ROOT/'experiments/tiny-llm-rv32/link.ld').read_text())
    (out/'runtime-build.json').write_text(json.dumps(dict(sources=bindings(),
        object_sha256=hashlib.sha256((out/'runtime.o').read_bytes()).hexdigest(),
        native_sha256=hashlib.sha256((out/'native').read_bytes()).hexdigest()),indent=2)+'\n')
    return link(out,image)


def link(out,image):
    out, image = out.resolve(), image.resolve()
    binding = json.loads((out/'runtime-build.json').read_text())
    assert binding['sources'] == bindings(), 'Runtime sources changed; rebuild the runtime object'
    assert binding['object_sha256'] == hashlib.sha256((out/'runtime.o').read_bytes()).hexdigest()
    assert binding['native_sha256'] == hashlib.sha256((out/'native').read_bytes()).hexdigest()
    zig = Path(__import__('ziglang').__file__).parent / ('zig.exe' if sys.platform == 'win32' else 'zig')
    assembly = image/'model.S'
    assembly.write_text('.section .model,"a",@progbits\n.global model_blob\nmodel_blob:\n.incbin '+
                        json.dumps(str(image/'model.bin'))+'\n')
    command = [str(zig),'cc','-target','riscv32-freestanding','-mcpu=generic_rv32+m','-mabi=ilp32',
               '-ffreestanding','-nostdlib','-static','-Wl,--gc-sections','-Wl,-T,'+str(out/'link.ld'),
               str(out/'runtime.o'),str(ROOT/'experiments/tiny-llm-rv32/start.S'),str(assembly),
               str(out/'compiler_rt.a'),'-o',str(image/'engine.elf')]
    subprocess.run(command,check=True)
    info = inspect(image/'engine.elf')
    static = info['symbols']['__scratch_end']-0x01000100
    assert 65536-256-static-1024 >= 1024
    meta = json.loads((image/'pack.json').read_text())
    meta.update(elf=info,static_bytes=static,stack_reserved=1024,
                linked_margin=65536-256-static-1024,native=str(out/'native'),link_command=command,
                runtime_binding=binding,
                runtime_sha256=hashlib.sha256((HERE/'runtime.c').read_bytes()).hexdigest())
    (image/'build.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(dict(image=str(image),static_bytes=static,margin=meta['linked_margin'])))
    return meta


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--image',type=Path,required=True)
    p.add_argument('--link-only',action='store_true')
    a = p.parse_args()
    (link if a.link_only else build)(a.out,a.image)
