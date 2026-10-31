#!/usr/bin/env python3
"""Build the identical engine for the host and RV32IM without system installs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'experiments/dmc-rv32'))
from elf_info import inspect
from pack import pack, RAM_MARGIN


def encode(text, tokenizer):
    vocab = [bytes(p) for p in tokenizer['pieces']]
    lookup = {s: i for i, s in enumerate(vocab)}
    tokens = [1]
    if text:
        tokens.append(lookup[b' '])
    for char in text:
        b = char.encode()
        tokens.extend([lookup[b]] if b in lookup else [c + 3 for c in b])
    while True:
        best = None
        for i in range(1, len(tokens) - 1):
            merged = lookup.get(vocab[tokens[i]] + vocab[tokens[i + 1]])
            if merged is not None and (best is None or tokenizer['scores'][merged] > best[0]):
                best = tokenizer['scores'][merged], i, merged
        if best is None:
            return tokens
        _, i, merged = best
        tokens[i:i + 2] = [merged]


def build(out, prompt, steps, context=64, zig=None, model=None, tokenizer=None,
          kv_bits=16, kv_frac=10, stack_bytes=1024):
    out = out.resolve()
    downloads = ROOT / '.cache/tiny-llm/downloads/stories260K'
    meta = pack(model or downloads / 'stories260K.bin', tokenizer or downloads / 'tok512.bin',
                out, context, kv_bits, kv_frac, stack_bytes)
    tokens = encode(prompt, json.loads((out / 'tokenizer.json').read_text()))
    if not len(tokens) <= steps <= context:
        raise ValueError('Require prompt length <= steps <= context')
    (out / 'workload.h').write_text(f'#define RUN_STEPS {steps}\n#define PROMPT_LEN {len(tokens)}\n'
        'static const int prompt_tokens[] = {' + ','.join(map(str, tokens)) + '};\n')
    zig = zig or Path(__import__('ziglang').__file__).parent / ('zig.exe' if sys.platform == 'win32' else 'zig')
    if zig is None or not Path(zig).is_file():
        raise FileNotFoundError('Missing zig; pass --zig pointing to an existing Zig binary')
    common = ['-O2', '-fno-unroll-loops', '-fno-vectorize', '-fno-slp-vectorize',
              '-fwrapv', '-std=c11', '-I' + str(out)]
    commands = [['clang', *common, '-DNATIVE', str(HERE / 'engine.c'), '-o', str(out / 'native')]]
    assembly = out / 'model.S'
    assembly.write_text('.section .model,"a",@progbits\n.global model_blob\nmodel_blob:\n.incbin '
                        + json.dumps(str(out / 'model.bin')) + '\n')
    linker = out / 'link.ld'
    linker.write_text(f'STACK_BYTES = {stack_bytes};\nRAM_MARGIN = {RAM_MARGIN};\n' + (HERE / 'link.ld').read_text())
    commands.append([str(zig), 'cc', '-target', 'riscv32-freestanding', '-mcpu=generic_rv32+m',
        '-mabi=ilp32', *common, '-ffreestanding', '-fno-builtin', '-nostdlib', '-static',
        '-ffunction-sections', '-fdata-sections', '-Wl,--gc-sections',
        '-Wl,-T,' + str(linker),
        str(HERE / 'engine.c'), str(HERE / 'start.S'), str(assembly), '-o', str(out / 'engine.elf')])
    runtime = out / 'compiler_rt.a'
    commands.insert(1, [str(zig), 'build-lib', str(Path(zig).parent / 'lib/compiler_rt.zig'),
        '-target', 'riscv32-freestanding', '-mcpu', 'generic_rv32+m', '-O', 'ReleaseSmall',
        '-ffunction-sections', '-fdata-sections', '-femit-bin=' + str(runtime)])
    commands[-1].insert(-2, str(runtime))
    for command in commands:
        subprocess.run(command, check=True)
    info = inspect(out / 'engine.elf')
    static_bytes = info['symbols']['__scratch_end'] - 0x01000100
    margin = 65536 - 256 - static_bytes - stack_bytes
    if margin < RAM_MARGIN:
        raise ValueError(f'RAM margin {margin} is below {RAM_MARGIN} bytes')
    result = dict(pack=meta, prompt=prompt, prompt_tokens=tokens, steps=steps,
                  commands=commands, elf=info,
                  ram=dict(static_bytes=static_bytes, bottom_reserved=256, stack_reserved=stack_bytes,
                           total_reserved=256+static_bytes+stack_bytes, margin_bytes=margin),
                  compiler_versions={name:subprocess.check_output(command, text=True).strip()
                      for name,command in [('native',['clang','--version']),('zig',[str(zig),'version'])]},
                  engine_sha256=hashlib.sha256((HERE / 'engine.c').read_bytes()).hexdigest(),
                  native_sha256=hashlib.sha256((out / 'native').read_bytes()).hexdigest())
    (out / 'build.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=ROOT / '.cache/tiny-llm/build')
    p.add_argument('--prompt', default='Once upon a time')
    p.add_argument('--steps', type=int, default=64)
    p.add_argument('--context', type=int, default=64)
    p.add_argument('--zig', type=Path)
    p.add_argument('--model', type=Path)
    p.add_argument('--tokenizer', type=Path)
    p.add_argument('--kv-bits', type=int, choices=[8, 16], default=16)
    p.add_argument('--kv-frac', type=int, default=10)
    p.add_argument('--stack-bytes', type=int, default=1024)
    a = p.parse_args()
    print(json.dumps(build(a.out, a.prompt, a.steps, a.context, a.zig, a.model, a.tokenizer,
                           a.kv_bits, a.kv_frac, a.stack_bytes), indent=2))
