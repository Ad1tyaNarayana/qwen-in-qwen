#!/usr/bin/env python3
"""Observe the existing RV32 record machine without supplying guest state."""
import argparse
from collections import Counter
import functools
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'experiments/dmc-rv32'))
from elf_info import inspect


def run(out, max_instructions):
    out = out.resolve()
    build = json.loads((out / 'build.json').read_text())
    info = inspect(out / 'engine.elf')
    if info['sha256'] != build['elf']['sha256']:
        raise ValueError('ELF changed after build')
    spec = importlib.util.spec_from_file_location('prepare', ROOT / 'experiments/memory-map/prepare.py')
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
    scratch = out / ('machine-' + str(time.time_ns()))
    manifest = prep.prepare(ROOT / '.upstream/qwenputer', scratch, 64, 4096)
    sys.path.insert(0, str(scratch))
    from chipformer.rv import ref, elf, spec as S
    ref.decode = functools.lru_cache(None)(ref.decode)
    m = ref.Machine.from_program(elf.program(out / 'engine.elf'))
    syms = info['symbols']
    dot_begin = syms['dot_loop_begin']
    dot_branches = []
    for pc in range(dot_begin, dot_begin + 128, 4):
        ins = ref.decode(m.rom[(pc - S.ROM_BASE) // 4])
        if ins.op in ref.BRANCH_OPS and pc + ins.imm == dot_begin:
            dot_branches.append(pc)
    if len(dot_branches) != 1:
        raise ValueError('Cannot identify the complete compiled matvec inner loop')
    dot_end = dot_branches[0] + 4
    native = subprocess.check_output([str(out / 'native'), str(out / 'model.bin')])
    positions = []
    mix, totalmix = Counter(), Counter()
    tokens, dot_ins, dot_tokens, mat_ins, mat_tokens = 0, 0, 0, 0, 0
    start_i = start_t = 0
    active = False
    mat_return = None
    min_sp = S.SP_INIT
    started = time.monotonic()
    last = 0
    while not m.halted and m.steps < max_instructions:
        refresh = m.n == S.N_MAX
        pc = m.pc
        if not refresh:
            if mat_return == pc:
                mat_return = None
            if pc == syms['matvec']:
                mat_return = m.x[1]
            if pc == syms['position_begin']:
                active = True
                start_i, start_t = m.steps, tokens
                mix = Counter()
                dot_ins = dot_tokens = mat_ins = mat_tokens = 0
            if pc == syms['position_end']:
                active = False
                positions.append(dict(position=m.x[10], phase='prefill' if m.x[10] < len(build['prompt_tokens']) - 1 else 'decode',
                    instructions=m.steps-start_i, trace_tokens=tokens-start_t,
                    opcodes=dict(mix), dot_body_instructions=dot_ins, dot_body_tokens=dot_tokens,
                    matvec_instructions=mat_ins, matvec_tokens=mat_tokens))
                (out / 'progress.json').write_text(json.dumps(positions, indent=2))
                print(f"position {m.x[10]} instructions {m.steps-start_i} tokens {tokens-start_t}", flush=True)
        op = None if refresh else ref.decode(m.rom[(pc - S.ROM_BASE) // 4]).op
        record = m.record()
        count = len(record)
        tokens += count
        min_sp = min(min_sp, m.x[2])
        if op:
            totalmix[op] += 1
            if active:
                mix[op] += 1
        if active and mat_return is not None:
            mat_ins += int(not refresh)
            mat_tokens += count
        if active and dot_begin <= pc < dot_end:
            dot_ins += int(not refresh)
            dot_tokens += count
        if m.steps - last >= 5000000:
            last = m.steps
            print(f'{m.steps} retired; {time.monotonic()-started:.1f}s', flush=True)
    (out / 'guest.txt').write_bytes(m.out)
    complete = m.halted and m.exit_code == 0 and m.fault is None
    result = dict(kind='Existing software RV32 reference only; no neural execution',
        complete=complete, bit_exact=complete and bytes(m.out) == native,
        positions=positions, instructions=m.steps, trace_tokens=tokens,
        count_scope='record() lengths including refresh; excludes prefill/GO; position intervals include begin marker and exclude end marker and byte output',
        opcodes=dict(totalmix), exit_code=m.exit_code, fault=m.fault,
        elapsed_seconds=time.monotonic()-started,
        minimum_sp=min_sp, stack_bytes=S.SP_INIT-min_sp,
        scratch_end=syms['__scratch_end'], stack_clearance=min_sp-syms['__scratch_end'],
        dot_loop_pc_range=[dot_begin, dot_end],
        elf_sha256=info['sha256'], native_output_sha256=hashlib.sha256(native).hexdigest(),
        guest_output_sha256=hashlib.sha256(m.out).hexdigest(), prepare_manifest=manifest)
    (out / 'reference.json').write_text(json.dumps(result, indent=2) + '\n')
    if not result['bit_exact'] or result['stack_clearance'] < 0:
        raise SystemExit('Reference gate failed; inspect reference.json and guest.txt')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=ROOT / '.cache/tiny-llm/build')
    p.add_argument('--max-instructions', type=int, default=1000000000)
    a = p.parse_args()
    run(a.out, a.max_instructions)
