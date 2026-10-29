#!/usr/bin/env python3
"""Native quality comparison, with both free-running and common-history argmax."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
from build import build, ROOT, HERE


def records(raw):
    return [tuple(map(int, m)) for m in re.findall(rb'^@(\d+) (\d+) (\d+) (\d+) (\d+)\n', raw, re.M)]


def text(raw):
    chunks = re.split(rb'^@\d+ \d+ \d+ \d+ \d+\n', raw, flags=re.M)[1:]
    assert all(c.startswith(b'>') and c.endswith(b'\n') for c in chunks)
    return b''.join(c[1:-1] for c in chunks).decode(errors='replace')


def verify(out, steps):
    out.mkdir(parents=True, exist_ok=True)
    source = ROOT / '.cache/tiny-llm/downloads/llama2.c/run.c'
    shutil.copyfile(source, out / 'run.c')
    shutil.copyfile(source.with_name('LICENSE'), out / 'LICENSE')
    commands = [
        ['clang', '-O2', '-Wno-unknown-pragmas', '-I'+str(out), str(HERE / 'float_probe.c'), '-lm', '-o', str(out / 'float-probe')],
        ['clang', '-O2', '-Wno-unknown-pragmas', str(out / 'run.c'), '-lm', '-o', str(out / 'float-original')]]
    for command in commands:
        subprocess.run(command, check=True)
    models = ROOT / '.cache/tiny-llm/downloads/stories260K'
    cases = []
    for i, prompt in enumerate(['Once upon a time', 'One day, a little girl', 'There was a big dog']):
        case = out / f'case-{i}'
        meta = build(case, prompt, steps)
        integer = subprocess.check_output([str(case / 'native'), str(case / 'model.bin')])
        args = [str(models / 'stories260K.bin'), str(models / 'tok512.bin'), prompt, str(steps)]
        floating = subprocess.check_output([str(out / 'float-probe'), *args])
        ir, fr = records(integer), records(floating)
        (case / 'history.txt').write_text('\n'.join(str(r[1]) for r in ir))
        teacher = subprocess.check_output([str(out / 'float-probe'), *args, str(case / 'history.txt')])
        tr = records(teacher)
        original = subprocess.run([str(out / 'float-original'), args[0], '-z', args[1], '-i', prompt,
            '-t', '0', '-n', str(steps)], check=True, capture_output=True).stdout
        for name, raw in [('integer', integer), ('float', floating), ('teacher', teacher), ('original', original)]:
            (case / (name + '.txt')).write_bytes(raw)
        n = len(meta['prompt_tokens']) - 1
        assert len(ir) == len(fr) == len(tr) == steps
        assert [r[1] for r in ir[:n+1]] == [r[1] for r in fr[:n+1]], 'Prompt tokenizer differs'
        assert [r[1] for r in ir] == [r[1] for r in tr], 'Teacher history differs'
        assert original == text(floating).encode() + b'\n', 'Unmodified float CLI and probe text differ'
        same = sum(a[2] == b[2] for a, b in zip(ir[n:], fr[n:]))
        teacher_same = sum(a[2] == b[2] for a, b in zip(ir[n:], tr[n:]))
        first = next((a[0] for a, b in zip(ir[n:], fr[n:]) if a[2] != b[2]), None)
        cases.append(dict(prompt=prompt, prompt_tokens=meta['prompt_tokens'],
            generated_predictions=steps-n, free_running_matches=same, free_running_agreement=same/(steps-n),
            common_history_matches=teacher_same, common_history_agreement=teacher_same/(steps-n),
            first_divergence_position=first, integer_text=text(integer), float_text=text(floating),
            original_cli_text=original.decode(errors='replace'), kv_clips=ir[-1][4],
            integer_records=ir, float_records=fr, teacher_records=tr,
            elf_sha256=meta['elf']['sha256'], rom_sha256=meta['pack']['rom_sha256']))
    first = out / 'case-0'
    sanitizer_command = ['clang', '-O1', '-fno-asynchronous-unwind-tables', '-fno-unwind-tables',
        '-fno-unroll-loops', '-fno-vectorize', '-fno-slp-vectorize', '-fwrapv', '-std=c11',
        '-I'+str(first), '-DNATIVE', '-fsanitize=undefined,address', str(HERE/'engine.c'),
        '-o', str(out/'native-sanitized')]
    subprocess.run(sanitizer_command, check=True)
    sanitized = subprocess.run([str(out/'native-sanitized'), str(first/'model.bin')],
        check=True, capture_output=True)
    assert not sanitized.stderr and sanitized.stdout == (first/'integer.txt').read_bytes()
    result = dict(kind='Native integer vs original llama2.c float, greedy argmax, three prompts',
        steps=steps, cases=cases, float_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        float_commands=commands,
        sanitizer=dict(passed=True, command=sanitizer_command,
            output_sha256=hashlib.sha256(sanitized.stdout).hexdigest()),
        free_running_agreement=sum(c['free_running_matches'] for c in cases)/sum(c['generated_predictions'] for c in cases),
        common_history_agreement=sum(c['common_history_matches'] for c in cases)/sum(c['generated_predictions'] for c in cases))
    (out / 'quality.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=ROOT / '.cache/tiny-llm/quality')
    p.add_argument('--steps', type=int, default=64)
    a = p.parse_args()
    r = verify(a.out.resolve(), a.steps)
    print(json.dumps({k:v for k,v in r.items() if k != 'cases'}, indent=2))
