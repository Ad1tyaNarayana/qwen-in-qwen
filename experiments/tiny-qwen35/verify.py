"""Reproducible native legacy parity and immutable code-ROM checks."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess

from pack import llama, ROOT, OLD
from build import build, link


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def code(path):
    raw = path.read_bytes()
    header = struct.unpack_from('<16sHHIIIIIHHHHHH',raw)
    result = []
    for i in range(header[10]):
        entry = struct.unpack_from('<8I',raw,header[5]+i*header[9])
        kind, offset, vaddr, _, filesz, memsz, flags, _ = entry
        if kind == 1 and flags & 1:
            result.append((vaddr,memsz,raw[offset:offset+filesz]))
    assert result
    return result


def records(raw):
    return [list(map(int,r)) for r in re.findall(rb'^@(\d+) (\d+) (\d+) (\d+) (\d+)\n',raw,re.M)]


def legacy(out):
    out.mkdir(parents=True,exist_ok=True)
    cases = []
    models = ROOT/'.cache/tiny-llm'
    inputs = [('stories',models/'downloads/stories260K/stories260K.bin',
               models/'downloads/stories260K/tok512.bin',64,
               ['Once upon a time','One day, a little girl','There was a big dog']),
              ('custom',models/'custom/model.bin',models/'custom/tok1024.bin',129,
               ['Once upon a time, there was a little','Lily found a small bird in the garden.',
                'Tom was sad because he lost his toy.'])]
    baseline_code = None
    for name,model,tok,context,prompts in inputs:
        for index,prompt in enumerate(prompts):
            target = out/f'{name}-{index}'
            meta = llama(model,tok,target,context,prompt,context)
            runtime = out/'runtime'
            compiled = link(runtime,target) if (runtime/'runtime.o').exists() else build(runtime,target)
            image_code = code(target/'engine.elf')
            if baseline_code is None:
                baseline_code = image_code
            assert image_code == baseline_code, 'Executable bytes depend on model data'
            old = target/'legacy'
            (old/'workload.h').write_text(f'#define RUN_STEPS {context}\n#define PROMPT_LEN {len(meta["prompt_tokens"])}\n'+
                'static const int prompt_tokens[] = {'+','.join(map(str,meta['prompt_tokens']))+'};\n')
            subprocess.run(['clang','-O2','-fno-unroll-loops','-fno-vectorize','-fno-slp-vectorize','-fwrapv',
                            '-std=c11','-DNATIVE','-I'+str(old),str(OLD/'engine.c'),'-o',str(old/'native')],check=True)
            expected = subprocess.check_output([str(old/'native'),str(old/'model.bin')])
            actual = subprocess.check_output([str(runtime/'native'),str(target/'model.bin')])
            (target/'old-output.bin').write_bytes(expected)
            (target/'native-output.bin').write_bytes(actual)
            assert expected == actual, (name,index,'IDs, hashes, clips or bytes differ')
            assert len(records(actual)) == context
            cases.append(dict(model=name,prompt=prompt,positions=context,output_sha256=sha(actual),
                              records=records(actual),rom_sha256=meta['rom_sha256'],
                              executable_bytes_identical=True,ram=meta['ram'],linked_margin=compiled['linked_margin']))
    native = out/'runtime/native'
    sanitized = out/'runtime/native-sanitized'
    subprocess.run(['clang','-O1','-DNATIVE','-fwrapv','-fsanitize=undefined,address',
                    str(Path(__file__).with_name('runtime.c')),'-o',str(sanitized)],check=True)
    for name in ['stories-0','custom-0']:
        case = out/name
        result = subprocess.run([str(sanitized),str(case/'model.bin')],capture_output=True,check=True)
        assert not result.stderr, result.stderr.decode()
        assert result.stdout == (case/'native-output.bin').read_bytes()
    result = dict(status='pass',kind='native old-vs-new integer runtime parity',cases=cases,
                  total_positions=sum(c['positions'] for c in cases),
                  native_binary_sha256=sha(native.read_bytes()),
                  executable_sha256=[sha(c[2]) for c in baseline_code],
                  asan_ubsan=True,software_reference='pending')
    (out/'legacy-parity.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    a = p.parse_args()
    r = legacy(a.out.resolve())
    print(json.dumps({k:v for k,v in r.items() if k != 'cases'},indent=2))
