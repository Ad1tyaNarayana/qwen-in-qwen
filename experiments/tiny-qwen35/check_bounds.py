"""Exercise a valid large positive decay input without overflowing its LUT index."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess

from pack import HERE


def check(image, out):
    out.mkdir(parents=True,exist_ok=True)
    metadata = json.loads((image/'pack.json').read_text())
    raw = bytearray((image/'model.bin').read_bytes())
    h = metadata['header']
    l = metadata['graph'][0]
    assert l['kind'] == 3
    raw[h['emb']:h['emb']+h['vocab']*h['dim']] = bytes(h['vocab']*h['dim'])
    for i in range(l['value_heads']):
        struct.pack_into('<i',raw,l['dt']+4*i,2**23)
    (out/'model.bin').write_bytes(raw)
    executable = out/'sanitized'
    subprocess.run(['clang','-O1','-DNATIVE','-fsanitize=undefined,address',str(HERE/'runtime.c'),
                    '-o',str(executable)],check=True)
    result = subprocess.run([str(executable.resolve()),str(out.resolve()/'model.bin')],capture_output=True)
    (out/'stderr.txt').write_bytes(result.stderr)
    (out/'output.bin').write_bytes(result.stdout)
    assert result.returncode == 0 and not result.stderr,(result.returncode,result.stderr.decode())
    assert b'!' not in result.stdout
    rows = [line.split() for line in result.stdout.splitlines() if line.startswith(b'@')]
    assert len(rows) == h['steps'] and all(row[2] == b'0' for row in rows)
    report = dict(status='pass',probe='zero embeddings and maximal Q12 dt_bias; all output logits stay zero',
                  steps=len(rows),output_sha256=hashlib.sha256(result.stdout).hexdigest(),
                  source_sha256=hashlib.sha256((HERE/'runtime.c').read_bytes()).hexdigest(),
                  asan=True,ubsan=True,fwrapv=False)
    (HERE/'evidence/large-decay-bound.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a = p.parse_args()
    check(a.image,a.out)
