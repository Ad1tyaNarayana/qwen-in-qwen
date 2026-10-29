"""Verify sanitizers and both memory-budget rejection gates."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from build import ROOT,HERE

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--quality',type=Path,default=ROOT / '.cache/tiny-llm/custom-quality-final')
a = p.parse_args()
quality = json.loads((a.quality / 'quality.json').read_text())
results = []
for i,case in enumerate(quality['cases']):
    prompt_index = i//4
    out = a.quality / f"case-{prompt_index}-kv{case['kv_bits']}-q{case['kv_frac']}"
    command = ['clang','-O1','-fno-asynchronous-unwind-tables','-fno-unwind-tables',
               '-fno-unroll-loops','-fno-vectorize','-fno-slp-vectorize','-fwrapv',
               '-std=c11','-DNATIVE','-fsanitize=address,undefined','-I'+str(out),
               str(HERE / 'engine.c'),'-o',str(out / 'sanitized')]
    subprocess.run(command,check=True)
    actual = subprocess.run([str(out / 'sanitized'),str(out / 'model.bin')],capture_output=True,check=True)
    assert not actual.stderr and actual.stdout == (out / 'integer.txt').read_bytes()
    results.append(dict(prompt_index=prompt_index,kv_bits=case['kv_bits'],kv_frac=case['kv_frac'],
                        sanitized_output_equal=True,sha256=hashlib.sha256(actual.stdout).hexdigest()))
rejections = []
for bits,frac,context in [(16,10,130),(8,3,256)]:
    out = ROOT / f'.cache/tiny-llm/reject-{bits}-{context}'
    command = ['python3',str(HERE / 'build.py'),'--out',str(out),'--model',
               str(ROOT / '.cache/tiny-llm/custom/model.bin'),'--tokenizer',
               str(ROOT / '.cache/tiny-llm/custom/tok1024.bin'),'--context',str(context),
               '--steps','5','--prompt','Once','--kv-bits',str(bits),'--kv-frac',str(frac)]
    result = subprocess.run(command,capture_output=True)
    message = result.stderr.decode(errors='replace')
    assert result.returncode != 0
    assert ('RAM needs' in message if bits == 16 else 'Static RAM overlaps stack or margin' in message),message
    rejections.append(dict(kv_bits=bits,context=context,returncode=result.returncode,
                           gate='packer RAM budget' if bits == 16 else 'linker static-RAM margin'))
report = dict(sanitizer_cases=results,context_rejections=rejections)
(a.quality / 'checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
