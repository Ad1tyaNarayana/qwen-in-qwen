#!/usr/bin/env python3
"""Copy small raw audit outputs after GPU work, excluding input ELFs and checkpoints."""
import base64,json,subprocess,sys,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
remote=r'C:\Users\Admin\qwenputer-games-work\artifacts\audit-1'
# Use the existing remote Python only to build an archive; no CUDA initialization.
source=r'''
import tarfile
from pathlib import Path
p=Path(r'C:\Users\Admin\qwenputer-games-work\artifacts\audit-1')
with tarfile.open(p/'outputs.tar','w') as tf:
    for f in p.rglob('*'):
        if not f.is_file() or any(x in f.relative_to(p).parts for x in ('kc','scripts','guests','guests-deep','fixture-sha256','runtime','scratch')): continue
        if f.suffix not in ('.json','.txt','.log','.u32','.bin','.jsonl'): continue
        tf.add(f,arcname=str(f.relative_to(p)))
'''
ps="$s=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('"+base64.b64encode(source.encode()).decode()+r"')); & 'C:\Users\Admin\qwenputer-games-work\.venv\Scripts\python.exe' -c $s"
enc=base64.b64encode(ps.encode('utf-16le')).decode()
subprocess.run(['ssh','gpu','powershell -NoProfile -EncodedCommand '+enc],check=True,stdout=subprocess.DEVNULL)
out=ROOT/'.cache/audit/remote';out.mkdir(parents=True,exist_ok=True)
with (out/'outputs.tar').open('wb') as f:subprocess.run(['ssh','gpu',f'type {remote}\\outputs.tar'],stdout=f,check=True)
with tarfile.open(out/'outputs.tar') as tf:tf.extractall(out,filter='data')
print(out)
