"""Bind the retained Mac and PC inputs and immutable runtime dependencies."""
import base64
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
D=Path(__file__).resolve().parent
ROOT=D.parents[1]
files=[]
for name in ('ndot5-20261030','ndot-story38-20261030'):
    files.extend(f'.cache/tiny-llm/{name}/{file}' for file in ('engine.elf','reference.json','reference.tokens.u32','model.bin'))
files.extend('experiments/'+name for name in ('dmc-feedback-fusion/fusion.py','dmc-rom-fusion/fusion.py',
    'tiny-llm-ndot-fused/fusion.py','tiny-llm-ndot-fused/runtime.py','tiny-llm-ndot-fused/isolation.py','tiny-llm-ndot-fused/bench.py'))
script="$ProgressPreference='SilentlyContinue'\n$ErrorActionPreference='Stop'\n$r=@{}\n"
for name in files:
    script+=f"$r['{name}']=(Get-FileHash -Algorithm SHA256 'C:/Users/Admin/qwenputer-games-work/repo/{name}').Hash.ToLowerInvariant()\n"
script+='$r | ConvertTo-Json\n'
encoded=base64.b64encode(script.encode('utf-16-le')).decode()
remote=json.loads(subprocess.check_output(['ssh','gpu','powershell -NoProfile -EncodedCommand '+encoded]))
local={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files}
assert remote==local
report=dict(status='pass',utc=datetime.now(timezone.utc).isoformat(),mac_pc_sha256_equal=True,files=local,
            scope='Read-only provenance binding after fusion acceptance and during the autonomous story. No input or GPU state altered.')
(D/'evidence/input-bindings.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(status='pass',files=len(files))))
