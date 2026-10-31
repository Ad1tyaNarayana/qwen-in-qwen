"""Read detached job progress through cmd and encoded PowerShell."""
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[2] / 'tools'))
from remote_config import SSH_HOST, REMOTE_ROOT, LOCK, TRAIN_ROOT
import argparse
import base64
import subprocess
p=argparse.ArgumentParser()
p.add_argument('base')
a=p.parse_args()
assert a.base.startswith((REMOTE_ROOT + '/artifacts/tiny-llm-ndot-'))
script=f'''$ProgressPreference='SilentlyContinue'
$base='{a.base}'
$r=@{{python=@(Get-CimInstance Win32_Process | Where-Object {{$_.Name -eq 'python.exe'}} | Select-Object ProcessId,CommandLine)}}
foreach($name in @('apply.exit.json','feedback.exit.json','baseline.exit.json','fused.exit.json','compare.exit.json','bench.exit.json','sequence.json','prefix.exit.json','restore.exit.json','full.exit.json','main/heartbeat.json')) {{
if(Test-Path "$base/$name") {{$r[$name]=Get-Content "$base/$name" -Raw | ConvertFrom-Json}}
}}
foreach($name in @('baseline.stdout.log','fused.stdout.log','bench.stdout.log','full.stdout.log')) {{
if(Test-Path "$base/$name") {{$r[$name]=[string]::Join(" ", [string[]]@(Get-Content "$base/$name" -Tail 1))}}
}}
foreach($name in @('baseline.stderr.log','fused.stderr.log','bench.stderr.log','prefix.stderr.log','restore.stderr.log','full.stderr.log')) {{
if(Test-Path "$base/$name") {{$r[$name]=[string]::Join(" ", [string[]]@(Get-Content "$base/$name" -Tail 5))}}
}}
$r | ConvertTo-Json -Depth 5
'''
encoded=base64.b64encode(script.encode('utf-16-le')).decode()
subprocess.run(['ssh',SSH_HOST,'powershell -NoProfile -EncodedCommand '+encoded],check=True)
