"""Detach two short checked profiler runs after required stock acceptance."""
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[2] / 'tools'))
from remote_config import SSH_HOST, REMOTE_ROOT, LOCK, TRAIN_ROOT
import argparse
import base64
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/neural-accelerator'))
from launch_legacy import REMOTE,sync

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--label',required=True)
parser.add_argument('--validation',type=Path,required=True)
args=parser.parse_args()
assert args.label.replace('-','').isalnum()
assert json.loads(args.validation.read_text())['status']=='pass'
base=f'{REMOTE}/artifacts/tiny-llm-ndot-{args.label}'
repo=f'{REMOTE}/repo'
script=f'''$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
if(Test-Path '{base}') {{ throw 'Fresh label required' }}
if(!(Test-Path '{REMOTE}/.venv/Scripts/python.exe')) {{ throw 'Existing venv missing' }}
New-Item -ItemType Directory '{base}' | Out-Null
'''
for name,inputs,extra in [('original','.cache/tiny-llm/neural-stories','--original-circuit'),
                          ('unified','.cache/tiny-llm/ndot5-20261030','')]:
    script+=f'''
tasklist /FI "IMAGENAME eq python.exe" | Out-File -Append '{base}/tasklist.log'
if(Get-CimInstance Win32_Process | Where-Object {{$_.Name -eq 'python.exe'}}) {{ throw 'Another Python job is running' }}
$p=Start-Process -FilePath '{REMOTE}/.venv/Scripts/python.exe' -ArgumentList '{repo}/experiments/tiny-llm-ndot/profile_gpu.py --repo {repo} --upstream {REMOTE}/.upstream/qwenputer --inputs {repo}/{inputs} --out {base}/{name} {extra}' -WorkingDirectory '{REMOTE}' -RedirectStandardOutput '{base}/{name}.stdout.log' -RedirectStandardError '{base}/{name}.stderr.log' -PassThru
$handle=$p.Handle
$p.WaitForExit()
@{{exit_code=$p.ExitCode}} | ConvertTo-Json | Set-Content '{base}/{name}.exit.json'
if($p.ExitCode -ne 0) {{ exit $p.ExitCode }}
'''
script+=f"@{{complete=$true}} | ConvertTo-Json | Set-Content '{base}/sequence.json'\n"
local=ROOT/f'.cache/tiny-llm/ndot-{args.label}.ps1'
local.write_text(script)
sync(['experiments/tiny-llm-ndot/profile_gpu.py',local.relative_to(ROOT).as_posix()])
child=base64.b64encode(f"& '{repo}/{local.relative_to(ROOT).as_posix()}'".encode('utf-16-le')).decode()
command="Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='powershell -ExecutionPolicy Bypass -NoProfile -EncodedCommand "+child+"'} | ConvertTo-Json"
encoded=base64.b64encode(command.encode('utf-16-le')).decode()
subprocess.run(['ssh',SSH_HOST,'powershell -NoProfile -EncodedCommand '+encoded],check=True)
print(base)
