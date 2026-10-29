"""Detach a fresh stock-only run after a non-numerical startup repair."""
import argparse
import base64
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/neural-accelerator'))
from launch_legacy import REMOTE,sync

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--label',required=True)
parser.add_argument('--tokens',type=int,default=32768)
parser.add_argument('--allocator-config')
args=parser.parse_args()
assert args.label.replace('-','').isalnum()
if args.allocator_config:
    assert all(ch.isalnum() or ch in ':_,.' for ch in args.allocator_config)
allocator=("$env:PYTORCH_ALLOC_CONF='"+args.allocator_config+"'\n") if args.allocator_config else ''
base=f'{REMOTE}/artifacts/tiny-llm-ndot-{args.label}'
repo=f'{REMOTE}/repo'
script=f'''$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
{allocator}if(Test-Path '{base}') {{ throw 'Fresh label required' }}
tasklist /FI "IMAGENAME eq python.exe"
if(Get-CimInstance Win32_Process | Where-Object {{$_.Name -eq 'python.exe'}}) {{ throw 'Another Python job is running' }}
if(!(Test-Path '{REMOTE}/.venv/Scripts/python.exe')) {{ throw 'Existing venv missing' }}
Remove-Item '{repo}/experiments/tiny-llm-ndot/profile.py' -ErrorAction SilentlyContinue
Get-ChildItem '{repo}/experiments/tiny-llm-ndot/__pycache__' -Filter 'profile.*.pyc' -ErrorAction SilentlyContinue | Remove-Item
New-Item -ItemType Directory '{base}' | Out-Null
& '{REMOTE}/.venv/Scripts/python.exe' -c "import sys; sys.path.insert(0,'{repo}/experiments/tiny-llm-ndot'); import torch,cProfile,profile; assert sys.prefix.endswith('.venv'); assert 'tiny-llm-ndot' not in profile.__file__; print(sys.executable,profile.__file__)" | Out-File '{base}/import-control.log'
if($LASTEXITCODE -ne 0) {{ throw 'Import control failed' }}
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv | Out-File '{base}/gpu-processes.csv'
$p=Start-Process -FilePath '{REMOTE}/.venv/Scripts/python.exe' -ArgumentList '{repo}/experiments/tiny-llm-ndot/stock_prefix.py --upstream {REMOTE}/.upstream/qwenputer --inputs {repo}/.cache/tiny-llm/ndot5-20261030 --out {base}/stock --tokens {args.tokens}' -WorkingDirectory '{REMOTE}' -RedirectStandardOutput '{base}/stock.stdout.log' -RedirectStandardError '{base}/stock.stderr.log' -PassThru
$handle=$p.Handle
$p.WaitForExit()
@{{exit_code=$p.ExitCode}} | ConvertTo-Json | Set-Content '{base}/stock-success.exit.json'
if($p.ExitCode -ne 0) {{ exit $p.ExitCode }}
@{{complete=$true}} | ConvertTo-Json | Set-Content '{base}/sequence.json'
'''
local=ROOT/f'.cache/tiny-llm/ndot-{args.label}.ps1'
local.write_text(script)
sync(['experiments/tiny-llm-ndot/stock_prefix.py','experiments/tiny-llm-ndot/profile_gpu.py',local.relative_to(ROOT).as_posix()])
child=base64.b64encode(f"& '{repo}/{local.relative_to(ROOT).as_posix()}'".encode('utf-16-le')).decode()
command="Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='powershell -ExecutionPolicy Bypass -NoProfile -EncodedCommand "+child+"'} | ConvertTo-Json"
encoded=base64.b64encode(command.encode('utf-16-le')).decode()
subprocess.run(['ssh','gpu','powershell -NoProfile -EncodedCommand '+encoded],check=True)
print(base)
