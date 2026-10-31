"""Read bounded M5 progress and process evidence through encoded PowerShell."""
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[2] / 'tools'))
from remote_config import SSH_HOST, REMOTE_ROOT, LOCK, TRAIN_ROOT
import argparse
import base64
import subprocess

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--label',required=True)
args=parser.parse_args()
assert args.label.replace('-','').isalnum()
base=(REMOTE_ROOT + '/artifacts/tiny-llm-ndot-')+args.label
script=f'''$ProgressPreference='SilentlyContinue'
$base='{base}'
tasklist /FI "IMAGENAME eq python.exe"
Get-ChildItem $base -Filter '*.exit.json' | ForEach-Object {{ $_.Name; Get-Content -Raw $_.FullName }}
foreach ($name in @('main/heartbeat.json','main/report.json','stock/report.json','sequence.json')) {{
 $p=Join-Path $base $name
 if(Test-Path $p) {{ $name; Get-Content -Raw $p | ConvertFrom-Json | Select-Object complete,status,generated_tokens,mismatches,tokens_per_second,compute_seconds,checkpoint_offset,output,error | ConvertTo-Json }}
}}
Get-ChildItem $base -Filter '*.stderr.log' | Where-Object {{$_.Length -gt 0}} | ForEach-Object {{ $_.Name; Get-Content -Tail 12 $_.FullName }}
'''
encoded=base64.b64encode(script.encode('utf-16-le')).decode()
subprocess.run(['ssh',SSH_HOST,'powershell -NoProfile -EncodedCommand '+encoded],check=True)
