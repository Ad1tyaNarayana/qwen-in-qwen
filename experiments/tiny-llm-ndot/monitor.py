"""Retain bounded live progress and GPU utilization without starting a GPU job."""
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[2] / 'tools'))
from remote_config import SSH_HOST, REMOTE_ROOT, LOCK, TRAIN_ROOT
import argparse
import base64
import json
from pathlib import Path
import subprocess
import time

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--label',required=True)
parser.add_argument('--log',type=Path,required=True)
args=parser.parse_args()
assert args.label.replace('-','').isalnum()
base=(REMOTE_ROOT + '/artifacts/tiny-llm-ndot-')+args.label
script=f'''$ProgressPreference='SilentlyContinue'
$r=@{{}}
$base='{base}'
foreach ($name in @('main/heartbeat.json','main/report.json','stock/report.json','sequence.json')) {{
 $p=Join-Path $base $name
 if(Test-Path $p) {{ $r[$name]=Get-Content -Raw $p | ConvertFrom-Json | Select-Object complete,status,generated_tokens,mismatches,tokens_per_second,checkpoint_offset,output,error }}
}}
$r['latest_exit']=Get-ChildItem $base -Filter '*.exit.json' | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty Name
$r['python_processes']=@(Get-CimInstance Win32_Process | Where-Object {{$_.Name -eq 'python.exe'}} | Select-Object -ExpandProperty ProcessId)
$r['gpu']=nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.free,power.draw --format=csv,noheader
$r | ConvertTo-Json -Depth 5 -Compress
'''
encoded=base64.b64encode(script.encode('utf-16-le')).decode()
started=time.monotonic()
while time.monotonic()-started < 4*3600:
    output=subprocess.check_output(['ssh',SSH_HOST,'powershell -NoProfile -EncodedCommand '+encoded],text=True)
    data=json.loads(output)
    with args.log.open('a') as stream:
        stream.write(json.dumps(data)+'\n')
    print(json.dumps(data),flush=True)
    if data.get('sequence.json',{}).get('complete'):
        break
    if any(isinstance(value,dict) and value.get('error') for value in data.values()):
        raise SystemExit(1)
    if not data['python_processes']:
        raise SystemExit('No worker active before sequence completion')
    time.sleep(45)
else:
    raise SystemExit('Monitoring deadline exceeded')
