"""Read-only detached-batch monitoring; never restarts GPU work or changes state."""
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[2] / 'tools'))
from remote_config import SSH_HOST, REMOTE_ROOT, LOCK, TRAIN_ROOT
import base64
import json
from pathlib import Path
import subprocess
import time
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'.cache/checkpoint-integrity/monitor.jsonl'
script=("$ProgressPreference='SilentlyContinue'\n$base='" + REMOTE_ROOT + "/artifacts/checkpoint-integrity-20261030a'\n$result=@{utc=(Get-Date).ToUniversalTime().ToString('o')}\nforeach($name in @('scalar','ndot')) {\n $h=$base+'/'+$name+'/heartbeat.json'\n if(Test-Path $h) {$result[$name]=Get-Content -Raw $h|ConvertFrom-Json}\n}\nif(Test-Path ($base+'/sequence.json')) {$result['sequence']=Get-Content -Raw ($base+'/sequence.json')|ConvertFrom-Json}\n$result['processes']=@(Get-CimInstance Win32_Process | Where-Object {$_.Name -eq 'python.exe'} | Select-Object ProcessId,CommandLine)\n$result['lock_exists']=Test-Path '" + LOCK + "'\n$result|ConvertTo-Json -Depth 5 -Compress\n")
command=['ssh',SSH_HOST,'powershell -NoProfile -EncodedCommand '+base64.b64encode(script.encode('utf-16-le')).decode()]
while True:
    control=subprocess.run(['ssh','-O','check',SSH_HOST],capture_output=True,text=True)
    if control.returncode:
        print('SSH control check failed; stopping all GPU monitoring. '+control.stderr.strip(),flush=True)
        raise SystemExit(control.returncode)
    result=subprocess.run(command,capture_output=True,text=True)
    if result.returncode:
        print('SSH status failed; stopping. '+result.stderr.strip(),flush=True)
        raise SystemExit(result.returncode)
    value=json.loads(result.stdout)
    with OUT.open('a') as f:f.write(json.dumps(value)+'\n')
    summary={name:{key:r.get(key) for key in ['generated_tokens','total_tokens','mismatches','compute_seconds','checkpoint_offset']} for name in ['scalar','ndot'] if (r:=value.get(name))}
    print(json.dumps(dict(utc=value['utc'],runs=summary,sequence=value.get('sequence'),lock_exists=value['lock_exists'])),flush=True)
    if 'sequence' in value:break
    time.sleep(55)
