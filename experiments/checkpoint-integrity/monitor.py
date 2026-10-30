"""Read-only detached-batch monitoring; never restarts GPU work or changes state."""
import base64
import json
from pathlib import Path
import subprocess
import time
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'.cache/checkpoint-integrity/monitor.jsonl'
script="""$ProgressPreference='SilentlyContinue'
$base='C:/Users/Admin/qwenputer-games-work/artifacts/checkpoint-integrity-20261030a'
$result=@{utc=(Get-Date).ToUniversalTime().ToString('o')}
foreach($name in @('scalar','ndot')) {
 $h=$base+'/'+$name+'/heartbeat.json'
 if(Test-Path $h) {$result[$name]=Get-Content -Raw $h|ConvertFrom-Json}
}
if(Test-Path ($base+'/sequence.json')) {$result['sequence']=Get-Content -Raw ($base+'/sequence.json')|ConvertFrom-Json}
$result['processes']=@(Get-CimInstance Win32_Process | Where-Object {$_.Name -eq 'python.exe'} | Select-Object ProcessId,CommandLine)
$result['lock_exists']=Test-Path 'C:/Users/Admin/gpu-lock'
$result|ConvertTo-Json -Depth 5 -Compress
"""
command=['ssh','gpu','powershell -NoProfile -EncodedCommand '+base64.b64encode(script.encode('utf-16-le')).decode()]
while True:
    control=subprocess.run(['ssh','-O','check','gpu'],capture_output=True,text=True)
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
