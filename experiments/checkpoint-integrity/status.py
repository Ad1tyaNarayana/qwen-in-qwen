"""Read-only status for this task's detached GPU batch, via the configured SSH host."""
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[2] / 'tools'))
from remote_config import SSH_HOST, REMOTE_ROOT, LOCK, TRAIN_ROOT
import base64
import subprocess
script=("$ProgressPreference='SilentlyContinue'\n$base='" + REMOTE_ROOT + "/artifacts/checkpoint-integrity-20261030a'\nforeach($name in @('sequence.json','mutations/report.json','scalar/heartbeat.json','scalar/report.json','ndot/heartbeat.json','ndot/report.json')) {\n $path=$base+'/'+$name\n if(Test-Path $path) {$name;Get-Content -Raw $path | ConvertFrom-Json | Select-Object complete,status,generated_tokens,total_tokens,mismatches,compute_seconds,tokens_per_second,checkpoint_offset,updated_utc,error | ConvertTo-Json}\n}\nGet-CimInstance Win32_Process | Where-Object {$_.Name -eq 'python.exe'} | Select-Object ProcessId,CommandLine | ConvertTo-Json\n")
subprocess.run(['ssh','-O','check',SSH_HOST],check=True)
subprocess.run(['ssh',SSH_HOST,'powershell -NoProfile -EncodedCommand '+base64.b64encode(script.encode('utf-16-le')).decode()],check=True)
