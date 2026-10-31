"""Record host CPU and GPU load beside performance evidence."""
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[2] / 'tools'))
from remote_config import SSH_HOST, REMOTE_ROOT, LOCK, TRAIN_ROOT
import base64
import argparse
p=argparse.ArgumentParser()
p.add_argument("--name",default="host-before-bench.json")
a=p.parse_args()
assert a.name.endswith(".json") and "/" not in a.name
from pathlib import Path
import subprocess
script="""$ProgressPreference='SilentlyContinue'
@{utc=(Get-Date).ToUniversalTime().ToString('o'); cpu=@(Get-CimInstance Win32_Processor | Select-Object NumberOfLogicalProcessors,LoadPercentage); python=@(Get-CimInstance Win32_Process | Where-Object {$_.Name -eq 'python.exe'} | Select-Object ProcessId,CommandLine); gpu=@(nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.free,clocks.sm,power.draw --format=csv)} | ConvertTo-Json -Depth 5
"""
encoded=base64.b64encode(script.encode('utf-16-le')).decode()
result=subprocess.check_output(['ssh',SSH_HOST,'powershell -NoProfile -EncodedCommand '+encoded])
Path(__file__).with_name('evidence').joinpath(a.name).write_bytes(result)
print(result.decode())
