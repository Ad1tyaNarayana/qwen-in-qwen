"""Detach the revised stress diagnostics and a separate untimed profile."""
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[2] / 'tools'))
from remote_config import SSH_HOST, REMOTE_ROOT, LOCK, TRAIN_ROOT
import base64
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/neural-accelerator'))
from launch_legacy import REMOTE,sync
base=f'{REMOTE}/artifacts/tiny-llm-ndot-fused-extra-20261030a'
repo=f'{REMOTE}/repo'
d=f'{repo}/experiments/tiny-llm-ndot-fused'
inputs=f'{repo}/.cache/tiny-llm/ndot5-20261030'
script=f'''$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
if(Test-Path '{base}') {{throw 'Fresh path required'}}
New-Item -ItemType Directory '{base}' | Out-Null
& '{REMOTE}/.venv/Scripts/python.exe' -c "import sys; assert sys.prefix.endswith('.venv'); print(sys.executable)" | Out-File '{base}/venv.log'
if($LASTEXITCODE -ne 0) {{throw 'Venv verification failed'}}
'''
for name,cmd in [('bench',f'{d}/bench.py --inputs {inputs} --out {base}/bench --gate {REMOTE}/artifacts/tiny-llm-ndot-fused-gate-20261030a --tokens 65536'),
                 ('apply',f'{d}/apply_race.py --inputs {inputs} --out {base}/apply.json'),
                 ('profile',f'{d}/profile_fused.py --inputs {inputs} --out {base}/profile')]:
    script+=f'''
tasklist /FI "IMAGENAME eq python.exe" /V | Out-File -Append '{base}/tasklist.log'
if(Get-CimInstance Win32_Process | Where-Object {{$_.Name -eq 'python.exe'}}) {{throw 'Another Python job is running'}}
$p=Start-Process -FilePath '{REMOTE}/.venv/Scripts/python.exe' -ArgumentList '{cmd}' -WorkingDirectory '{REMOTE}' -RedirectStandardOutput '{base}/{name}.stdout.log' -RedirectStandardError '{base}/{name}.stderr.log' -PassThru
$handle=$p.Handle
$p.WaitForExit()
@{{exit_code=$p.ExitCode}} | ConvertTo-Json | Set-Content '{base}/{name}.exit.json'
if($p.ExitCode -ne 0) {{exit $p.ExitCode}}
'''
script+=f"@{{complete=$true}} | ConvertTo-Json | Set-Content '{base}/sequence.json'\n"
path=ROOT/'.cache/tiny-llm/fused-extra-20261030a.ps1'
path.write_text(script)
sync(['experiments/tiny-llm-ndot-fused',path.relative_to(ROOT).as_posix()])
child=base64.b64encode(f"& '{repo}/{path.relative_to(ROOT).as_posix()}'".encode('utf-16-le')).decode()
command="Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='powershell -ExecutionPolicy Bypass -NoProfile -EncodedCommand "+child+"'} | ConvertTo-Json"
encoded=base64.b64encode(command.encode('utf-16-le')).decode()
subprocess.run(['ssh',SSH_HOST,'powershell -NoProfile -EncodedCommand '+encoded],check=True)
print(base)
