"""Tar-sync and detach Gate 1 on the configured SSH environment."""
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[2] / 'tools'))
from remote_config import SSH_HOST, REMOTE_ROOT, LOCK, TRAIN_ROOT
import argparse
import base64
from pathlib import Path
import subprocess
from launch_legacy import ROOT, REMOTE, sync


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--label', required=True)
    a = p.parse_args()
    assert a.label.replace('-', '').isalnum()
    inputs = '.cache/neural-accelerator/legacy-neural-inputs-20261030'
    stock = '.cache/neural-accelerator/m4-actual'
    base = f'{REMOTE}/artifacts/neural-accelerator-m5-{a.label}'
    script = f'''
$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
if (Test-Path '{base}') {{ throw 'Fresh label required' }}
if (!(Test-Path '{REMOTE}/.venv/Scripts/python.exe')) {{ throw 'Existing venv missing' }}
New-Item -ItemType Directory '{base}' | Out-Null
tasklist /FI "IMAGENAME eq python.exe" | Out-File '{base}/tasklist.log'
if (Get-CimInstance Win32_Process | Where-Object {{$_.Name -eq 'python.exe'}}) {{ throw 'Another Python job is running' }}
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv | Out-File '{base}/gpu-processes.csv'
$p=Start-Process -FilePath '{REMOTE}/.venv/Scripts/python.exe' -ArgumentList '{REMOTE}/repo/experiments/neural-accelerator/unified_gpu.py --out {base}/run --inputs {REMOTE}/repo/{inputs} --stock {REMOTE}/repo/{stock}' -WorkingDirectory '{REMOTE}' -RedirectStandardOutput '{base}/stdout.log' -RedirectStandardError '{base}/stderr.log' -PassThru
$handle=$p.Handle
$p.WaitForExit()
@{{exit_code=$p.ExitCode}} | ConvertTo-Json | Set-Content '{base}/exit.json'
'''
    path = ROOT / f'.cache/neural-accelerator/m5-{a.label}.ps1'
    path.write_text(script)
    sync(['experiments/neural-accelerator', 'experiments/memory-map/prepare.py', inputs, stock, path.relative_to(ROOT).as_posix()])
    child = base64.b64encode(f"& '{REMOTE}/repo/{path.relative_to(ROOT).as_posix()}'".encode('utf-16-le')).decode()
    command = "Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='powershell -ExecutionPolicy Bypass -NoProfile -EncodedCommand " + child + "'} | ConvertTo-Json"
    encoded = base64.b64encode(command.encode('utf-16-le')).decode()
    subprocess.run(['ssh', SSH_HOST, 'powershell -NoProfile -EncodedCommand ' + encoded], check=True)
    print(base)


if __name__ == '__main__':
    main()
