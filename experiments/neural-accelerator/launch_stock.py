"""Sync and detach one stock gate, refusing to overlap another Python GPU job."""
import argparse
import base64
from datetime import datetime, timezone
from pathlib import Path
import subprocess

from launch_legacy import ROOT, REMOTE, sync


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--label', default=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    a = p.parse_args()
    if not a.label.replace('-', '').isalnum():
        p.error('label must contain only letters, digits or hyphens')
    base = f'{REMOTE}/artifacts/neural-accelerator-m3-{a.label}'
    script = f"""
$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
tasklist /FI "IMAGENAME eq python.exe"
$active=Get-CimInstance Win32_Process | Where-Object {{$_.Name -eq 'python.exe'}}
if ($active) {{ throw 'Another Python job is running' }}
$base='{base}'
if (Test-Path $base) {{ throw 'Choose a fresh output label' }}
New-Item -ItemType Directory $base | Out-Null
tasklist /FI "IMAGENAME eq python.exe" | Out-File ($base+'/tasklist.log')
$args='{REMOTE}/repo/experiments/neural-accelerator/stock_m3.py --upstream {REMOTE}/.upstream/qwenputer --out '+$base+'/run --device cuda'
$p=Start-Process -FilePath '{REMOTE}/.venv/Scripts/python.exe' -ArgumentList $args -WorkingDirectory '{REMOTE}' -RedirectStandardOutput ($base+'/stdout.log') -RedirectStandardError ($base+'/stderr.log') -PassThru
$handle=$p.Handle
$p.WaitForExit()
@{{exit_code=$p.ExitCode; finished_utc=(Get-Date).ToUniversalTime().ToString('o')}} | ConvertTo-Json | Set-Content ($base+'/exit.json')
exit $p.ExitCode
"""
    launch_file = ROOT / f'.cache/neural-accelerator/stock-{a.label}.ps1'
    launch_file.parent.mkdir(parents=True, exist_ok=True)
    launch_file.write_text(script)
    sync(['experiments/neural-accelerator', 'experiments/memory-map/prepare.py',
          launch_file.relative_to(ROOT).as_posix()])
    child = f"& '{REMOTE}/repo/{launch_file.relative_to(ROOT).as_posix()}'"
    encoded_child = base64.b64encode(child.encode('utf-16-le')).decode()
    remote_script = f'{REMOTE}/repo/{launch_file.relative_to(ROOT).as_posix()}'
    launcher = "$tokens=$null; $errors=$null; [System.Management.Automation.Language.Parser]::ParseFile('" + remote_script + "',[ref]$tokens,[ref]$errors) | Out-Null; if($errors.Count){throw ($errors | Out-String)}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='powershell -ExecutionPolicy Bypass -NoProfile -EncodedCommand " + encoded_child + "'} | ConvertTo-Json"
    encoded = base64.b64encode(launcher.encode('utf-16-le')).decode()
    subprocess.run(['ssh', 'gpu', 'powershell -NoProfile -EncodedCommand ' + encoded], check=True)
    print(base)


if __name__ == '__main__':
    main()
