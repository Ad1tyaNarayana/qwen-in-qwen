"""Sync and detach the sequential reduced-circuit legacy gate through ssh gpu."""
import argparse
import base64
from datetime import datetime, timezone
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
REMOTE = 'C:/Users/Admin/qwenputer-games-work'


def sync(paths):
    archive = subprocess.Popen(['tar', '-cf', '-', *paths], cwd=ROOT, stdout=subprocess.PIPE)
    try:
        subprocess.run(['ssh', 'gpu', f'tar -xf - -C {REMOTE}/repo'], stdin=archive.stdout, check=True)
    finally:
        archive.stdout.close()
    if archive.wait():
        raise RuntimeError('tar failed')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--label', default=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    p.add_argument('--include-w8', action='store_true')
    a = p.parse_args()
    if not a.label.replace('-', '').isalnum():
        p.error('label must contain only letters, digits or hyphens')
    inputs = a.inputs.resolve().relative_to(ROOT).as_posix()
    base = f'{REMOTE}/artifacts/neural-accelerator-legacy-{a.label}'
    widths = '4,8' if a.include_w8 else '4'
    script = f"""
$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
$base='{base}'
if (Test-Path $base) {{ throw 'Choose a fresh output label' }}
New-Item -ItemType Directory $base | Out-Null
foreach ($width in @({widths})) {{
 foreach ($fixture in @('doom','stories')) {{
  tasklist /FI "IMAGENAME eq python.exe" | Out-File -Append ($base+'/tasklist.log')
  $active=Get-CimInstance Win32_Process | Where-Object {{$_.Name -eq 'python.exe'}}
  if ($active) {{ throw 'Another Python job is running' }}
  $label='w'+$width+'-'+$fixture
  $args='{REMOTE}/repo/experiments/neural-accelerator/legacy_neural.py --upstream {REMOTE}/.upstream/qwenputer --inputs {REMOTE}/repo/{inputs} --out '+$base+'/'+$label+' --fixture '+$fixture+' --width '+$width
  $p=Start-Process -FilePath '{REMOTE}/.venv/Scripts/python.exe' -ArgumentList $args -WorkingDirectory '{REMOTE}' -RedirectStandardOutput ($base+'/'+$label+'.stdout.log') -RedirectStandardError ($base+'/'+$label+'.stderr.log') -PassThru
  $handle=$p.Handle
  $p.WaitForExit()
  @{{exit_code=$p.ExitCode; finished_utc=(Get-Date).ToUniversalTime().ToString('o')}} | ConvertTo-Json | Set-Content ($base+'/'+$label+'.exit.json')
  if ($p.ExitCode -ne 0) {{ exit $p.ExitCode }}
 }}
}}
"""
    launch_file = ROOT / f'.cache/neural-accelerator/legacy-{a.label}.ps1'
    launch_file.parent.mkdir(parents=True, exist_ok=True)
    launch_file.write_text(script)
    sync(['experiments/neural-accelerator', 'experiments/memory-map/prepare.py', inputs,
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
