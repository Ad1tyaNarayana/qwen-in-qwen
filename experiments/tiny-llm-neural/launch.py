"""Launch detached workers through WMI, Start-Process, and encoded PowerShell."""
import argparse
import base64
from pathlib import Path
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--stop-after', type=int)
p.add_argument('--label', default='full')
p.add_argument('--hf-after-main', action='store_true')
a = p.parse_args()
if not a.label.replace('-','').isalnum():
    raise ValueError('Label must be alphanumeric with optional hyphens')
root = 'C:/Users/Admin/qwenputer-games-work'
base = f'{root}/artifacts/tiny-llm-neural'
wait = ''
if a.hf_after_main:
    output = f'{root}/artifacts/tiny-llm-neural-hf'
    args = [f'{root}/repo/experiments/tiny-llm-neural/hf_prefix.py',
            '--upstream',f'{root}/.upstream/qwenputer','--inputs',f'{root}/repo/.cache/tiny-llm/neural-stories',
            '--output',output,'--max-seconds','1500','--allocator-gib','14']
    wait = f"""
$deadline=(Get-Date).AddHours(4)
while ($true) {{
 if ((Get-Date) -gt $deadline) {{ throw 'Main-run wait deadline exceeded' }}
 $r=Get-Content -Raw '{base}/report.json' | ConvertFrom-Json
 if ($r.error) {{ throw 'Main circuit run failed. HF has not started.' }}
 if ($r.complete) {{ break }}
 Start-Sleep -Seconds 10
}}
"""
    cleanup = ''
else:
    output = base
    args = [f'{root}/repo/experiments/tiny-llm-neural/worker.py','--repo',f'{root}/repo',
            '--upstream',f'{root}/.upstream/qwenputer','--inputs',f'{root}/repo/.cache/tiny-llm/neural-stories',
            '--out',output,'--max-seconds','14400']
    if a.stop_after:
        args += ['--stop-after',str(a.stop_after)]
    cleanup = f"""
$active=Get-CimInstance Win32_Process | Where-Object {{$_.Name -eq 'python.exe' -and $_.CommandLine -like '*tiny-llm-neural/worker.py*'}}
if ($active) {{ throw 'A compact worker is already running' }}
if (Test-Path '{output}/scratch') {{ Remove-Item -Recurse -Force '{output}/scratch' }}
"""
script = f"""
$ErrorActionPreference='Stop'
New-Item -ItemType Directory -Force '{output}' | Out-Null
{wait}
{cleanup}
$p = Start-Process -FilePath '{root}/.venv/Scripts/python.exe' -ArgumentList '{' '.join(args)}' -WorkingDirectory '{root}' -RedirectStandardOutput '{output}/{a.label}.stdout.log' -RedirectStandardError '{output}/{a.label}.stderr.log' -PassThru
$handle = $p.Handle
$p.WaitForExit()
@{{exit_code=$p.ExitCode; finished_utc=(Get-Date).ToUniversalTime().ToString('o')}} | ConvertTo-Json | Set-Content '{output}/{a.label}.exit.json'
exit $p.ExitCode
"""
local_root = Path(__file__).resolve().parents[2]
launch_file = local_root / f'.cache/tiny-llm/neural-launch-{a.label}.ps1'
launch_file.write_text(script)
archive = subprocess.Popen(['tar','-cf','-',str(launch_file.relative_to(local_root))],cwd=local_root,stdout=subprocess.PIPE)
subprocess.run(['ssh','gpu',f'tar -xf - -C {root}/repo'],stdin=archive.stdout,check=True)
archive.stdout.close()
assert archive.wait() == 0
child_script = f"& '{root}/repo/{launch_file.relative_to(local_root)}'"
child = base64.b64encode(child_script.encode('utf-16-le')).decode()
launcher = "Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='powershell -ExecutionPolicy Bypass -NoProfile -EncodedCommand " + child + "'} | ConvertTo-Json"
encoded = base64.b64encode(launcher.encode('utf-16-le')).decode()
subprocess.run(['ssh','gpu',f'powershell -NoProfile -EncodedCommand {encoded}'],check=True)
