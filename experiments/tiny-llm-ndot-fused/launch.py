"""Sync a bounded sequential GPU gate and detach through Win32_Process.Create."""
import argparse
import base64
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/neural-accelerator'))
from launch_legacy import REMOTE, sync


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--label',required=True)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    assert a.label.replace('-','').isalnum()
    base=f'{REMOTE}/artifacts/tiny-llm-ndot-fused-{a.label}'
    repo=f'{REMOTE}/repo'
    scriptdir=f'{repo}/experiments/tiny-llm-ndot-fused'
    inputs=f'{repo}/.cache/tiny-llm/ndot5-20261030'
    n=256 if a.smoke else 65536
    commands=[('apply',f'{scriptdir}/apply_race.py --inputs {inputs} --out {base}/apply.json'),
              ('feedback',f'{scriptdir}/feedback_race.py --inputs {inputs} --out {base}/feedback.json')]
    for side in ('baseline','fused'):
        extra=f' --compare-dir {base}/baseline' if side=='fused' else ''
        commands.append((side,f'{scriptdir}/isolation.py --repo {repo} --inputs {inputs} --out {base}/{side} --side {side} --tokens {n} --render-tokens {n} --chunk-tokens 4096 --max-seconds 1800{extra}'))
    commands.append(('compare',f'{scriptdir}/compare.py --left {base}/baseline --right {base}/fused --out {base}/comparison.json'))
    if not a.smoke:
        commands.append(('bench',f'{scriptdir}/bench.py --inputs {inputs} --out {base}/bench --gate {base} --tokens 65536'))
    script=f'''$ErrorActionPreference='Stop'
if(Test-Path '{base}') {{throw 'Fresh path required'}}
New-Item -ItemType Directory '{base}' | Out-Null
& '{REMOTE}/.venv/Scripts/python.exe' -c "import sys; assert sys.prefix.endswith('.venv'); print(sys.executable)" | Out-File '{base}/venv.log'
if($LASTEXITCODE -ne 0) {{throw 'Venv verification failed'}}
'''
    for name,cmd in commands:
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
    path=ROOT/f'.cache/tiny-llm/fused-{a.label}.ps1'
    path.write_text(script)
    sync(['experiments/tiny-llm-ndot-fused','experiments/tiny-llm-ndot','experiments/neural-accelerator',
          'experiments/dmc-feedback-fusion','experiments/dmc-rom-fusion','experiments/memory-map/prepare.py',
          '.cache/tiny-llm/ndot5-20261030',path.relative_to(ROOT).as_posix()])
    child=base64.b64encode(f"& '{repo}/{path.relative_to(ROOT).as_posix()}'".encode('utf-16-le')).decode()
    command="Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='powershell -ExecutionPolicy Bypass -NoProfile -EncodedCommand "+child+"'} | ConvertTo-Json"
    encoded=base64.b64encode(command.encode('utf-16-le')).decode()
    subprocess.run(['ssh','gpu','powershell -NoProfile -EncodedCommand '+encoded],check=True)
    print(base)


if __name__=='__main__':
    main()
