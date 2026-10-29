"""Sync and detach sequential M5 checks using the verified existing GPU venv."""
import argparse
import base64
from pathlib import Path
import subprocess
import sys
import json

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/neural-accelerator'))
from launch_legacy import REMOTE, sync


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label',required=True)
    parser.add_argument('--gate',type=Path,required=True)
    args = parser.parse_args()
    assert args.label.replace('-','').isalnum()
    gate=json.loads(args.gate.read_text())
    assert gate['status']=='pass' and gate['gate']==2
    base = f'{REMOTE}/artifacts/tiny-llm-ndot-{args.label}'
    repo = f'{REMOTE}/repo'
    fixture = '.cache/tiny-llm/ndot5-20261030'
    original = '.cache/tiny-llm/neural-stories'
    commands = []
    def worker(name,inputs,extra=''):
        commands.append((name,f'{repo}/experiments/tiny-llm-ndot/worker.py --repo {repo} --upstream {REMOTE}/.upstream/qwenputer --inputs {repo}/{inputs} --out {base}/{name} --max-seconds 14400 {extra}'))
    worker('main',fixture,'--stop-after 65536')
    commands.append(('retain-prefix',None))
    worker('main',fixture,'--stop-after 65792')
    commands.append(('retain-resume',None))
    worker('main',fixture)
    for repeat in range(5):
        worker(f'original-unfused-{repeat}',original,'--original-circuit --stop-after 65536')
        worker(f'unified-original-{repeat}',original,'--stop-after 65536')
        worker(f'unified-ndot-{repeat}',fixture,'--stop-after 65536')
    commands.append(('stock',f'{repo}/experiments/tiny-llm-ndot/stock_prefix.py --upstream {REMOTE}/.upstream/qwenputer --inputs {repo}/{fixture} --out {base}/stock --tokens 32768'))
    script = f'''$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
if (Test-Path '{base}') {{ throw 'Fresh label required' }}
if (!(Test-Path '{REMOTE}/.venv/Scripts/python.exe')) {{ throw 'Existing venv missing' }}
New-Item -ItemType Directory '{base}' | Out-Null
& '{REMOTE}/.venv/Scripts/python.exe' -c "import sys; assert sys.prefix.endswith('.venv'); print(sys.executable)" | Out-File '{base}/venv.log'
if ($LASTEXITCODE -ne 0) {{ throw 'Venv verification failed' }}
'''
    for number,(name,command) in enumerate(commands):
        if command is None:
            label = name.removeprefix('retain-')
            script += f"Copy-Item '{base}/main/report.json' '{base}/main/{label}-report.json'\n"
            continue
        if name == 'stock':
            script += "$env:PYTORCH_ALLOC_CONF='expandable_segments:True'\n"
        script += f'''
tasklist /FI "IMAGENAME eq python.exe" | Out-File -Append '{base}/tasklist.log'
if (Get-CimInstance Win32_Process | Where-Object {{$_.Name -eq 'python.exe'}}) {{ throw 'Another Python job is running' }}
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.free --format=csv | Out-File -Append '{base}/gpu-before.csv'
$apps=nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader
$apps | Out-File -Append '{base}/gpu-processes.csv'
if ($apps -match 'python|llama|ollama|cuda.*exe') {{ throw 'Another CUDA worker is running' }}
if (Test-Path '{base}/{name}/scratch') {{ Remove-Item -Recurse -Force '{base}/{name}/scratch' }}
$p=Start-Process -FilePath '{REMOTE}/.venv/Scripts/python.exe' -ArgumentList '{command}' -WorkingDirectory '{REMOTE}' -RedirectStandardOutput '{base}/{number}-{name}.stdout.log' -RedirectStandardError '{base}/{number}-{name}.stderr.log' -PassThru
$handle=$p.Handle
$p.WaitForExit()
@{{exit_code=$p.ExitCode; finished_utc=(Get-Date).ToUniversalTime().ToString('o')}} | ConvertTo-Json | Set-Content '{base}/{number}-{name}.exit.json'
if ($p.ExitCode -ne 0) {{ exit $p.ExitCode }}
'''
    script += f"@{{complete=$true}} | ConvertTo-Json | Set-Content '{base}/sequence.json'\n"
    local = ROOT/f'.cache/tiny-llm/ndot-{args.label}.ps1'
    local.write_text(script)
    sync(['experiments/tiny-llm-ndot','experiments/neural-accelerator','experiments/memory-map/prepare.py',
          'experiments/dmc-cuda/source/helpers/state.py',fixture,original,local.relative_to(ROOT).as_posix()])
    child = base64.b64encode(f"& '{repo}/{local.relative_to(ROOT).as_posix()}'".encode('utf-16-le')).decode()
    command = "Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='powershell -ExecutionPolicy Bypass -NoProfile -EncodedCommand "+child+"'} | ConvertTo-Json"
    encoded = base64.b64encode(command.encode('utf-16-le')).decode()
    subprocess.run(['ssh','gpu','powershell -NoProfile -EncodedCommand '+encoded],check=True)
    print(base)


if __name__ == '__main__':
    main()
