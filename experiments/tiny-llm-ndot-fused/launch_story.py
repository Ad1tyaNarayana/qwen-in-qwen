"""Detach a fresh story only after the exact-state fusion gate passes."""
import argparse
import base64
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/neural-accelerator'))
from launch_legacy import REMOTE,sync


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--label',required=True)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--gate',type=Path)
    a=p.parse_args()
    assert a.label.replace('-','').isalnum()
    meta=json.loads((a.inputs/'reference.json').read_text())
    fused=a.gate is not None
    if fused:
        gate=json.loads((a.gate/'comparison.json').read_text())
        assert all(w['tokens']>=65536 and w['digest_mismatches']==0 for w in gate['windows'].values())
        timing=json.loads((a.gate/'bench.json').read_text())
        assert len(timing['runs'])==20 and all(x['mismatches']==0 for x in timing['runs'])
        import hashlib
        assert timing['scripts']['fusion.py']==hashlib.sha256(Path(__file__).with_name('fusion.py').read_bytes()).hexdigest()
        projected=meta['tokens']*max(w['fused']['median_us_per_token'] for w in timing['windows'].values())/1e6
        print(json.dumps(dict(projected_generation_seconds=projected,positions=meta['steps'])),flush=True)
    base=f'{REMOTE}/artifacts/tiny-llm-ndot-story-{a.label}'
    repo=f'{REMOTE}/repo'
    inputs=a.inputs.resolve().relative_to(ROOT).as_posix()
    worker='tiny-llm-ndot-fused/story.py' if fused else 'tiny-llm-ndot/worker.py'
    script=f'''$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
if(Test-Path '{base}') {{throw 'Fresh path required'}}
New-Item -ItemType Directory '{base}' | Out-Null
& '{REMOTE}/.venv/Scripts/python.exe' -c "import sys; assert sys.prefix.endswith('.venv'); print(sys.executable)" | Out-File '{base}/venv.log'
if($LASTEXITCODE -ne 0) {{throw 'Venv verification failed'}}
'''
    for name,stop in [('prefix','--stop-after 65536'),('restore','--stop-after 65792'),('full','')]:
        cmd=f'{repo}/experiments/{worker} --repo {repo} --upstream {REMOTE}/.upstream/qwenputer --inputs {repo}/{inputs} --out {base}/main --max-seconds 18000 {stop}'
        script+=f'''
tasklist /FI "IMAGENAME eq python.exe" /V | Out-File -Append '{base}/tasklist.log'
while(Get-CimInstance Win32_Process | Where-Object {{$_.Name -eq 'python.exe'}}) {{Start-Sleep -Seconds 15}}
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.free,clocks.sm,power.draw --format=csv | Out-File -Append '{base}/gpu-before.csv'
if(Test-Path '{base}/main/scratch') {{Remove-Item -Recurse -Force '{base}/main/scratch'}}
$p=Start-Process -FilePath '{REMOTE}/.venv/Scripts/python.exe' -ArgumentList '{cmd}' -WorkingDirectory '{REMOTE}' -RedirectStandardOutput '{base}/{name}.stdout.log' -RedirectStandardError '{base}/{name}.stderr.log' -PassThru
$handle=$p.Handle
$p.WaitForExit()
@{{exit_code=$p.ExitCode; finished_utc=(Get-Date).ToUniversalTime().ToString('o')}} | ConvertTo-Json | Set-Content '{base}/{name}.exit.json'
if($p.ExitCode -ne 0) {{exit $p.ExitCode}}
Copy-Item '{base}/main/report.json' '{base}/main/{name}-report.json'
'''
    script+=f"@{{complete=$true}} | ConvertTo-Json | Set-Content '{base}/sequence.json'\n"
    path=ROOT/f'.cache/tiny-llm/story-{a.label}.ps1'
    path.write_text(script)
    sync(['experiments/tiny-llm-ndot-fused','experiments/tiny-llm-ndot','experiments/neural-accelerator',
          'experiments/dmc-feedback-fusion','experiments/dmc-rom-fusion','experiments/memory-map/prepare.py',
          'experiments/dmc-cuda/source/helpers/state.py',inputs,path.relative_to(ROOT).as_posix()])
    child=base64.b64encode(f"& '{repo}/{path.relative_to(ROOT).as_posix()}'".encode('utf-16-le')).decode()
    command="Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='powershell -ExecutionPolicy Bypass -NoProfile -EncodedCommand "+child+"'} | ConvertTo-Json"
    encoded=base64.b64encode(command.encode('utf-16-le')).decode()
    subprocess.run(['ssh','gpu','powershell -NoProfile -EncodedCommand '+encoded],check=True)
    print(base)


if __name__=='__main__':
    main()
