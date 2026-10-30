"""Acquire the shared GPU lock and detach a sequential, bounded showcase batch."""
import argparse
import base64
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/neural-accelerator'))
from launch_legacy import REMOTE,sync
LOCK=r'C:\Users\Admin\gpu-lock'


def ps(script):
    encoded=base64.b64encode(script.encode('utf-16-le')).decode()
    return subprocess.check_output(['ssh','gpu','powershell -NoProfile -EncodedCommand '+encoded],text=True)


def launch(label,demos):
    if not label.replace('-','').isalnum():raise ValueError('Alphanumeric label required')
    for demo in demos:
        out=ROOT/'.cache/showcase'/demo
        ref=json.loads((out/'reference.json').read_text())
        if ref['tokens']>2000000:raise ValueError('Guest exceeds batch budget')
    # mkdir is the atomic acquisition. No GPU job or sync starts before ownership.
    acquired=subprocess.run(['ssh','gpu',f'mkdir {LOCK}'],capture_output=True,text=True)
    if acquired.returncode:
        print('GPU lock is held. No job started. '+acquired.stderr.strip(),flush=True)
        return False
    detached=False
    try:
        owner=f'showcase-{label}; start={datetime.now(timezone.utc).isoformat()}; expected <= 2 hours'
        ps(f"Set-Content -LiteralPath '{LOCK}/owner.txt' -Value '{owner}'")
        base=f'{REMOTE}/artifacts/showcase-{label}'
        repo=f'{REMOTE}/repo'
        script=f'''$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
$deadline=(Get-Date).AddHours(2.75)
try {{
 if(Test-Path '{base}') {{throw 'Fresh artifact directory required'}}
 New-Item -ItemType Directory '{base}' | Out-Null
 & '{REMOTE}/.venv/Scripts/python.exe' -c "import sys,torch,triton; assert sys.prefix.endswith('.venv'); print(sys.executable)" | Out-File '{base}/venv.log'
 if($LASTEXITCODE -ne 0) {{throw 'Existing venv verification failed'}}
 $env:PYTORCH_ALLOC_CONF='expandable_segments:True'
 function Run-Guest($arguments,$name) {{
  if((Get-Date) -gt $deadline) {{throw 'Batch deadline reached'}}
  tasklist /FI "IMAGENAME eq python.exe" /V | Out-File -Append '{base}/tasklist.log'
  if(Get-CimInstance Win32_Process | Where-Object {{$_.Name -eq 'python.exe'}}) {{throw 'Another Python process is running; refusing to overlap'}}
  $p=Start-Process -FilePath '{REMOTE}/.venv/Scripts/python.exe' -ArgumentList $arguments -WorkingDirectory '{REMOTE}' -RedirectStandardOutput "{base}/$name.stdout.log" -RedirectStandardError "{base}/$name.stderr.log" -PassThru
  $handle=$p.Handle
  if(-not $p.WaitForExit(2700000)) {{$p.Kill();throw "Own worker exceeded 45 minutes: $name"}}
  $p.Refresh()
  @{{exit_code=$p.ExitCode;finished_utc=(Get-Date).ToUniversalTime().ToString('o')}} | ConvertTo-Json | Set-Content "{base}/$name.exit.json"
  if($p.ExitCode -ne 0) {{throw "Worker failed: $name"}}
 }}
'''
        for demo in demos:
            inputs=f'{repo}/.cache/showcase/{demo}'
            target=f'{base}/{demo}'
            tokens=json.loads((ROOT/'.cache/showcase'/demo/'reference.json').read_text())['tokens']
            for name,stop in [('prefix','--stop-after 65536'),('restore','--stop-after 65792'),('full','')]:
                script+=f"\n if(Test-Path '{target}/fused/scratch') {{Remove-Item -Recurse -Force '{target}/fused/scratch'}}\n"
                cmd=f'{repo}/experiments/showcase/gpu_worker.py --repo {repo} --upstream {REMOTE}/.upstream/qwenputer --inputs {inputs} --out {target}/fused --max-seconds 2600 {stop}'
                script+=f" Run-Guest '{cmd}' '{demo}-{name}'\n Copy-Item '{target}/fused/report.json' '{target}/fused/{name}-report.json'\n"
            # Whole SHA is shorter than the checkpoint windows, but stock remains bounded.
            cmd=f'{repo}/experiments/showcase/gpu_stock.py --upstream {REMOTE}/.upstream/qwenputer --inputs {inputs} --out {target}/stock --tokens {min(16384 if demo=='mnist' else 4096,tokens)}'
            script+=f" Run-Guest '{cmd}' '{demo}-stock'\n"
        script+=f" @{{status='pass';complete=$true}} | ConvertTo-Json | Set-Content '{base}/sequence.json'\n"
        script+=f'''}} catch {{
 @{{status='fail';error=$_.ToString()}} | ConvertTo-Json | Set-Content '{base}/sequence.json'
 throw
}} finally {{
 cmd /c 'rmdir /s /q {LOCK}'
}}
'''
        path=ROOT/'.cache/showcase'/f'launch-{label}.ps1'
        path.write_text(script)
        paths=['experiments/showcase','experiments/tiny-llm-ndot','experiments/tiny-llm-ndot-fused',
            'experiments/neural-accelerator','experiments/dmc-feedback-fusion','experiments/dmc-rom-fusion',
            'experiments/memory-map/prepare.py','experiments/dmc-cuda/source/helpers/state.py',
            path.relative_to(ROOT).as_posix()]
        for demo in demos:
            for name in ['engine.elf','reference.json','reference.tokens.u32']:
                paths.append(f'.cache/showcase/{demo}/{name}')
        sync(paths)
        child=base64.b64encode(f"& '{repo}/{path.relative_to(ROOT).as_posix()}'".encode('utf-16-le')).decode()
        command="Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='powershell -ExecutionPolicy Bypass -NoProfile -EncodedCommand "+child+"'} | ConvertTo-Json"
        validation="$tokens=$null; $errors=$null; [System.Management.Automation.Language.Parser]::ParseFile('"+repo+"/"+path.relative_to(ROOT).as_posix()+"',[ref]$tokens,[ref]$errors) | Out-Null; if($errors.Count){throw ($errors | Out-String)}; "
        result=ps(validation+command)
        if json.loads(result)['ReturnValue']!=0:raise RuntimeError(result)
        detached=True
        print(base,flush=True)
        (ROOT/'.cache/showcase/remote.json').write_text(json.dumps(dict(base=base,label=label,demos=demos),indent=2)+'\n')
        return True
    finally:
        if not detached:subprocess.run(['ssh','gpu',f'rmdir /s /q {LOCK}'],check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--label',required=True)
    p.add_argument('--demos',nargs='+',choices=['sha256','life','tictactoe','mnist','mnist-scalar'],default=['sha256','life','tictactoe','mnist','mnist-scalar'])
    p.add_argument('--wait-seconds',type=int,default=0)
    a=p.parse_args()
    if not 0<=a.wait_seconds<=14400:p.error('Wait budget must be 0..14400 seconds')
    deadline=time.monotonic()+a.wait_seconds
    while not launch(a.label,a.demos):
        remaining=deadline-time.monotonic()
        if remaining<=0:raise SystemExit(75)
        time.sleep(min(30,remaining))
