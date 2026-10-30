"""Detach the corruption controls and same-network complete stories measurement.

Run under an already atomically acquired lock owned by this task. Resume only
this batch's actual checkpoints; never reference-seed the execution.
"""
import base64
import json
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'experiments/neural-accelerator'))
from launch_legacy import REMOTE, sync

def ps(script):
    return subprocess.check_output(['ssh','gpu','powershell -NoProfile -EncodedCommand '+base64.b64encode(script.encode('utf-16-le')).decode()], text=True)


def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--label',required=True)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if not a.label.replace('-','').isalnum(): raise ValueError('Invalid label')
    label=a.label
    base=f'{REMOTE}/artifacts/checkpoint-integrity-{label}'
    repo=f'{REMOTE}/repo'
    owner=f'checkpoint-integrity-{label}'
    # The caller acquires with ssh gpu mkdir before any work; never claim an existing hold.
    assert owner in ps("Get-Content 'C:/Users/Admin/gpu-lock/owner.txt'")
    script=f'''$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
try {{
 $base='{base}'
'''
    script+=f" if(-not (Test-Path $base)) {{New-Item -ItemType Directory $base | Out-Null}}\n" if a.resume else " if(Test-Path $base) {throw 'Fresh artifact directory required'}\n New-Item -ItemType Directory $base | Out-Null\n"
    script+=f'''
 & '{REMOTE}/.venv/Scripts/python.exe' -c "import sys,torch,triton; assert sys.prefix.endswith('.venv'); print(sys.executable)" | Out-File "$base/venv.log"
 if($LASTEXITCODE -ne 0) {{throw 'Existing venv verification failed'}}
 $env:PYTORCH_ALLOC_CONF='expandable_segments:True'
 function Run-Worker($arguments,$name) {{
  if(Get-CimInstance Win32_Process | Where-Object {{$_.Name -eq 'python.exe'}}) {{throw 'Another Python process is running'}}
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.free,clocks.sm,power.draw --format=csv | Out-File -Append "$base/gpu-before.csv"
  $p=Start-Process -FilePath '{REMOTE}/.venv/Scripts/python.exe' -ArgumentList $arguments -WorkingDirectory '{REMOTE}' -RedirectStandardOutput "$base/$name.stdout.log" -RedirectStandardError "$base/$name.stderr.log" -PassThru
  $handle=$p.Handle
  if(-not $p.WaitForExit(10000000)) {{$p.Kill();throw 'Own worker deadline exceeded'}}
  $p.Refresh()
  @{{exit_code=$p.ExitCode;finished_utc=(Get-Date).ToUniversalTime().ToString('o')}} | ConvertTo-Json | Set-Content "$base/$name.exit.json"
  if($p.ExitCode -ne 0) {{throw "Worker failed: $name"}}
 }}
'''
    scenarios=f'{repo}/experiments/checkpoint-integrity/gpu_ckpt.py --repo {repo} --upstream {REMOTE}/.upstream/qwenputer --inputs {repo}/.cache/showcase/sha256 --out {base}/mutations'
    script+=f" if(-not (Test-Path '{base}/mutations/report.json')) {{Run-Worker '{scenarios}' 'mutations'}}\n"
    commands=[]
    for name,fixture in [('scalar','neural-stories'),('ndot','ndot5-20261030')]:
        target=f'{base}/{name}'
        for stage,stop in [('prefix','--stop-after 65536'),('restore','--stop-after 65792'),('full','')]:
            cmd=f'{repo}/experiments/showcase/gpu_worker.py --repo {repo} --upstream {REMOTE}/.upstream/qwenputer --inputs {repo}/.cache/tiny-llm/{fixture} --out {target} --max-seconds 9800 {stop}'
            commands.append(dict(name=name,stage=stage,arguments=cmd))
            script+=f''' if(-not (Test-Path '{target}/{stage}-report.json')) {{
  if(Test-Path '{target}/scratch') {{Remove-Item -Recurse -Force '{target}/scratch'}}
  Run-Worker '{cmd}' '{name}-{stage}'
  Copy-Item '{target}/report.json' '{target}/{stage}-report.json'
 }}
'''
    script+=f''' @{{complete=$true;status='pass'}} | ConvertTo-Json | Set-Content "$base/sequence.json"
}} catch {{
 @{{complete=$false;status='fail';error=$_.ToString()}} | ConvertTo-Json | Set-Content '{base}/sequence.json'
 throw
}} finally {{
 cmd /c 'rmdir /s /q C:\\Users\\Admin\\gpu-lock'
}}
'''
    path=ROOT/f'.cache/checkpoint-integrity/{label}.ps1'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(script)
    paths=['experiments/checkpoint-integrity','experiments/tiny-llm-ndot','experiments/tiny-llm-ndot-fused','experiments/showcase','experiments/neural-accelerator','experiments/dmc-feedback-fusion','experiments/dmc-rom-fusion','experiments/memory-map/prepare.py','experiments/dmc-cuda/source/helpers/state.py',path.relative_to(ROOT).as_posix()]
    for fixture in ['.cache/showcase/sha256','.cache/tiny-llm/neural-stories','.cache/tiny-llm/ndot5-20261030']:
        for fn in ['engine.elf','reference.json','reference.tokens.u32','reference-ram.bin']:
            paths.append(f'{fixture}/{fn}')
    detached=False
    try:
        # Required comparison fields must exist before spending any GPU time.
        import hashlib
        for fixture in ['.cache/showcase/sha256','.cache/tiny-llm/neural-stories','.cache/tiny-llm/ndot5-20261030']:
            inputs=ROOT/fixture
            reference=json.loads((inputs/'reference.json').read_text())
            assert len(reference['final_registers'])==32, fixture
            for filename,key in [('engine.elf','elf_sha256'),('reference.tokens.u32','sha256')]:
                assert hashlib.sha256((inputs/filename).read_bytes()).hexdigest()==reference[key],fixture
        sync(paths)
        command=f"& '{repo}/{path.relative_to(ROOT).as_posix()}'"
        encoded=base64.b64encode(command.encode('utf-16-le')).decode()
        validate=f"$tokens=$null;$errors=$null;[System.Management.Automation.Language.Parser]::ParseFile('{repo}/{path.relative_to(ROOT).as_posix()}',[ref]$tokens,[ref]$errors)|Out-Null;if($errors.Count){{throw ($errors|Out-String)}};"
        result=ps(validate+"Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='powershell -ExecutionPolicy Bypass -NoProfile -EncodedCommand "+encoded+"'} | ConvertTo-Json")
        if json.loads(result)['ReturnValue']!=0: raise RuntimeError(result)
        detached=True
        (ROOT/'experiments/checkpoint-integrity/evidence/launch.json').write_text(json.dumps(dict(base=base,label=label,commands=commands,detach=json.loads(result)),indent=2)+'\n')
        print(base)
    finally:
        if not detached: subprocess.run(['ssh','gpu',r'rmdir /s /q C:\Users\Admin\gpu-lock'],check=True)


if __name__=='__main__': main()
