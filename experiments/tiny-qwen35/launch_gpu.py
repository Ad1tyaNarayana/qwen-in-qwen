"""Detach the exact-state neural and stock checks after the Mac gates pass."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess

from gate import gate
from pack import HERE, ROOT
from gpu_lock import locked

REMOTE = 'C:/Users/Admin/qwenputer-games-work'
PRIVATE = REMOTE+'/tiny-qwen35-runtime'


def launch(legacy, program, quality, label):
    assert label.replace('-','').isalnum()
    accepted = gate(legacy,program,quality)
    fusion = ROOT/'experiments/tiny-llm-ndot-fused'
    checks = json.loads((fusion/'evidence/final-checks.json').read_text())
    assert checks['status'] == 'pass'
    for name in ['fusion.py','runtime.py']:
        assert checks['fusion']['source_sha256'][name] == hashlib.sha256((fusion/name).read_bytes()).hexdigest()
    base = REMOTE+'/artifacts/tiny-qwen35-'+label
    relative = program.resolve().relative_to(ROOT).as_posix()
    script = f'''$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
if(Test-Path '{base}') {{throw 'Fresh artifact label required'}}
New-Item -ItemType Directory '{base}' | Out-Null
$created=$true
'''
    stages = [('prefix','tiny-llm-ndot-fused/story.py','--stop-after 65536'),
              ('restore','tiny-llm-ndot-fused/story.py','--stop-after 65792'),
              ('full','tiny-llm-ndot-fused/story.py',''),
              ('stock','tiny-llm-ndot/stock_prefix.py','--tokens 16384')]
    for name,worker,extra in stages:
        if name == 'stock':
            args = f'{PRIVATE}/experiments/{worker} --upstream {REMOTE}/.upstream/qwenputer --inputs {PRIVATE}/{relative} --out {base}/stock {extra}'
            script += "$env:PYTORCH_ALLOC_CONF='expandable_segments:True'\n"
        else:
            args = f'{PRIVATE}/experiments/{worker} --repo {PRIVATE} --upstream {REMOTE}/.upstream/qwenputer --inputs {PRIVATE}/{relative} --out {base}/main --max-seconds 14100 {extra}'
        action = f'''
tasklist /FI "IMAGENAME eq python.exe" /FO CSV | Out-File -Append '{base}/tasklist.log'
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.free --format=csv | Out-File -Append '{base}/gpu-before.csv'
if(Test-Path '{base}/main/scratch') {{Remove-Item -Recurse -Force '{base}/main/scratch'}}
$p=Start-Process -FilePath '{REMOTE}/.venv/Scripts/python.exe' -ArgumentList '{args}' -WorkingDirectory '{REMOTE}' -RedirectStandardOutput '{base}/{name}.stdout.log' -RedirectStandardError '{base}/{name}.stderr.log' -PassThru
$handle=$p.Handle
$p.WaitForExit()
@{{exit_code=$p.ExitCode;finished_utc=[DateTime]::UtcNow.ToString('o')}} | ConvertTo-Json | Set-Content '{base}/{name}.exit.json'
if($p.ExitCode -ne 0) {{throw ('Stage {name} failed with exit '+$p.ExitCode)}}
'''
        if name != 'stock':
            action += f"Copy-Item '{base}/main/report.json' '{base}/main/{name}-report.json'\n"
        script += locked(action,'tiny-qwen35-'+label+'-'+name,235 if name == 'full' else 60)
    script += f"@{{complete=$true}} | ConvertTo-Json | Set-Content '{base}/sequence.json'\n"
    script = f'''$created=$false
try {{
{script}
}} catch {{
 if($created) {{
  @{{complete=$false;error=($_ | Out-String);utc=[DateTime]::UtcNow.ToString('o')}} | ConvertTo-Json | Set-Content '{base}/sequence-failure.json'
 }}
 throw
}}
'''
    launcher = ROOT/'.cache/tiny-qwen35'/('gpu-'+label+'.ps1')
    launcher.write_text(script)
    paths = ['experiments/tiny-qwen35','experiments/tiny-llm-ndot-fused','experiments/tiny-llm-ndot',
             'experiments/neural-accelerator','experiments/dmc-feedback-fusion','experiments/dmc-rom-fusion',
             'experiments/memory-map/prepare.py','experiments/dmc-cuda/source/helpers/state.py',
             launcher.relative_to(ROOT).as_posix()]
    paths += [relative+'/'+name for name in ['engine.elf','model.bin','reference.json','reference.tokens.u32']]
    # A private mirror prevents syncing over the other task's active source tree.
    windows_private = PRIVATE.replace('/','\\')
    subprocess.run(['ssh','gpu',f'if not exist "{windows_private}" mkdir "{windows_private}"'],check=True)
    archive = subprocess.Popen(['tar','--exclude=__pycache__','--exclude=._*','-cf','-',*paths],
                               cwd=ROOT,stdout=subprocess.PIPE,env={**__import__('os').environ,'COPYFILE_DISABLE':'1'})
    try:
        subprocess.run(['ssh','gpu',f'tar -xf - -C {PRIVATE}'],stdin=archive.stdout,check=True)
    finally:
        archive.stdout.close()
    assert archive.wait() == 0
    child = base64.b64encode(f"& '{PRIVATE}/{launcher.relative_to(ROOT).as_posix()}'".encode('utf-16le')).decode()
    command = "Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='powershell -NoProfile -ExecutionPolicy Bypass -EncodedCommand "+child+"'} | ConvertTo-Json"
    encoded = base64.b64encode(command.encode('utf-16le')).decode()
    result = subprocess.run(['ssh','gpu','powershell -NoProfile -EncodedCommand '+encoded],
                            capture_output=True,text=True,check=True)
    assert json.loads(result.stdout)['ReturnValue'] == 0,result.stdout
    report = dict(remote=base,private_source=PRIVATE,stdout=result.stdout,stderr=result.stderr,
                  mac_gate=accepted,launcher_sha256=hashlib.sha256(launcher.read_bytes()).hexdigest())
    (HERE/'evidence/neural-launch.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--legacy',type=Path,required=True)
    p.add_argument('--program',type=Path,required=True)
    p.add_argument('--quality',type=Path,required=True)
    p.add_argument('--label',required=True)
    a = p.parse_args()
    launch(a.legacy,a.program,a.quality,a.label)
