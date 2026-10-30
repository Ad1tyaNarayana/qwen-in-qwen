"""Acquire the shared GPU lock, tar-sync dependencies, and detach a fused capture."""
import argparse
import base64
from datetime import datetime,timezone
from pathlib import Path
import re
import subprocess
import io
import tarfile
from common import HERE, ROOT

REMOTE='C:/Users/Admin/qwenputer-games-work'
LOCK=r'C:\Users\Admin\gpu-lock'

def ps(script):
    encoded=base64.b64encode(script.encode('utf-16-le')).decode()
    return subprocess.check_output(['ssh','gpu','powershell -NoProfile -EncodedCommand '+encoded],text=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--label',required=True)
    p.add_argument('--inputs',type=Path,default=ROOT/'.cache/tiny-llm/ndot-story38-20261030')
    p.add_argument('--stop',type=int)
    p.add_argument('--every',type=int,default=32768)
    a=p.parse_args()
    if not re.fullmatch('[A-Za-z0-9-]+',a.label): p.error('label must be alphanumeric with hyphens')
    if a.every<=0 or a.stop is not None and a.stop<=0: p.error('positive token counts required')
    inputs=a.inputs.resolve().relative_to(ROOT).as_posix()
    base=f'{REMOTE}/artifacts/explorer-{a.label}'
    repo=base+'/repo'
    acquired=subprocess.run(['ssh','gpu',f'mkdir {LOCK}']).returncode==0
    if not acquired:
        print('GPU lock held. No GPU job started. Retry after its owner releases it.')
        return 2
    detached=False
    try:
        owner=f'Qwenputer Explorer {a.label}; start={datetime.now(timezone.utc).isoformat()}; expected <=15 minutes'
        print(ps(f"Set-Content 'C:/Users/Admin/gpu-lock/owner.txt' '{owner}'; if(Test-Path '{base}') {{throw 'Fresh artifact path required'}}; New-Item -ItemType Directory '{repo}' -Force | Out-Null"))
        fixtures=[inputs+'/'+name for name in ('engine.elf','reference.json','pack.json') if (ROOT/inputs/name).exists()]
        paths=['experiments/explorer','experiments/tiny-llm-ndot','experiments/tiny-llm-ndot-fused',
            'experiments/neural-accelerator','experiments/memory-map/prepare.py','experiments/dmc-feedback-fusion',
            'experiments/dmc-rom-fusion','experiments/dmc-cuda/source/helpers/state.py',
            'experiments/showcase/manifest',*fixtures]
        send=subprocess.Popen(['tar','-cf','-','--exclude=__pycache__','--exclude=data','--exclude=screenshots','--exclude=*.u32','--exclude=reference-package','-C',str(ROOT),*paths],stdout=subprocess.PIPE)
        receive=subprocess.run(['ssh','gpu',f'tar -xf - -C {repo}'],stdin=send.stdout)
        send.stdout.close()
        assert send.wait()==receive.returncode==0
        # Comparison tokens are required; sync the fixture's trace separately.
        send=subprocess.Popen(['tar','-cf','-','-C',str(ROOT),inputs+'/reference.tokens.u32'],stdout=subprocess.PIPE)
        receive=subprocess.run(['ssh','gpu',f'tar -xf - -C {repo}'],stdin=send.stdout)
        send.stdout.close(); assert send.wait()==receive.returncode==0
        cmd=f"{repo}/experiments/explorer/record.py capture --upstream {REMOTE}/.upstream/qwenputer --inputs {repo}/{inputs} --out {base}/bundle --every {a.every}"+(f' --stop {a.stop}' if a.stop else '')
        script=f'''$ErrorActionPreference='Stop'
try {{
 & '{REMOTE}/.venv/Scripts/python.exe' -c "import sys; assert sys.prefix.endswith('.venv'); print(sys.executable)" | Out-File '{base}/venv.log'
 if($LASTEXITCODE -ne 0) {{throw 'Verified venv required'}}
 $p=Start-Process -FilePath '{REMOTE}/.venv/Scripts/python.exe' -ArgumentList '{cmd}' -WorkingDirectory '{REMOTE}' -RedirectStandardOutput '{base}/stdout.log' -RedirectStandardError '{base}/stderr.log' -PassThru
 $handle=$p.Handle
 if(-not $p.WaitForExit(900000)) {{Stop-Process -Id $p.Id; throw 'Explorer capture exceeded its 15-minute limit'}}
 @{{exit_code=$p.ExitCode}} | ConvertTo-Json | Set-Content '{base}/exit.json'
}} catch {{ @{{error=$_.ToString();exit_code=1}} | ConvertTo-Json | Set-Content '{base}/exit.json' }} finally {{ cmd /c 'rmdir /s /q C:\\Users\\Admin\\gpu-lock' }}
'''
        payload=io.BytesIO()
        with tarfile.open(fileobj=payload,mode='w') as archive:
            raw=script.encode('utf-8'); member=tarfile.TarInfo('supervisor.ps1'); member.size=len(raw)
            archive.addfile(member,io.BytesIO(raw))
        subprocess.run(['ssh','gpu',f'tar -xf - -C {base}'],input=payload.getvalue(),check=True)
        command=f"powershell -ExecutionPolicy Bypass -NoProfile -EncodedCommand "+base64.b64encode(f"& '{base}/supervisor.ps1'".encode('utf-16-le')).decode()
        result=ps("Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"+command+"'} | ConvertTo-Json")
        import json
        status=json.loads(result)
        assert status['ReturnValue']==0,status
        print(result)
        detached=True
        print(base)
        print('Collect only after exit.json appears. The detached process releases the GPU lock in finally.')
    finally:
        if not detached: subprocess.run(['ssh','gpu',f'rmdir /s /q {LOCK}'])
    return 0

if __name__=='__main__': raise SystemExit(main())
