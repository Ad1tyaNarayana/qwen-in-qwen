"""Observe a detached story and run the independent Mac verifier when it exits."""
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[2] / 'tools'))
from remote_config import SSH_HOST, REMOTE_ROOT, LOCK, TRAIN_ROOT
import argparse
import base64
import bisect
import json
from pathlib import Path
import subprocess
import sys
import time
p=argparse.ArgumentParser()
p.add_argument('--base',required=True)
p.add_argument('--inputs',type=Path,required=True)
p.add_argument('--actual',type=Path,required=True)
a=p.parse_args()
assert a.base.startswith((REMOTE_ROOT + '/artifacts/tiny-llm-ndot-story-'))
D=Path(__file__).resolve().parent
meta=json.loads((a.inputs/'reference.json').read_text())
boundaries=[x['end_token'] for x in meta['positions']]
script=f'''$ProgressPreference='SilentlyContinue'
$base='{a.base}'
$r=@{{python=@(Get-CimInstance Win32_Process | Where-Object {{$_.Name -eq 'python.exe'}} | Select-Object ProcessId,CommandLine)}}
foreach($name in @('sequence.json','prefix.exit.json','restore.exit.json','full.exit.json','main/heartbeat.json','main/report.json')) {{
if(Test-Path "$base/$name") {{$r[$name]=Get-Content "$base/$name" -Raw | ConvertFrom-Json}}
}}
$r | ConvertTo-Json -Depth 6
'''
encoded=base64.b64encode(script.encode('utf-16-le')).decode()
while True:
    try:
        result=subprocess.check_output(['ssh',SSH_HOST,'powershell -NoProfile -EncodedCommand '+encoded],timeout=30)
        state=json.loads(result)
    except (subprocess.SubprocessError,json.JSONDecodeError) as error:
        print(json.dumps(dict(observation_error=str(error),generation_not_restarted=True)),flush=True)
        time.sleep(55)
        continue
    for name in ('prefix.exit.json','restore.exit.json','full.exit.json'):
        if name in state and state[name]['exit_code']!=0:
            raise RuntimeError(json.dumps(state))
    if 'error' in state.get('main/report.json',{}):
        raise RuntimeError(json.dumps(state))
    h=state.get('main/heartbeat.json',{})
    tokens=h.get('generated_tokens',0)
    print(json.dumps(dict(tokens=tokens,total_tokens=meta['tokens'],mismatches=h.get('mismatches'),
        compute_seconds=h.get('compute_seconds'),tokens_per_second=h.get('tokens_per_second'),
        checkpoint_offset=h.get('checkpoint_offset'),
        completed_position_boundaries=bisect.bisect_right(boundaries,tokens),
        live_python_processes=len(state['python']),sequence_complete=state.get('sequence.json',{}).get('complete',False))),flush=True)
    if state.get('sequence.json',{}).get('complete'):
        break
    time.sleep(55)
subprocess.run([sys.executable,str(D/'collect_story.py'),'--base',a.base,'--out',str(a.actual)],check=True)
subprocess.run([sys.executable,str(D/'verify_story.py'),'--inputs',str(a.inputs),'--actual',str(a.actual),
                '--out',str(D/'evidence/story-validation.json')],check=True)
print(json.dumps(dict(complete=True,validation=str(D/'evidence/story-validation.json'))),flush=True)
