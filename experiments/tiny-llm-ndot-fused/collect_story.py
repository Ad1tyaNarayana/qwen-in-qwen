"""Collect actual GPU story artifacts for an independent Mac comparison."""
import argparse
import base64
import json
from pathlib import Path
import subprocess
p=argparse.ArgumentParser()
p.add_argument('--base',required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
assert a.base.startswith('C:/Users/Admin/qwenputer-games-work/artifacts/tiny-llm-ndot-story-')
remote='C:/Users/Admin/qwenputer-games-work'
script=f'''$ProgressPreference='SilentlyContinue'
$ErrorActionPreference='Stop'
tasklist /FI "IMAGENAME eq python.exe" /V
while(Get-CimInstance Win32_Process | Where-Object {{$_.Name -eq 'python.exe'}}) {{Start-Sleep -Seconds 15}}
& '{remote}/.venv/Scripts/python.exe' '{remote}/repo/experiments/tiny-llm-ndot/export.py' --base '{a.base}'
exit $LASTEXITCODE
'''
encoded=base64.b64encode(script.encode('utf-16-le')).decode()
subprocess.run(['ssh','gpu','powershell -NoProfile -EncodedCommand '+encoded],check=True)
files=['main/'+name for name in ('report.json','prefix-report.json','restore-report.json','full-report.json',
       'actual.tokens.u32','actual-ram.bin','actual-output.bin','actual-registers.json','chunks.jsonl')]
files+=['prefix.exit.json','restore.exit.json','full.exit.json','checkpoint-metadata.json','sequence.json','tasklist.log','venv.log','gpu-before.csv']
a.out.mkdir(parents=True,exist_ok=False)
proc=subprocess.Popen(['ssh','gpu','tar -cf - -C '+a.base+' '+' '.join(files)],stdout=subprocess.PIPE)
try:
    subprocess.run(['tar','-xf','-','-C',str(a.out)],stdin=proc.stdout,check=True)
finally:
    proc.stdout.close()
assert proc.wait()==0
(a.out/'collection.json').write_text(json.dumps(dict(remote_artifacts=a.base,files=files),indent=2)+'\n')
print(a.out)
