"""Collect successful GPU files and retain the separate failed startup evidence."""
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[2] / 'tools'))
from remote_config import SSH_HOST, REMOTE_ROOT, LOCK, TRAIN_ROOT
import argparse
import base64
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/neural-accelerator'))
from launch_legacy import REMOTE,sync

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--main-label',required=True)
parser.add_argument('--stock-label',required=True)
parser.add_argument('--memory-label')
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
for label in (args.main_label,args.stock_label,*([args.memory_label] if args.memory_label else [])):
    assert label.replace('-','').isalnum()
main=f'{REMOTE}/artifacts/tiny-llm-ndot-{args.main_label}'
stock=f'{REMOTE}/artifacts/tiny-llm-ndot-{args.stock_label}'
sync(['experiments/tiny-llm-ndot/export.py'])
script=f'''$ProgressPreference='SilentlyContinue'
tasklist /FI "IMAGENAME eq python.exe"
if(Get-CimInstance Win32_Process | Where-Object {{$_.Name -eq 'python.exe'}}) {{ throw 'Wait for GPU jobs to exit before export' }}
& '{REMOTE}/.venv/Scripts/python.exe' '{REMOTE}/repo/experiments/tiny-llm-ndot/export.py' --base '{main}' --stock-base '{stock}'
exit $LASTEXITCODE
'''
encoded=base64.b64encode(script.encode('utf-16-le')).decode()
subprocess.run(['ssh',SSH_HOST,'powershell -NoProfile -EncodedCommand '+encoded],check=True)

def pull(base,files,directory):
    directory.mkdir(parents=True,exist_ok=True)
    archive=subprocess.Popen(['ssh',SSH_HOST,'tar -cf - -C '+base+' '+' '.join(files)],stdout=subprocess.PIPE)
    try:
        subprocess.run(['tar','-xf','-','-C',str(directory)],stdin=archive.stdout,check=True)
    finally:
        archive.stdout.close()
    assert archive.wait()==0

files=['main/'+name for name in ('report.json','prefix-report.json','resume-report.json','actual.tokens.u32',
       'actual-ram.bin','actual-output.bin','actual-registers.json','chunks.jsonl')]
files+=['0-main.exit.json','2-main.exit.json','4-main.exit.json','checkpoint-metadata.json']
number=5
for index in range(5):
    for label in ('original-unfused','unified-original','unified-ndot'):
        name=f'{label}-{index}'
        files += [name+'/'+f for f in ('report.json','actual.tokens.u32','chunks.jsonl')]
        files.append(f'{number}-{name}.exit.json')
        number+=1
pull(main,files,args.out)
pull(stock,['stock/report.json','stock/actual.tokens.u32','stock-success.exit.json','import-control.log'],args.out)
pull(main,['20-stock.exit.json','20-stock.stderr.log','stock/report.json'],args.out/'stock-startup-failure')
if args.memory_label:
    memory=f'{REMOTE}/artifacts/tiny-llm-ndot-{args.memory_label}'
    pull(memory,['stock/report.json','stock/actual.tokens.u32','stock-success.exit.json'],args.out/'stock-memory-failure')
provenance=dict(main_artifacts=main,stock_artifacts=stock,
    main_sequence='Neural and all 15 baselines exit zero; old stock startup exits two and is retained separately',
    stock_sequence='Fresh stock-only run after helper rename; independent token verification still required',
    successful_exit_files=19,failed_startup='stock-startup-failure/20-stock.exit.json')
(args.out/'collection.json').write_text(json.dumps(provenance,indent=2)+'\n')
print(json.dumps(provenance,indent=2))
