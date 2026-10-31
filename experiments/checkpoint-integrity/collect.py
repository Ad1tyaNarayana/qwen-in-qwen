"""Collect only small reports until complete, then raw traces for Mac validation."""
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[2] / 'tools'))
from remote_config import SSH_HOST, REMOTE_ROOT, LOCK, TRAIN_ROOT
import argparse
import subprocess
from pathlib import Path
import base64
ROOT=Path(__file__).resolve().parents[2]
BASE=(REMOTE_ROOT + '/artifacts/checkpoint-integrity-20261030a')
p=argparse.ArgumentParser();p.add_argument('--complete',action='store_true');a=p.parse_args()
subprocess.run(['ssh','-O','check',SSH_HOST],check=True)
out=ROOT/'.cache/checkpoint-integrity/actual-20261030a';out.mkdir(parents=True,exist_ok=True)
files=['mutations/report.json','mutations/head-names.json','mutations.exit.json','venv.log']
if a.complete:
 files+=['sequence.json','gpu-before.csv','sequence-initial-failure.json','scalar/full-initial-failure-report.json','scalar-full-initial-failure.exit.json','scalar-full-initial-failure.stderr.log']
 for guest in ['scalar','ndot']:
  files += [f'{guest}/{name}' for name in ['report.json','prefix-report.json','restore-report.json','full-report.json','actual.tokens.u32','actual-ram.bin','actual-output.bin','actual-registers.json','chunks.jsonl']]
  files += [f'{guest}-{stage}.exit.json' for stage in ['prefix','restore','full']]
archive=out/('complete.tar' if a.complete else 'mutations.tar')
with archive.open('wb') as f:
 subprocess.run(['ssh',SSH_HOST,f'tar -cf - -C {BASE} '+' '.join(files)],stdout=f,check=True)
subprocess.run(['tar','-xf',str(archive),'-C',str(out)],check=True)
print(out)
