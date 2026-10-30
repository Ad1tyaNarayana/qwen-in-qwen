"""Collect only small reports until complete, then raw traces for Mac validation."""
import argparse
import subprocess
from pathlib import Path
import base64
ROOT=Path(__file__).resolve().parents[2]
BASE='C:/Users/Admin/qwenputer-games-work/artifacts/checkpoint-integrity-20261030a'
p=argparse.ArgumentParser();p.add_argument('--complete',action='store_true');a=p.parse_args()
subprocess.run(['ssh','-O','check','gpu'],check=True)
out=ROOT/'.cache/checkpoint-integrity/actual-20261030a';out.mkdir(parents=True,exist_ok=True)
files=['mutations/report.json','mutations/head-names.json','mutations.exit.json','venv.log']
if a.complete:
 files+=['sequence.json','gpu-before.csv','sequence-initial-failure.json','scalar/full-initial-failure-report.json','scalar-full-initial-failure.exit.json','scalar-full-initial-failure.stderr.log']
 for guest in ['scalar','ndot']:
  files += [f'{guest}/{name}' for name in ['report.json','prefix-report.json','restore-report.json','full-report.json','actual.tokens.u32','actual-ram.bin','actual-output.bin','actual-registers.json','chunks.jsonl']]
  files += [f'{guest}-{stage}.exit.json' for stage in ['prefix','restore','full']]
archive=out/('complete.tar' if a.complete else 'mutations.tar')
with archive.open('wb') as f:
 subprocess.run(['ssh','gpu',f'tar -cf - -C {BASE} '+' '.join(files)],stdout=f,check=True)
subprocess.run(['tar','-xf',str(archive),'-C',str(out)],check=True)
print(out)
