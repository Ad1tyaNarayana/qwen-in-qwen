"""Collect only actual trace/state and small reports, excluding GPU checkpoints."""
import argparse
import json
from pathlib import Path
import subprocess
import tarfile
ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--remote',type=Path,default=ROOT/'.cache/showcase/remote.json')
    p.add_argument('--demo',nargs='+',choices=['sha256','life','tictactoe','mnist','mnist-scalar'])
    a=p.parse_args();remote=json.loads(a.remote.read_text())
    base=remote['base'];out=ROOT/'.cache/showcase'/('actual-'+remote['label'])
    out.mkdir(parents=True,exist_ok=True)
    paths=['venv.log']+([] if a.demo else ['tasklist.log','sequence.json'])
    for demo in a.demo or remote['demos']:
        paths += [f'{demo}/fused/{name}' for name in ['actual.tokens.u32','actual-ram.bin','actual-registers.json',
            'actual-output.bin','report.json','prefix-report.json','restore-report.json','full-report.json','chunks.jsonl']]
        paths += [f'{demo}/stock/report.json',f'{demo}/stock/actual.tokens.u32']
        for name in ['prefix','restore','full','stock']:
            paths += [f'{demo}-{name}.{suffix}' for suffix in ['exit.json','stdout.log','stderr.log']]
    archive=out/'actual.tar'
    with archive.open('wb') as f:
        subprocess.run(['ssh','gpu',f'tar -cf - -C {base} '+' '.join(paths)],stdout=f,check=True)
    with tarfile.open(archive) as tar:tar.extractall(out,filter='data')
    print(out)


if __name__=='__main__':main()
