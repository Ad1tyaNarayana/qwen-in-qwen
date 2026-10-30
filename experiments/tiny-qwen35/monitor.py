"""Copy small live worker reports without starting Python or GPU work on the PC."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

from pack import ROOT


def fetch(path):
    result = subprocess.run(['ssh','-o','ConnectTimeout=10','gpu','type '+path.replace('/','\\')],
                            capture_output=True,timeout=20)
    return json.loads(result.stdout.decode('utf-8-sig')) if result.returncode == 0 else None


def monitor(base):
    assert base.startswith('C:/Users/Admin/qwenputer-games-work/artifacts/tiny-qwen35-')
    destination = ROOT/'.cache/tiny-qwen35/gpu-progress.json'
    while True:
        try:
            main = fetch(base+'/main/heartbeat.json')
            stock = fetch(base+'/stock/report.json')
            sequence = fetch(base+'/sequence.json')
            keys = ['complete','status','generated_tokens','total_tokens','mismatches','compute_seconds',
                    'tokens_per_second','checkpoint_offset','updated_utc','minimum_free_bytes','error']
            result = dict(sampled_utc=datetime.now(timezone.utc).isoformat(),remote=base,
                          scope='live worker reports; independent final Mac validation remains separate',
                          main={k:main[k] for k in keys if k in main} if main else None,
                          stock={k:stock[k] for k in keys if k in stock} if stock else None,
                          sequence=sequence)
            temp = destination.with_suffix('.tmp')
            temp.write_text(json.dumps(result,indent=2)+'\n')
            temp.replace(destination)
            print(json.dumps(result),flush=True)
            if sequence and sequence['complete']:
                return
            progress = json.loads((destination.parent/'progress.json').read_text())
            if progress['stage'] == 'failed':
                return
        except (subprocess.TimeoutExpired,json.JSONDecodeError) as error:
            print(json.dumps(dict(observation_error=repr(error))),flush=True)
        time.sleep(45)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',required=True)
    a = p.parse_args()
    monitor(a.base)
