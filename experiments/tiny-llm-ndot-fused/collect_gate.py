"""Export gate reports and retain failed startup separately from accepted timing."""
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[2] / 'tools'))
from remote_config import SSH_HOST, REMOTE_ROOT, LOCK, TRAIN_ROOT
import argparse
import json
from pathlib import Path
import subprocess
p=argparse.ArgumentParser()
p.add_argument('--base',required=True)
p.add_argument('--extra',required=True)
a=p.parse_args()
for base in (a.base,a.extra):
    assert base.startswith((REMOTE_ROOT + '/artifacts/tiny-llm-ndot-fused-'))
out=Path(__file__).with_name('evidence')
private=Path(__file__).resolve().parents[2]/('.cache/tiny-llm/'+a.base.rsplit('/',1)[-1]+'-actual')
private.mkdir(parents=True,exist_ok=True)


def pull(base,files,directory):
    directory.mkdir(parents=True,exist_ok=True)
    proc=subprocess.Popen(['ssh',SSH_HOST,'tar -cf - -C '+base+' '+' '.join(files)],stdout=subprocess.PIPE)
    try:
        subprocess.run(['tar','-xf','-','-C',str(directory)],stdin=proc.stdout,check=True)
    finally:
        proc.stdout.close()
    assert proc.wait()==0


files=['baseline/isolation.json','fused/isolation.json','comparison.json','apply.json','feedback.json',
       'apply.exit.json','feedback.exit.json','baseline.exit.json','fused.exit.json','compare.exit.json',
       'bench.exit.json','bench.stderr.log','tasklist.log','venv.log']
pull(a.base,files,private)
extra=['bench/bench.json','apply.json','profile/report.json','sequence.json','bench.exit.json','apply.exit.json','profile.exit.json','tasklist.log','venv.log']
pull(a.extra,extra,private/'extra')
for source,target in [('baseline/isolation.json','baseline-isolation.json'),('fused/isolation.json','fused-isolation.json'),
                      ('comparison.json','comparison.json'),('apply.json','initial-apply-race.json'),('feedback.json','feedback-race.json'),
                      ('extra/bench/bench.json','bench.json'),('extra/apply.json','apply-race.json'),('extra/profile/report.json','profile.json')]:
    (out/target).write_bytes((private/source).read_bytes())
for name in ('apply','feedback','baseline','fused','compare'):
    assert json.loads((private/(name+'.exit.json')).read_text(encoding='utf-8-sig'))['exit_code']==0
for name in ('bench','apply','profile'):
    assert json.loads((private/'extra'/(name+'.exit.json')).read_text(encoding='utf-8-sig'))['exit_code']==0
assert json.loads((private/'extra/sequence.json').read_text(encoding='utf-8-sig'))['complete']
(out/'benchmark-startup-failure.json').write_text(json.dumps(dict(exit_code=json.loads((private/'bench.exit.json').read_text(encoding='utf-8-sig'))['exit_code'],
    error=(private/'bench.stderr.log').read_text(),tokens_generated=0,repaired='Import fusion after runtime.build prepares chipformer',
    accepted_timing_artifacts=a.extra),indent=2)+'\n')
(out/'collection.json').write_text(json.dumps(dict(remote_artifacts=a.base,extra_artifacts=a.extra,private_export=str(private),files=files,extra_files=extra),indent=2)+'\n')
print(private)
