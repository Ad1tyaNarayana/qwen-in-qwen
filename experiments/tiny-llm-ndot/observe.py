"""Decode a copied live GPU token prefix and compare it independently on Mac."""
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
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--label',required=True)
parser.add_argument('--evidence',type=Path,required=True)
args=parser.parse_args()
assert args.label.replace('-','').isalnum()
base=(REMOTE_ROOT + '/artifacts/tiny-llm-ndot-')+args.label+'/main'
script=f"$ProgressPreference='SilentlyContinue'; Copy-Item '{base}/actual.tokens.u32' '{base}/live.tokens.u32' -Force"
encoded=base64.b64encode(script.encode('utf-16-le')).decode()
subprocess.run(['ssh',SSH_HOST,'powershell -NoProfile -EncodedCommand '+encoded],check=True)
directory=ROOT/'.cache/tiny-llm'/('ndot-live-'+args.label)
directory.mkdir(exist_ok=True)
stream=subprocess.Popen(['ssh',SSH_HOST,f'tar -cf - -C {base} live.tokens.u32'],stdout=subprocess.PIPE)
subprocess.run(['tar','-xf','-','-C',str(directory)],stdin=stream.stdout,check=True)
stream.stdout.close()
assert stream.wait()==0
actual=np.fromfile(directory/'live.tokens.u32',dtype='<u4')
fixture=ROOT/'.cache/tiny-llm/ndot5-20261030'
expected=np.memmap(fixture/'reference.tokens.u32',mode='r',dtype='<u4')
assert np.array_equal(actual,expected[:len(actual)])
sys.path.insert(0,str(fixture/'reference-package'))
from chipformer.rv import spec
base,size=spec.FAMILIES['BYTE']
observed=(actual[(actual>=base)&(actual<base+256)]-base).astype('u1').tobytes()
report=dict(tokens_checked=len(actual),mismatches=0,output=observed.decode(),
            source='Snapshot copied from actual self-fed GPU token file during execution')
args.evidence.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
