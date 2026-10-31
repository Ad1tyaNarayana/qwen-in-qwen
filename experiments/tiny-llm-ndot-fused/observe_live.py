"""Read an actual trace prefix without altering the running GPU process."""
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[2] / 'tools'))
from remote_config import SSH_HOST, REMOTE_ROOT, LOCK, TRAIN_ROOT
import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
p=argparse.ArgumentParser()
p.add_argument('--base',required=True)
p.add_argument('--inputs',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
assert a.base.startswith((REMOTE_ROOT + '/artifacts/tiny-llm-ndot-story-'))
script=f'''$ProgressPreference='SilentlyContinue'
$f=[System.IO.File]::Open('{a.base}/main/actual.tokens.u32',[System.IO.FileMode]::Open,[System.IO.FileAccess]::Read,[System.IO.FileShare]::ReadWrite)
try {{
$n=[int]($f.Length - ($f.Length % 4))
$b=New-Object byte[] $n
$offset=0
while($offset -lt $n) {{$read=$f.Read($b,$offset,$n-$offset); if($read -eq 0) {{throw 'Short read'}}; $offset+=$read}}
[Convert]::ToBase64String($b)
}} finally {{$f.Dispose()}}
'''
encoded=base64.b64encode(script.encode('utf-16-le')).decode()
raw=base64.b64decode(subprocess.check_output(['ssh',SSH_HOST,'powershell -NoProfile -EncodedCommand '+encoded]))
actual=np.frombuffer(raw,dtype='<u4')
reference=np.memmap(a.inputs/'reference.tokens.u32',dtype='<u4',mode='r')
assert np.array_equal(actual,reference[:len(actual)])
sys.path.insert(0,str(a.inputs/'reference-package'))
from chipformer.rv import spec
byte_base,_=spec.FAMILIES['BYTE']
selected=actual[(actual>=byte_base)&(actual<byte_base+256)]
observed=(selected-byte_base).astype('u1').tobytes()
report=dict(source='Read-only snapshot of actual self-fed GPU trace; independent Mac prefix comparison',
            generated_tokens=len(actual),mismatches=0,sha256=hashlib.sha256(raw).hexdigest(),
            observed_byte_output=observed.decode(),remote_artifacts=a.base)
a.out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
