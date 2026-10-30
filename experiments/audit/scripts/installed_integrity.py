#!/usr/bin/env python3
"""Check installed HF source against its wheel RECORD, without importing torch."""
import base64,hashlib,importlib.metadata,json,sys
from pathlib import Path
d=importlib.metadata.distribution('transformers')
names={'transformers/models/qwen3_5/modeling_qwen3_5.py','transformers/generation/utils.py','transformers/cache_utils.py','transformers/utils/generic.py'}
out=dict(version=d.version,optimize=sys.flags.optimize,scope='installed wheel RECORD integrity; not publisher attestation',files={})
for f in d.files:
    if str(f).replace('\\','/') not in names:continue
    raw=Path(d.locate_file(f)).read_bytes();h=hashlib.sha256(raw).digest();encoded=base64.urlsafe_b64encode(h).decode().rstrip('=')
    out['files'][str(f)]=dict(sha256=h.hex(),record_sha256=f.hash.value,record_matches=f.hash.mode=='sha256' and f.hash.value==encoded)
out['passed']=len(out['files'])==len(names) and all(v['record_matches'] for v in out['files'].values()) and out['optimize']==0
print(json.dumps(out,indent=2))
