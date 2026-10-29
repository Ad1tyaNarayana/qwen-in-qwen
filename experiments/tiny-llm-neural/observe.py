"""Decode emitted bytes from a retained actual-GPU token file."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--package',type=Path,required=True)
p.add_argument('--trace',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
a = p.parse_args()
sys.path.insert(0,str(a.package.resolve()))
from chipformer.rv import spec
import numpy as np
trace = np.memmap(a.trace,mode='r',dtype='<u4')
base, size = spec.FAMILIES['BYTE']
assert size >= 257
raw = (trace[(trace >= base) & (trace < base+256)]-base).astype('u1').tobytes()
report = dict(kind='Observed BYTE records from actual self-fed GPU trace',trace_tokens=len(trace),
              trace_sha256=hashlib.sha256(a.trace.read_bytes()).hexdigest(),output=raw.decode(errors='replace'),
              output_sha256=hashlib.sha256(raw).hexdigest(),reference_data_used=False)
a.out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
