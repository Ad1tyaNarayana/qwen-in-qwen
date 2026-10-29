"""Profile an independently checked unfused window outside reported timings."""
import argparse
from pathlib import Path
import json
import sys
import types

import worker

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo',type=Path,required=True)
parser.add_argument('--upstream',type=Path,required=True)
parser.add_argument('--inputs',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
parser.add_argument('--original-circuit',action='store_true')
args=parser.parse_args()
args.out.mkdir(parents=True,exist_ok=False)
import numpy as np
import torch
meta=json.loads((args.inputs/'reference.json').read_text())
fr,_,_,_=worker.build(args,meta)
expected=np.memmap(args.inputs/'reference.tokens.u32',mode='r',dtype='<u4')
with torch.no_grad():
    assert np.array_equal(fr.run(2048)[0],expected[:2048])
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
        actual=fr.run(64)[0]
    assert np.array_equal(actual,expected[2048:2112])
rows=prof.key_averages()
table=rows.table(sort_by='self_cuda_time_total',row_limit=20)
(args.out/'profile.txt').write_text(table)
prof.export_chrome_trace(str(args.out/'profile.json'))
report=dict(status='pass',original_circuit=args.original_circuit,profiled_tokens=64,warmup_tokens=2048,mismatches=0,
            scope='Separate profiler run excluded from benchmark timings',table=table)
(args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
