"""Profile separately from timing and check every generated token."""
import argparse
import json
from pathlib import Path
import runtime
p=argparse.ArgumentParser()
p.add_argument('--inputs',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
a.out.mkdir(parents=True,exist_ok=False)
fr,_,_,metadata=runtime.build(a.out,a.inputs)
from fusion import install
fused=install(fr)
fr.capture()
fixtures,_=runtime.fixtures(fr,a.inputs,2112)
runtime.reset(fr,fixtures[0][1],fused)
import numpy as np
import torch
expected=fixtures[0][2]
assert np.array_equal(fr.run(2048)[0],expected[:2048])
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
    actual=fr.run(64)[0]
assert np.array_equal(actual,expected[2048:])
rows=prof.key_averages()
table=rows.table(sort_by='self_cuda_time_total',row_limit=20)
prof.export_chrome_trace(str(a.out/'profile.json'))
report=dict(status='pass',profiled_tokens=64,warmup_tokens=2048,mismatches=0,
            scope='Separate fused profiler sample, excluded from benchmark timers',table=table,
            batches=[dict(indices=b['indices'],value_heads=b['V'],widths=b['widths']) for b in fused['batches'].values()],
            metadata=metadata)
(a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(table)
