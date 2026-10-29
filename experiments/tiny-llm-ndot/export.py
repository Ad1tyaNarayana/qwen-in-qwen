"""Export actual-state checkpoint provenance and a bounded artifact file list."""
import argparse
import hashlib
import json
from pathlib import Path
import torch

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--base',type=Path,required=True)
parser.add_argument('--stock-base',type=Path)
args=parser.parse_args()
if args.stock_base:
    assert json.loads((args.stock_base/'sequence.json').read_text())['complete']
    assert json.loads((args.stock_base/'stock/report.json').read_text())['status']=='pass'
else:
    assert json.loads((args.base/'sequence.json').read_text())['complete']
report=json.loads((args.base/'main/report.json').read_text())
assert report['complete'] and report['mismatches']==0
checkpoint=args.base/'main/checkpoint.pt'
state=torch.load(checkpoint,map_location='cpu',weights_only=True)
assert state['offset']==report['generated_tokens'] and state['compute_seconds']==report['compute_seconds']
r=dict(checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),offset=state['offset'],
       token=state['tok'].tolist(),binding=state['binding'],compute_seconds=state['compute_seconds'],
       heads=[{k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in head.items()} for head in state['heads']],
       source='Actual generated GPU state saved by worker; no reference state restored after initial boot')
(args.base/'checkpoint-metadata.json').write_text(json.dumps(r,indent=2)+'\n')
files=[]
for folder in args.base.iterdir():
    if folder.is_dir():
        for name in ('report.json','prefix-report.json','resume-report.json','actual.tokens.u32','actual-ram.bin','actual-output.bin','actual-registers.json','chunks.jsonl'):
            path=folder/name
            if path.exists(): files.append(path.relative_to(args.base).as_posix())
    elif folder.suffix in ('.json','.log','.csv'):
        files.append(folder.name)
(args.base/'export-files.txt').write_text('\n'.join(sorted(files))+'\n')
print(json.dumps(dict(offset=state['offset'],files=len(files))))
