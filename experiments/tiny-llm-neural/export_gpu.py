"""Export checkpoint provenance and wall-clock metadata without exporting tensors."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
out = ROOT.parent / 'artifacts/tiny-llm-neural'
report = json.loads((out / 'report.json').read_text())
assert report['complete'] and report['halted'] and report['mismatches'] == 0
prefix = json.loads((out / 'prefix-report.json').read_text())
resume = json.loads((out / 'resume-report.json').read_text())
checkpoint = out / 'checkpoint.pt'
state = torch.load(checkpoint,map_location='cpu',weights_only=True)
assert state['offset'] == report['generated_tokens']
assert state['compute_seconds'] == report['compute_seconds']
assert prefix['generated_tokens'] == resume['started_from_checkpoint_offset'] == 65536
assert resume['generated_tokens'] == report['started_from_checkpoint_offset'] == 65792
start = (out / 'prefix-report.json').stat().st_mtime-prefix['process_wall_seconds']
end = (out / 'report.json').stat().st_mtime
metadata = dict(kind='Metadata read from the actual GPU checkpoint and completed worker reports',
    checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    checkpoint_offset=state['offset'],checkpoint_token=state['tok'].tolist(),
    binding=state['binding'],heads=[{k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in h.items()}
                                  for h in state['heads']],
    first_successful_worker_start_utc=datetime.fromtimestamp(start,timezone.utc).isoformat(),
    final_report_utc=datetime.fromtimestamp(end,timezone.utc).isoformat(),
    elapsed_wall_seconds_including_restart_gaps=end-start,
    sum_successful_process_wall_seconds=sum(r['process_wall_seconds'] for r in (prefix,resume,report)),
    wall_scope='First successful prefix worker start estimated from report mtime minus its perf-counter duration, through completed full report mtime. Includes setup, IO, checkpointing, observation, and hand-off gaps. Excludes the failed SSH-owned launch that produced no tokens.',
    checkpoint_source='worker.save after FreeRun generation; no reference snapshot is ever restored',
    prefix_mismatches=prefix['mismatches'],resume_mismatches=resume['mismatches'])
(out / 'export.json').write_text(json.dumps(metadata,indent=2)+'\n')
print(json.dumps({k:v for k,v in metadata.items() if k not in ['heads','binding']},indent=2))
