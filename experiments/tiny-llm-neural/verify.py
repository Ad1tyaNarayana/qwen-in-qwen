"""Independently compare the retained actual GPU artifacts with the Mac reference."""
import argparse
import bisect
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--actual',type=Path,required=True)
p.add_argument('--reference',type=Path,default=ROOT / '.cache/tiny-llm/neural-stories')
p.add_argument('--evidence',type=Path,default=Path(__file__).parent / 'evidence')
a = p.parse_args()
a.evidence.mkdir(parents=True,exist_ok=True)
r = json.loads((a.reference / 'reference.json').read_text())
g = json.loads((a.actual / 'report.json').read_text())
assert g['complete'] and g['mismatches'] == 0 and g['halted']
assert g['generated_tokens'] == r['tokens']
assert g['peak_allocated_bytes'] < 8*2**30
for name,digest in g['scripts'].items():
    assert hashlib.sha256((ROOT / name.replace('\\','/')).read_bytes()).hexdigest() == digest
acceptance = json.loads((ROOT / 'experiments/dmc-feedback-fusion/evidence/verify.json').read_text())
scripts = {name.replace('\\','/'):digest for name,digest in g['scripts'].items()}
assert scripts['experiments/dmc-feedback-fusion/fusion.py'] == acceptance['scripts']['fusion.py']
assert scripts['experiments/dmc-rom-fusion/fusion.py'] == acceptance['rom_fusion_sha256']
actual = np.memmap(a.actual / 'actual.tokens.u32',mode='r',dtype='<u4')
expected = np.memmap(a.reference / 'reference.tokens.u32',mode='r',dtype='<u4')
assert len(actual) == len(expected) == r['tokens']
mismatches = 0
for start in range(0,len(actual),65536):
    mismatches += int(np.count_nonzero(actual[start:start+65536] != expected[start:start+65536]))
assert mismatches == 0
sys.path.insert(0,str(a.reference / 'reference-package'))
from chipformer.rv import spec
byte_base,byte_size = spec.FAMILIES['BYTE']
assert byte_size >= 256
raw = (actual[(actual >= byte_base) & (actual < byte_base+256)]-byte_base).astype('u1').tobytes()
assert raw == (a.actual / 'actual-output.bin').read_bytes() == (a.reference / 'reference-output.bin').read_bytes()
ram = (a.actual / 'actual-ram.bin').read_bytes()
assert len(ram) == 65536 and ram == (a.reference / 'reference-ram.bin').read_bytes()
assert hashlib.sha256((a.actual / 'actual.tokens.u32').read_bytes()).hexdigest() == r['sha256']
records = [list(map(int,m)) for m in re.findall(rb'^@(\d+) (\d+) (\d+) (\d+) (\d+)\n',raw,re.M)]
assert len(records) == r['steps']
chunks = re.split(rb'^@\d+ \d+ \d+ \d+ \d+\n',raw,flags=re.M)[1:]
assert all(c.startswith(b'>') and c.endswith(b'\n') for c in chunks)
text = b''.join(c[1:-1] for c in chunks).decode()
first_decode = len(r['prompt_tokens'])-1
chunks = [json.loads(line) for line in (a.actual / 'chunks.jsonl').read_text().splitlines()]
ends, times = [0], [0.]
for chunk in chunks:
    end = chunk['end_offset']
    assert end == ends[-1]+chunk['amount']
    assert chunk['compute_seconds'] >= times[-1]
    raw_chunk = actual[ends[-1]:end].tobytes()
    assert hashlib.sha256(raw_chunk).hexdigest() == chunk['sha256']
    ends.append(end)
    times.append(chunk['compute_seconds'])
assert ends[-1] == len(actual) and times[-1] == g['compute_seconds']

def time_bounds(offset):
    i = bisect.bisect_left(ends,offset)
    return (times[i],times[i]) if ends[i] == offset else (times[i-1],times[i])

positions = []
for pos in r['positions']:
    t = pos['trace_tokens']
    start_low,start_high = time_bounds(pos['end_token']-t)
    end_low,end_high = time_bounds(pos['end_token'])
    positions.append(dict(pos,phase='prefill' if pos['position'] < first_decode else 'decode',
        minutes_from_whole_run_rate=t/g['tokens_per_second']/60,phase0_projected_minutes=t/4380/60,
        measured_compute_minutes_bracket=[max(0,end_low-start_high)/60,(end_high-start_low)/60]))
summary = dict(kind='Independent Mac validation of actual self-fed GPU trace and recurrent RAM',
    verified_utc=datetime.now(timezone.utc).isoformat(),tokens_checked=len(actual),mismatches=mismatches,
    all_ram_bytes_equal=True,ram_bytes_checked=len(ram),output_bytes_equal=True,generated_text=text,
    prompt_tokens=r['prompt_tokens'],prediction_ids=[p[2] for p in records],
    generated_prediction_ids=[p[2] for p in records[first_decode:]],llm_records=records,
    positions=positions,inference_positions=r['steps'],prefill_positions=first_decode,
    generated_predictions=r['steps']-first_decode,tokens_per_second=g['tokens_per_second'],
    compute_minutes=g['compute_seconds']/60,
    retained_worker_minutes=g['retained_worker_seconds']/60,
    mean_minutes_per_inference_position=g['compute_seconds']/r['steps']/60,
    trace_sha256=r['sha256'],ram_sha256=hashlib.sha256(ram).hexdigest(),
    chunk_timing_sha256=hashlib.sha256((a.actual / 'chunks.jsonl').read_bytes()).hexdigest(),
    position_timing_scope='Bounds use the actual cumulative generation timers on either side of each reference instruction marker. No within-chunk interpolation.',
    output_source='Actual GPU trace BYTE records. Reference used only for comparison.',
    rate_scope='One retained full execution. Generation timer includes CUDA graph replay, feedback, token copies and CPU transfer; excludes setup, checks, IO, observation, checkpoints and restart downtime.')
for name,value in [('reference.json',r),('gpu.json',g),('validation.json',summary)]:
    (a.evidence / name).write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(summary,indent=2))
