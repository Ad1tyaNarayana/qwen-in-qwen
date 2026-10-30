"""Require complete full-state and timing evidence bound to the accepted sources."""
import hashlib
import json
import statistics
from pathlib import Path
D=Path(__file__).resolve().parent
E=D/'evidence'
load=lambda name:json.loads((E/name).read_text())
comparison=load('comparison.json')
assert set(comparison['windows'])=={'boot','ndot_matvec_reference_seeded'}
for w in comparison['windows'].values():
    assert w['tokens']==65536 and w['digest_mismatches']==0
for side in ('baseline','fused'):
    r=load(side+'-isolation.json')
    assert r['one_bit_recording_control']
    assert r['reference_seed_offset']==32768
    for w in r['windows'].values():
        assert w['tokens']==65536 and w['reference_mismatches']==0
        assert w['first_state_mismatch'] is None and w['final_S_reconstruction_equal']
        assert w['max_delta_words']<4096
    for name,digest in r['scripts'].items():
        assert hashlib.sha256((D/name).read_bytes()).hexdigest()==digest,name
apply=load('apply-race.json')
feedback=load('feedback-race.json')
assert apply['tokens']>=8192 and apply['kernels']['barrier']['mismatched_words']==0
assert feedback['queued_replays']>=8192 and feedback['mismatched_values']==0
for stress in (apply,feedback):
    for name,digest in stress['scripts'].items():
        assert hashlib.sha256((D/name).read_bytes()).hexdigest()==digest,name
profile=load('profile.json')
assert profile['status']=='pass' and profile['mismatches']==0
assert profile['warmup_tokens']==2048 and profile['profiled_tokens']==64
b=load('bench.json')
assert len(b['runs'])==20
for window in comparison['windows']:
    runs=[r for r in b['runs'] if r['window']==window]
    assert [(r['repeat'],r['side']) for r in runs]==[(i,s) for i in range(5) for s in ('baseline','fused')]
    assert all(r['tokens']==65536 and r['mismatches']==0 for r in runs)
    for side in ('baseline','fused'):
        values=[r['us_per_token'] for r in runs if r['side']==side]
        summary=b['windows'][window][side]
        assert summary['median_us_per_token']==statistics.median(values)
        assert summary['range_us_per_token']==[min(values),max(values)]
for name in ('fusion.py','runtime.py','isolation.py','bench.py','profile_fused.py'):
    assert hashlib.sha256((D/name).read_bytes()).hexdigest()==b['scripts'][name],name
report=dict(status='pass',isolated_tokens_per_side=131072,windows=2,reference_mismatches=0,
            full_state_digest_mismatches=0,standalone_queued_replays=8192,timing_repeats_per_side_per_window=5,
            timed_tokens=sum(r['tokens'] for r in b['runs']),profiled_checked_tokens=2112,standalone_source_bindings=True,timing_reference_mismatches=0,
            source_sha256={n:hashlib.sha256((D/n).read_bytes()).hexdigest() for n in ('fusion.py','runtime.py','isolation.py','bench.py')})
(E/'checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
