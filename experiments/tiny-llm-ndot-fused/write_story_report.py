"""Write the dated result only after independent Mac final-state acceptance."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--actual',type=Path,required=True)
a=p.parse_args()
D=Path(__file__).resolve().parent
E=D/'evidence'
r=json.loads((E/'story-validation.json').read_text())
assert r['status']=='pass' and r['mismatches']==0 and r['positions']==38
run=json.loads((a.actual/'main/report.json').read_text())
assert run['complete'] and 'error' not in run
for source,target in [('main/report.json','story-run.json'),('checkpoint-metadata.json','story-checkpoint.json'),('collection.json','story-collection.json')]:
    (E/target).write_bytes((a.actual/source).read_bytes())
minutes=r['compute_seconds']/60
hours=r['compute_seconds']/3600
text=f'''## 2026-10-30 38-position autonomous story passes

The unified W=4 circuit with accepted ROM and feedback fusion generates all
{r['tokens']:,} trace tokens from ELF boot through HALT with zero mismatches
against the independent Mac trace. The prompt is `Once`. All 38 inference
positions complete, including prefill. The observed text is:

> {r['story_text']}

This text comes from actual self-fed GPU BYTE records. The Mac verifier
independently extracts those bytes from the exported actual token file and
compares the complete guest output with the native and software references.
All 65,536 final RAM bytes and all 32 registers in each of the three recurrent
register copies equal this ELF's own Mac reference literally. No state coordinate
is masked or normalized, and no tolerance is used.

The one complete execution measures {r['compute_seconds']:,.6f} generation seconds,
or {minutes:.3f} minutes and {hours:.3f} hours. Its rate is {r['tokens_per_second']:,.3f}
trace tokens/s and {r['min_per_position']:.4f} minutes per inference position.
These are measured full-run values. The timer includes graph replay, self-feedback,
synchronization, trace copies and CPU transfer. It excludes setup, validation,
observations, checkpoint IO and restart gaps. This is one full execution, separate
from the five-repeat bounded timing experiment. The longer context increases the
mean guest workload per position, so the earlier five-position projections use a
different denominator.

[Independent Mac acceptance](../tiny-llm-ndot-fused/evidence/story-validation.json)
checks every actual token, final RAM bytes, all three register copies, observed
output, timed chunk hashes and actual checkpoint restore boundaries. The run
restores actual generated GPU state at offsets 65,536 and 65,792 before continuing.
All recurrent S tensors, lag histories, next embedding and token come from the
GPU checkpoint. Ordinary and private outputs are asserted zero before saving.
The reference supplies comparisons and the stopping budget only. No reference
state enters after boot, and no checkpoint is synthesized by the host.

The 38-position fixture has 53,296,563 trace tokens. At the slower fused window
median it projected to 14,308.65 generation seconds, about 3.97 hours. This selected
38 positions within the requested approximate four-hour budget. The next position's
end boundary projects to 14,779.08 seconds before final diagnostic output, above
the conservative four-hour budget. [Budget counts](../tiny-llm-ndot-fused/evidence/story-budget.json)
remain reference-only projections. The guest model
file is byte-identical to M5. The generated unified circuit hash remains
`5e156bdb8fa6c543d96f0760a1a4d18dfea4cba3e558673a909bed1be0fde7da`.
See [fixture bindings](../tiny-llm-ndot-fused/evidence/story-fixture.json),
[the worker report](../tiny-llm-ndot-fused/evidence/story-run.json) and
[checkpoint provenance](../tiny-llm-ndot-fused/evidence/story-checkpoint.json).

The detached accepted directory is
`C:/Users/Admin/qwenputer-games-work/artifacts/tiny-llm-ndot-story-38-20261030b`.
The independent Mac fixture is `.cache/tiny-llm/ndot-story38-20261030`, and the actual
export is `.cache/tiny-llm/ndot-story38-20261030-actual`. Raw traces and checkpoints
stay private. GPU access uses `ssh gpu`, cmd and encoded PowerShell with the
verified existing venv. Jobs run sequentially and detach through
`Invoke-CimMethod Win32_Process Create`.

The first story directory `tiny-llm-ndot-story-38-20261030a` generated an exact
65,536-token prefix. Its restore setup refused the retained scratch package before
new generation. The launcher now removes its own scratch package between processes.
[The startup failure](../tiny-llm-ndot-fused/evidence/story-restore-startup-failure.json)
remains separate. Accepted run b starts fresh from boot. No failed numerical
comparison was retried or waived.

The fusion follow-up passes both full isolated 65,536-token windows and standalone
race stress. Five alternating timings measure 419.083 to 268.472 microseconds per
token at boot and 418.543 to 268.253 in the reference-seeded matvec window.
Those windows project the five-position M5 workload to 7.576 versus 4.853 and
7.566 versus 4.849 minutes per position. They are bounded-rate projections.
The complete 38-position result above is measured independently.
See [fusion acceptance and timing](../tiny-llm-ndot-fused/README.md).

There are no unresolved mismatches in the accepted runs. The exact fusion gate
covers this graph and the two stated windows; the story covers this complete
38-position execution. It is not a proof for arbitrary graphs or workloads.
All changes remain uncommitted. Upstream, circuit weights and the excluded
directories are unchanged. Work ran inline without independent agent review.
'''
ndot=D.parent/'tiny-llm-ndot/README.md'
s=ndot.read_text()
marker='## 2026-10-30 longer self-fed story run in progress'
assert marker in s
ndot.write_text(s[:s.index(marker)]+text)
fused=D/'README.md'
s=fused.read_text().replace('The longer story is running separately from boot.',
    'The separate 38-position story also passes independent Mac final-state validation.')
s+='\n## Complete autonomous story\n\n'+f'''The [dated story result](../tiny-llm-ndot/README.md#2026-10-30-38-position-autonomous-story-passes)
reports {r['tokens']:,} actual self-fed GPU trace tokens, zero mismatches, {r['tokens_per_second']:,.3f}
trace tokens/s and {r['min_per_position']:.4f} measured minutes per inference position.
The observed text is `{r['story_text']}`. All final RAM bytes and all three register
copies match independently on the Mac. Actual GPU checkpoints restore at 65,536
and 65,792 tokens. [story-validation.json](evidence/story-validation.json) retains
all acceptance checks. No reference-seeded state enters the story.
'''
fused.write_text(s)
checks=json.loads((E/'checks.json').read_text())
assert checks['status']=='pass'
final=dict(status='pass',fusion=checks,story=dict(tokens=r['tokens'],positions=r['positions'],mismatches=0,
    all_final_ram_bytes_equal=True,all_three_register_copies_equal=True,autonomous=True,
    trace_sha256=r['trace_sha256'],compute_seconds=r['compute_seconds']),
    evidence_sha256={name:hashlib.sha256((E/name).read_bytes()).hexdigest() for name in
                    ('checks.json','story-validation.json','story-run.json','story-checkpoint.json')})
(E/'final-checks.json').write_text(json.dumps(final,indent=2)+'\n')
with (D/'decisions.tsv').open('a') as f:
    f.write('\t'.join([datetime.now(timezone.utc).isoformat(),'complete','Accept the full autonomous story after independent Mac comparison',
        'Every token, final RAM byte, register copy, output byte and checkpoint boundary passes',
        'experiments/tiny-llm-ndot-fused/evidence/story-validation.json','38 positions; zero mismatches; no reference state after boot'])+'\n')
print(json.dumps(final,indent=2))
