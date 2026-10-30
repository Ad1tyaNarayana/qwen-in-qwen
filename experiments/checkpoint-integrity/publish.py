"""Publish only independently accepted controlled measurements and dated corrections."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
EVIDENCE=ROOT/'experiments/checkpoint-integrity/evidence'


def section(path,tag,body):
    text=path.read_text();begin=f'<!-- {tag}_BEGIN -->';end=f'<!-- {tag}_END -->'
    replacement=begin+'\n'+body.strip()+'\n'+end
    if begin in text:
        left=text.index(begin);right=text.index(end,left)+len(end);text=text[:left]+replacement+text[right:]
    else:text+='\n\n'+replacement+'\n'
    path.write_text(text)


def main():
    d=json.loads((EVIDENCE/'controlled-measurement.json').read_text());assert d['status']=='pass'
    a,b=d['runs']['scalar'],d['runs']['ndot'];ratio=d['scalar_over_ndot_generation_ratio']
    table='| Build | Generation seconds | CPU trace tokens | Trace tokens/s | Min/inference position |\n| --- | ---: | ---: | ---: | ---: |\n'
    for label,r in [('Scalar',a),('NDOT4',b)]:
        table+=f"| {label} | {r['generation_seconds']:,.2f} | {r['tokens']:,} | {r['tokens_per_second']:,.2f} | {r['minutes_per_position']:.4f} |\n"
    scope=f"One complete run per binary, from boot on the same unified W=4 compiled Qwen3.5-architecture network, with identical model data, prompt `[1, 403]`, a five-position budget and compact fused executor. Every token matches each binary's own independent reference, as do final RAM and all register copies. Both prediction-ID sequences are `{a['prediction_ids']}`. The shared stock-emitted BF16 weight identity is `0e515bbb…`; compact evaluation is fp32, bound to the same constructor. Stock `Qwen3_5ForCausalLM.generate` coverage remains bounded prefixes only, separate from these full runs."
    scope+=" The scalar worker initially stopped after generation because its older reference JSON omitted final registers. A fresh local replay reproduced its entire reference trace and RAM literally; validation then restored the terminal actual GPU checkpoint without new generation. The error, repair and validation-only restart are retained in the evidence."
    body=scope+'\n\n'+table+f'\nScalar generation time divided by NDOT4 generation time is **{ratio:.4f}×**. This is a same-network guest accelerator comparison, not an executor-only speedup or a repeated-run benchmark.\n\n![Complete same-network scalar and NDOT4 generation times](figures-llm/controlled-measurement.svg)\n\n*One complete run per build. Generation includes feedback, synchronization, copies and CPU transfer. Setup, comparisons, IO, checkpoint and restart gaps are excluded.*'
    section(ROOT/'docs/articles/tiny-llm-inside-qwen.md','CONTROLLED_MEASUREMENT',body)
    files=['experiments/tiny-llm-ndot/README.md','experiments/tiny-llm-ndot-fused/README.md','experiments/tiny-llm-neural/README.md','experiments/showcase/README.md']
    for file in files:
        note=f'''## 2026-10-31 timing correction after the audit

The historical approximately 3.4× ratio mixes an original-unfused projection with a shorter measured run. The approximately 5.3× ratio compares projections. The 1.86× historical ratio changes both network and executor. These are not measured same-network accelerator gains. Earlier projection tables retain their stated scope and are not measurements of complete scalar execution.

{scope}

{table}
Scalar / NDOT4 generation time is **{ratio:.4f}×**, one complete execution each. This is an accelerator comparison with the same fused executor on both sides. It does not apply the shorter fixture's ratio to the 38-position story. Generation timing excludes setup, comparisons, IO, checkpoint/restart gaps. The audit report is unchanged.

Evidence: [controlled measurement](../checkpoint-integrity/evidence/controlled-measurement.json), independently checked by [verify.py](../checkpoint-integrity/verify.py).'''
        section(ROOT/file,'AUDIT_TIMING_CORRECTION',note)
    # No such ratios occur in the Doom article; preserve its historical frame timings.
    section(ROOT/'docs/articles/doom-on-qwenputer.md','LLM_TIMING_CORRECTION',f'''## 2026-10-31 LLM timing correction

The [second article](tiny-llm-inside-qwen.md) replaces earlier LLM timing shorthand with a controlled same-network comparison. Its one complete scalar run and one complete NDOT4 run use the unified W=4 compiled Qwen3.5-architecture network and compact fused executor. The measured scalar/NDOT4 generation-time ratio is {ratio:.4f}× for that five-position stories260K fixture. Stock generation remains bounded-prefix evidence only. This does not alter the Doom measurements above or turn a historical cross-network comparison into an executor speedup.''')
    manifest=ROOT/'experiments/showcase/manifest/stories260k-38.json'
    data=json.loads(manifest.read_text())
    data['timing_correction_2026_10_31']={'historical_ratios':'3.4x mixes projection and measurement; 5.3x compares projections; 1.86x changes network and executor. None is a measured same-network accelerator gain.', 'controlled_five_position_evidence':'experiments/checkpoint-integrity/evidence/controlled-measurement.json','controlled_scalar_over_ndot_generation_ratio':ratio,'scope':'One complete five-position execution per build, same unified W=4 network and compact fused executor. This ratio is not applied to the 38-position story. Stock remains bounded-prefix evidence.'}
    manifest.write_text(json.dumps(data,indent=2)+'\n')
    (EVIDENCE/'timing-correction.json').write_text(json.dumps(data['timing_correction_2026_10_31'],indent=2)+'\n')
    ledger=ROOT/'docs/articles/SOURCES-LLM.md';text=ledger.read_text().replace('Pending until this evidence file exists and says pass. One complete run per binary','Accepted only after this evidence file exists and says pass. One complete run per binary');ledger.write_text(text)
    (EVIDENCE/'corrected-files.json').write_text(json.dumps(files+['docs/articles/doom-on-qwenputer.md','experiments/showcase/manifest/stories260k-38.json','experiments/showcase/make_manifests.py'],indent=2)+'\n')


if __name__=='__main__':main()
