"""Reproducible sensitivity estimates. No neural timing is measured here."""
import json
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
E = json.loads((HERE.parent/'tiny-llm-rv32/evidence/evidence.json').read_text())
US = 228.155
ROW_RATIO = 10816/3512
MIXED_COST = {4:(1.45,1.9), 8:(1.6,2.3), 16:(1.9,3.1)}
MATRIX_COST = {4:(1.15,1.45), 8:(1.25,1.7), 16:(1.5,2.2)}


def estimate(model, pos, width, mixed):
    p = E['positions'][pos]
    groups = p['hypothetical_dot'][str(width)]['groups']
    mat_overhead = p['matvec_tokens'] - p['dot_body_tokens']
    other = p['trace_tokens'] - p['matvec_tokens']
    # Assumed scalar inner-loop cost for each QK and weighted-V MAC.
    qk = 5*64*(pos+1)
    vmac = qk
    remaining = other - 16*(qk+vmac)
    assert remaining >= 0
    if model == 'custom':
        groups = 2445312//width
        mat_overhead *= ROW_RATIO
        qk = 6*192*(pos+1)
        vmac = qk
        remaining_lo, remaining_hi = remaining*1.8, remaining*3.6
        qk_groups = 6*12*(pos+1)*((16+width-1)//width)
    else:
        remaining_lo = remaining_hi = remaining
        qk_groups = 5*8*(pos+1)*((8+width-1)//width)
    # Two dot instructions return halves, then a guest 64-bit add,
    # two pointer increments and branch: assumed 17..21 tokens/group.
    attention_lo = 17*qk_groups if mixed else 16*qk
    attention_hi = 21*qk_groups if mixed else 16*qk
    tokens_lo = 7*groups+mat_overhead+remaining_lo+16*vmac+attention_lo
    tokens_hi = 8*groups+mat_overhead+remaining_hi+16*vmac+attention_hi
    growth = (MIXED_COST if mixed else MATRIX_COST)[width]
    return dict(model=model, position=pos, width=width, qk_accelerated=mixed,
                token_range=[tokens_lo,tokens_hi], transition_growth=list(growth),
                minutes=[tokens_lo*US*growth[0]/60e6,tokens_hi*US*growth[1]/60e6])


rows = [estimate(m,p,w,mixed) for mixed in (False,True)
        for m in ('stories260K','custom') for w in (4,8,16) for p in (4,63)]
result = dict(scope='Conditional fused-executor sensitivity, not stock-HF timing',
              base_us=US, custom_row_ratio=ROW_RATIO, rows=rows,
              script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              source_evidence_sha256=hashlib.sha256((HERE.parent/'tiny-llm-rv32/evidence/evidence.json').read_bytes()).hexdigest())
(HERE/'estimates.json').write_text(json.dumps(result,indent=2)+'\n')
for mixed in (False,True):
    print('matrix+QK' if mixed else 'matrix only')
    for w in (4,8,16):
        vals = [r for r in rows if r['width']==w and r['qk_accelerated']==mixed]
        print(w, ' | '.join(f"{r['model']} p{r['position']}: {r['minutes'][0]:.2f}–{r['minutes'][1]:.2f}" for r in vals))

# Standalone chart for the design review. Ranges are assumptions, not error bars.
bars = []
for i,r in enumerate(x for x in rows if x['width']==8):
    lo,hi=r['minutes']
    label=f"{r['model']} p{r['position']} {'+QK' if r['qk_accelerated'] else 'matrix'}"
    y=24+i*32
    bars.append(f'<text x="0" y="{y+4}">{label}</text><line x1="{230+lo*5:.1f}" x2="{230+hi*5:.1f}" y1="{y}" y2="{y}" stroke="var(--chart-1)" stroke-width="8"/><text x="{240+hi*5:.1f}" y="{y+4}">{lo:.1f}–{hi:.1f}</text>')
html='''<!doctype html><html><head><meta charset="utf-8"><style>body{margin:0;font-family:var(--font-sans);color:var(--foreground)}svg{width:100%;height:300px}text{fill:var(--foreground);font-size:12px}p{font-size:12px;color:var(--muted-foreground)}</style></head><body><svg viewBox="0 0 720 300" role="img" aria-label="Width 8 projected minutes per prediction">'''+''.join(bars)+'''</svg><p>Width 8, projected minutes per prediction. Range endpoints combine workload and cost assumptions. Matrix + QK leaves strided weighted-value dots scalar. Fused-executor baseline only; stock-HF latency is unmeasured.</p></body></html>'''
(HERE/'estimates.html').write_text(html)
