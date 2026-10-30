"""Generate a standalone chart from accepted timing evidence."""
import json
from pathlib import Path
D=Path(__file__).resolve().parent
b=json.loads((D/'evidence/bench.json').read_text())
parts=[]
for i,(window,values) in enumerate(b['windows'].items()):
    y=30+i*116
    label='Boot' if window=='boot' else 'Matvec, reference-seeded'
    parts.append(f'<text x="0" y="{y}" class="window">{label}</text>')
    for j,side in enumerate(('baseline','fused')):
        v=values[side];median=v['median_us_per_token'];lo,hi=v['range_us_per_token'];by=y+14+j*34
        x=110;scale=1.03
        name='Unfused' if side=='baseline' else 'Fused'
        parts.append(f'<text x="0" y="{by+17}" class="label">{name}</text><rect class="{side}" x="{x}" y="{by}" width="{median*scale:.3f}" height="24" rx="3"><title>{name}, {median:.3f} microseconds per token; range {lo:.3f} to {hi:.3f}</title></rect><path class="range" d="M{x+lo*scale:.3f},{by+12}H{x+hi*scale:.3f}"/><text x="{x+median*scale+8:.2f}" y="{by+17}">{median:.3f}</text>')
html='''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>
:root{font-family:var(--font-sans,system-ui);color:var(--foreground,currentColor)}html,body{margin:0}figure{margin:0}svg{width:100%;height:270px;display:block}text{fill:var(--foreground,currentColor);font-size:14px}.window{font-weight:600}.label{font-size:13px}.baseline{fill:var(--chart-2,currentColor);opacity:.55}.fused{fill:var(--chart-1,currentColor)}.range{stroke:var(--foreground,currentColor);stroke-width:2}figcaption{color:var(--muted-foreground,currentColor);font-size:13px;line-height:1.5}
</style></head><body><figure><svg viewBox="0 0 650 270" role="img" aria-label="Unfused and fused microseconds per token at boot and in the reference-seeded matvec window. Lower is faster.">'''+''.join(parts)+'''<text x="110" y="262" class="label">Microseconds per token. Lower is faster.</text></svg><figcaption>Five alternating repeats per backend and window. Each run checks 65,536 generated tokens. Zero mismatches. Bars show medians; marks show full ranges. About 1.56 times faster in both windows.</figcaption></figure></body></html>'''
(D/'timing.html').write_text(html)
