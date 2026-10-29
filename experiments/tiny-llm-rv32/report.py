#!/usr/bin/env python3
"""Summarize measured records and explicitly hypothetical dot-product costs."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from build import ROOT, HERE


def memory(d, h, layers, heads, kvheads, vocab, context):
    kd = d * kvheads // heads
    parameters = vocab*d + layers*(2*d*d + 2*kd*d + 3*d*h) + (2*layers+1)*d
    matrix_rows = vocab + layers*(3*d+2*kd+2*h)
    scratch = 4*(5*d+2*h+vocab+context)+2*max(d,h)
    kv = 4*layers*context*kd
    return dict(dim=d, hidden=h, layers=layers, heads=heads, kvheads=kvheads, vocab=vocab,
        context=context, parameters=parameters, kv_bytes=kv, scratch_bytes=scratch,
        ram_budget_with_8k_stack_and_256_reserved=kv+scratch+8192+256,
        approximate_weight_bytes=parameters+3*(2*layers+1)*d+4*matrix_rows,
        rom_note='Weight estimate includes norm int32 and per-row scales; add RoPE, LUTs, vocabulary, code constants. Shared classifier assumed.')


def report(builddir, qualitydir, evidence):
    b = json.loads((builddir / 'build.json').read_text())
    r = json.loads((builddir / 'reference.json').read_text())
    q = json.loads((qualitydir / 'quality.json').read_text())
    assert r['bit_exact'] and r['complete'] and r['elf_sha256'] == b['elf']['sha256']
    assert b['engine_sha256'] == hashlib.sha256((HERE/'engine.c').read_bytes()).hexdigest()
    assert b['pack']['rom_sha256'] == q['cases'][0]['rom_sha256']
    assert r['native_output_sha256'] == q['sanitizer']['output_sha256']
    cfg = b['pack']
    d,h,l,heads,kvh,v = [cfg[k] for k in ('dim','hidden','layers','heads','kvheads','vocab')]
    kd = d*kvh//heads
    shapes = [(l*d,d), (2*l*kd,d), (l*d,d), (2*l*h,d), (l*d,h), (v,d)]
    macs = sum(rows*n for rows,n in shapes)
    assert len(r['positions']) == b['steps']
    positions = []
    for p in r['positions']:
        assert p['dot_body_instructions'] == 7*macs, 'Compiled loop work does not match model MAC count'
        c = Counter(p['opcodes'])
        mix = dict(MUL=sum(c[k] for k in ['MUL','MULH','MULHU','MULHSU']),
            loads=sum(c[k] for k in ['LB','LH','LW','LBU','LHU']),
            stores=sum(c[k] for k in ['SB','SH','SW']),
            branches=sum(c[k] for k in ['BEQ','BNE','BLT','BGE','BLTU','BGEU']),
            jumps=c['JAL']+c['JALR'], division=sum(c[k] for k in ['DIV','DIVU','REM','REMU']))
        mix['ALU_other'] = p['instructions']-sum(mix.values())
        dot = {}
        for width in (4,8,16):
            groups = sum(rows*((n+width-1)//width) for rows,n in shapes)
            # A memory-to-accumulator dot, two pointer updates, and one loop branch.
            instructions = p['instructions']-p['dot_body_instructions']+4*groups
            low = p['trace_tokens']-p['dot_body_tokens']+7*groups
            high = low+groups
            dot[width] = dict(groups=groups, instructions=instructions,
                trace_tokens_low=low, trace_tokens_high=high,
                minutes_low=low/4380/60, minutes_high=high/4380/60)
        positions.append(dict(**p, instruction_mix=mix,
            dot_instruction_share=p['dot_body_instructions']/p['instructions'],
            dot_trace_share=p['dot_body_tokens']/p['trace_tokens'],
            matvec_instruction_share=p['matvec_instructions']/p['instructions'],
            minutes_at_4380=p['trace_tokens']/4380/60, hypothetical_dot=dot))
    phases = {}
    for phase in ('prefill','decode'):
        ps = [p for p in positions if p['phase']==phase]
        phases[phase] = dict(positions=len(ps),
            instructions=sum(p['instructions'] for p in ps),
            trace_tokens=sum(p['trace_tokens'] for p in ps),
            mean_instructions=sum(p['instructions'] for p in ps)/len(ps),
            mean_trace_tokens=sum(p['trace_tokens'] for p in ps)/len(ps))
    source_hashes = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.iterdir()
                     if p.suffix in ('.py','.c','.S','.ld')}
    result = dict(scope='Phase 0 software feasibility only; not compiled circuit or stock Qwen evidence',
        bit_exact=r['bit_exact'], compared_positions=len(positions),
        quality=q, build=b, reference={k:v for k,v in r.items() if k!='positions'},
        phases=phases, positions=positions, macs_per_position=macs,
        source_sha256=source_hashes,
        dot_assumptions='Hypothetical unchanged non-dot work. Each group uses one memory-reading int8 x int16 accumulate instruction, two pointer increments, one branch; dot output 2 to 3 trace tokens, pointer results 2 each, branch 1. Includes tails with padding. No latency/return-path/ISA implementation is established. No speed estimate for stock HF.',
        whole_run_per_position=dict(instructions=r['instructions']/len(positions),
            trace_tokens=r['trace_tokens']/len(positions),
            minutes=r['trace_tokens']/len(positions)/4380/60),
        custom_models=[memory(128,344,8,8,k,1024,c) for k,c in [(4,16),(4,22),(4,23),(2,32),(2,45)]],
        limitations=['One prompt has complete RV32/native verification; quality has three native prompts.',
          'Fixed-length runs include prompt positions. Context is a hard stop, with no sliding window.',
          'No neural/GPU execution, accelerator, new ISA, custom trained model, or stock-layer return path was tested.',
          'Dot projections omit implementation overhead and assume the accelerator reads both memory operands without extra guest loads.',
          'Custom geometry is a memory projection, not a quality or arithmetic-range proof.'])
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / 'evidence.json').write_text(json.dumps(result, indent=2)+'\n')
    colors = ['var(--chart-1)', 'var(--chart-2)', 'var(--chart-3)', 'var(--chart-4)']
    series = [('Measured scalar', [p['minutes_at_4380'] for p in positions])]
    series += [(f'Hypothetical dot {w}, high estimate',
        [p['hypothetical_dot'][w]['minutes_high'] for p in positions]) for w in (4,8,16)]
    ymax = max(series[0][1])*1.1
    svg = ['<svg viewBox="0 0 680 240" role="img" aria-label="Estimated minutes per LLM prediction by position">']
    for i in range(5):
        y = 200-i*44
        svg += [f'<path d="M42 {y}H662" stroke="var(--border)"/>',
                f'<text x="34" y="{y+4}" text-anchor="end">{ymax*i/4:.1f}</text>']
    for i,(name,ys) in enumerate(series):
        points=' '.join(f'{42+j*620/(len(ys)-1):.1f},{200-y/ymax*176:.1f}' for j,y in enumerate(ys))
        dashed='' if i==0 else 'stroke-dasharray="5 3"'
        svg += [f'<polyline points="{points}" fill="none" stroke="{colors[i]}" stroke-width="2" {dashed}/>']
    svg += ['<text x="42" y="224">0</text>',f'<text x="650" y="224">{len(positions)-1}</text>',
            '<text x="350" y="237" text-anchor="middle">Sequence position, including prompt</text></svg>']
    legend=''.join(f'<span style="color:{colors[i]}">{name}</span>' for i,(name,_) in enumerate(series))
    html='<!doctype html><html><head><meta charset="utf-8"><style>body{margin:0;color:var(--foreground);font-family:var(--font-sans)}svg{width:100%;height:240px}text{fill:var(--muted-foreground);font-size:11px}p{font-size:12px;color:var(--muted-foreground)}.legend{display:flex;gap:12px;flex-wrap:wrap;font-size:12px}</style></head><body><div class="legend">'+legend+'</div>'+''.join(svg)+'<p>Minutes per LLM prediction at an assumed 4,380 neural trace tokens/s. Scalar counts are measured in the software reference. Dotted lines are hypothetical memory-dot projections with three result trace tokens per dot. Setup and neural prefill are excluded.</p></body></html>'
    (evidence/'performance.html').write_text(html)
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build',type=Path,default=ROOT/'.cache/tiny-llm/build')
    p.add_argument('--quality',type=Path,default=ROOT/'.cache/tiny-llm/quality')
    p.add_argument('--evidence',type=Path,default=HERE/'evidence')
    a=p.parse_args()
    r=report(a.build,a.quality,a.evidence)
    print(json.dumps(dict(phases=r['phases'], first=r['positions'][0],last=r['positions'][-1]),indent=2))
