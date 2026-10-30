"""Export actual unified IR semantics and stock-emitter projection statistics on CPU."""
import argparse
from collections import Counter
import math
from pathlib import Path
import re
import numpy as np
from common import HERE, ROOT, compiler, sha, write_json

ROLES = {
    'pc': ('PC / fetch', '#e9bc68'), 'decode': ('Instruction decode', '#eda18b'),
    'register': ('Register file', '#7ed6d1'), 'ram': ('RAM', '#73b8ec'),
    'rom': ('Code ROM', '#b3a3dc'), 'drom': ('Data ROM', '#88b792'),
    'ndot': ('NDOT accelerator', '#e493bd'), 'alu': ('ALU / MUL / DIV / digit file', '#c5cc82'),
    'output': ('Token feedback / output', '#f0d4ad'), 'reuse': ('Residual clear', '#727f96'),
    'unknown': ('Unclassified compiler symbol', '#9ca5b4'),
    'unused': ('Unused stock capacity', '#566373'),
    'normalization': ('RMS normalization ballast', '#a89e7b'),
}


def role(name):
    if name.startswith('reuse.'): return 'reuse'
    if name.startswith(('ndot.',)): return 'ndot'
    if name.startswith('ram'): return 'ram'
    if name.startswith('drom'): return 'drom'
    if name.startswith('rom'): return 'rom'
    if name.startswith('regs.'): return 'register'
    if name.startswith(('exit.', 'out.', 'code', 'is', 'fits15', 'fits30', 'a7==', 'F0.', 'F1.', 'F2.', 'V.', 'J.', 'R.', 'BYTE.', 'QD.', 'BD.')): return 'output'
    if name.startswith(('pc', 'count', 'vcnt', 'anchor', 'link', 'n.', 'n>', 'j', 'v.', 'v>', 'w.', 'w+1', 'nd.')): return 'pc'
    if name.startswith(('uop', 'rs1', 'rs2', 'rd', 'imm', 'raw', 'funct', 'op.', 'dec')): return 'decode'
    if name.startswith(('add', 'mul', 'div', 'rem', 'shift', 'sh', 'alu', 'logic', 'xor', 'or', 'and', 'slt', 'cmp', 'val', 'mem.', 'load', 'store', 'fault', 'br', 'A.', 'B.', 'C.', 'Q.', 'EA', 'sd.', 'bmac', 'digit', 'text', 'cap', 'in_', 'mis_', 'oor_', 'result', 'take', 'target', 'sgn', 'neg', 'sign', 'abs', 'quot', 'zba', 'bn', 'quo', 'merge.', 'btgt', 'nd==', 'eq')): return 'alu'
    return 'unknown'


def form(e):
    return [[int(d), float(w)] for d, w in e.terms.items() if w]


def stats(values, total):
    a = np.asarray(values, dtype=np.float64)
    a = a[a != 0]
    return [total, len(a), float(min(0, a.min())) if len(a) else 0,
            float(max(0, a.max())) if len(a) else 0, float(np.linalg.norm(a))]


def axis(name, vh):
    if name.startswith('ram') and name[3:].isdigit():
        bank = int(name[3:])
        return dict(rows='128 striped word rows', columns='0,1 ballast; 2..33 stored bits; 34..65 partial-store delta bits',
            addresses=[0x1000000 + 4 * ((bank & 15) | (vh << 4) | (row << 5) | ((bank >> 4) << 12)) for row in range(128)],
            encoding='(S[row,2+b]+S[row,34+b])/128 gives word bit b')
    if (name.startswith('drom') and name[4:].isdigit()) or (name.startswith('rom') and name[3:].isdigit()):
        is_data=name.startswith('drom')
        bank=int(name[4:] if is_data else name[3:])
        base=0x2000000 if is_data else 0x40000
        return dict(rows='Rows p and p+64 hold complementary 32-byte halves of one two-hot read',
            columns='0,1 ballast. Rows 0..63 use columns 2..33; rows 64..127 use columns 34..65. Other data columns are zero.',
            addresses=[base+bank*8192+vh*4096+(row%64)*64+(row//64)*32 for row in range(128)],
            encoding='Raw S contains unsigned bytes'+('. Code ROM stores predecoded instruction records, not original instruction bytes.' if not is_data else '.'),
            column_starts=[2+32*(row//64) for row in range(128)])
    if name.startswith('regs.'):
        return dict(rows='x0..x31 register indices; x0 is implicit zero',columns='Ballast plus replicated register bits; see reg_cols',encoding='sum(S[reg,reg_cols[b]])/128 gives bit b')
    return dict(rows='IR query/key row index', columns='IR value column; 0,1 are ballast for ordinary heads',encoding='Exact compiler symbol. More specific axis semantics unavailable.')


def export(out):
    import torch
    torch.set_num_threads(2)
    c = compiler()
    from chipformer.q35.ir import FFN, Head, K
    from chipformer.q35 import emit
    from chipformer.q35.tiny import pick_layout
    lay = pick_layout(len(c.dims))
    projection_cache = {}
    def projection(e, gain=1):
        key = (tuple(e.terms.items()), gain)
        if key not in projection_cache:
            row = torch.zeros(lay.hidden)
            emit.row(e, lay, row, gain)
            emit.check_bf16('explorer projection', row)
            projection_cache[key] = row[row != 0].tolist()
        return projection_cache[key]
    ledger = c.reuse_report
    slots, next_slot = {}, 0
    handed = {h['new']:h['slot'] for h in ledger['handovers']}
    for life in sorted(ledger['lifetimes'],key=lambda x:(x['born'],x['dim'])):
        d = life['dim']
        if d in handed: slots[d] = handed[d]
        else:
            next_slot += 1
            slots[d] = next_slot
    assert next_slot == len(c.dims)-1
    owners = [[] for _ in c.dims]
    for life in ledger['lifetimes']:
        owners[slots[life['dim']]].append(dict(life, role=role(life['name'])))
    layers = []
    for L in range(c.n_layers()):
        units, heads = [], []
        counts = Counter()
        for n in c.nodes:
            if isinstance(n, FFN) and n.level == 2*L+2:
                r = role(n.name)
                for term, (coef,a,b) in enumerate(n.terms):
                    weights = projection(a,1/K) + projection(b,K) + [float(coef)]
                    units.append([n.name, r, n.out, float(coef), form(a), form(b), stats(weights,lay.hidden*3)])
                    counts[r] += 1
            if isinstance(n, Head) and n.level == 2*L+1:
                j = len(heads)
                vhs = []
                for x,v in enumerate(n.vheads()):
                    values = []
                    for channels in (n.q,n.k,v.v):
                        for ch in channels.values():
                            values += projection(ch.expr)
                            values += [float(t) for t in ch.taps if t]
                    for e in [v.a,*v.z.values()]: values += projection(e)
                    from chipformer.dsl.graph import Expr
                    beta = Expr({0:128}) if v.beta is None else 256*v.beta-128*Expr({0:1})
                    values += projection(beta)
                    values += [float(w) for _,_,w in v.out]
                    vhs.append(dict(value_head=2*j+x, stats=stats(values,len(values)),axis=axis(n.name,x),
                        output=[list(o) for o in v.out], decay=form(v.a),beta=None if v.beta is None else form(v.beta)))
                heads.append(dict(name=n.name,role=role(n.name),key_head=j,level=n.level,levels=n.levels,
                    query_rows=sorted(n.q), key_rows=sorted(n.k),values=vhs))
                counts[role(n.name)] += len(vhs)
        exit_level = max(n.level for n in c.nodes if n.name.startswith('exit.'))
        cleanup = 0
        if 2*L+2 == exit_level:
            from chipformer.dsl.graph import Expr
            for d in range(1,len(c.dims)):
                if d in c.keep: continue
                a,b = Expr({d:1}), Expr({0:1})
                units.append(['stock.exit.clear','output',d,-1,form(a),form(b),stats(projection(a,1/K)+projection(b,K)+[-1],lay.hidden*3)])
                cleanup += 1
            for i in range(lay.m):
                units.append(['stock.ballast.clear','output',None,-1,[],[],stats(projection(Expr({0:1}),K)+[1/(K*lay.scale),-1],lay.hidden*3),dict(hidden_coordinate=lay.off+i,gate='128 * ONE',up_weight=1/(K*lay.scale),down_weight=-1)])
                cleanup += 1
            counts['output'] += cleanup
        detail = dict(layer=L,heads=heads,units=units,unit_fields=['name','role','output_dim','coefficient','a','gate_b','weight_stats'],
                      weight_stats_fields=['coordinate_count','nonzero_count','min','max','l2'],weight_provenance='stock emitter row() with exact BF16 check; head stats include nonzero q/k/v/conv/z/a/b/out coefficients, shared q/k counted per value head')
        write_json(out/'layers'/f'{L}.json.gz',detail)
        layers.append(dict(index=L,type='linear_attention',heads=[{k:h[k] for k in ('name','role','key_head','values')} for h in heads],
                           units=len(units),roles=dict(counts),inactive_ffn_units=c.target.intermediate-len(units),inactive_value_heads=c.target.value_heads-sum(len(h['values']) for h in heads),detail=f'data/layers/{L}.json.gz',cleanup_units=cleanup))
        print(f'Layer {L}: {len(heads)} key heads, {len(units)} units',flush=True)
    layers.append(dict(index=c.n_layers(),type='full_attention',heads=[],units=9216,roles={'output':9216},zero=True,
                       note='Appended all-zero stock full-attention layer. 8 query heads, 2 KV heads, head_dim 256. No CPU state. Its MLP is also zero.'))
    physical_by_hidden={lay.dim(d):d for d in range(1,len(c.dims))}
    hidden_dimensions=[dict(index=h,kind='residual' if h in physical_by_hidden else 'ballast' if h<lay.m else 'tau' if h==lay.tau else 'padding',physical_dim=physical_by_hidden.get(h)) for h in range(lay.hidden)]
    graph = dict(schema='qwenputer.graph.v1',provenance='compiled IR',source=c.source_manifest,geometry=c.stats(),
        hidden=lay.hidden,ballast=lay.m,hidden_dimensions=hidden_dimensions,roles=ROLES,layers=layers,dimensions=owners,ledger=ledger,
        reg_cols=c.reg_cols,physical_to_hidden=[lay.dim(d) for d in range(1,len(c.dims))],
        notes=['FFN unit = coefficient * a * relu(gate_b). Stock emitter uses silu(128*gate_b)*(a/128).',
               'Unknown role labels retain the exact IR symbol; semantic grouping is prefix-based, never inferred from activity.',
               'Unused stock capacity has zero weights. Dense tensor weights are not distributed.'],
        source_files={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'experiments/neural-accelerator/constructor.py',ROOT/'experiments/neural-accelerator/reuse.py',ROOT/'experiments/neural-accelerator/stock.py',ROOT/'.upstream/qwenputer/experimental/riscv/chipformer/q35/emit.py']})
    write_json(out/'graph.json.gz',graph)
    print(graph['geometry'])

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=HERE/'data')
    export(p.parse_args().out)
