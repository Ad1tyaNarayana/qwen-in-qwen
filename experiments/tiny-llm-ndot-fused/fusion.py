"""Batch independent ROM heads while retaining ordered residual/history updates."""
import torch
import triton
import triton.language as tl
from chipformer.rv.tritonexec import _head_kernel


@triton.jit
def _parallel_heads(x, hist, S, y, D: tl.constexpr, V: tl.constexpr,
                    qrp, qix, qw, krp, kix, kw, vrp, vix, vw, zrp, zix, zw,
                    arp, aix, aw, brp, bix, bw, bconst, odim, ow, khead, nrows, groups,
                    KQ: tl.constexpr, KK: tl.constexpr, KV: tl.constexpr,
                    KZ: tl.constexpr, QHOT: tl.constexpr):
    group = tl.load(groups + tl.program_id(0))
    _head_kernel(x, hist, S, y + group * D, D, V,
                 qrp, qix, qw, krp, kix, kw, vrp, vix, vw, zrp, zix, zw,
                 arp, aix, aw, brp, bix, bw, bconst, odim, ow, khead, nrows,
                 KQ, KK, KV, KZ, QHOT)


@triton.jit
def _ordered_apply(x, y, hist, D: tl.constexpr, GROUPS: tl.constexpr,
                   BLOCK: tl.constexpr):
    d = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = d < D
    value = tl.load(x + d, mask, other=0.0)
    for group in range(GROUPS):
        h = hist + group * 3 * D + d
        out = y + group * D + d
        h0 = tl.load(h, mask, other=0.0)
        h1 = tl.load(h + D, mask, other=0.0)
        added = tl.load(out, mask, other=0.0)
        # Loads and stores can use different lane layouts across warps, so
        # finish every read of this group's history and output before writing.
        tl.debug_barrier()
        tl.store(h + 2 * D, h1, mask)
        tl.store(h + D, h0, mask)
        tl.store(h, value, mask)
        tl.store(out, 0.0, mask)
        value = value + added
    tl.store(x + d, value, mask)


def install_rom(ex):
    if ex.B != 1:
        raise ValueError('ROM fusion requires batch one.')
    original, steps = ex.step, ex.steps
    runs, current, written = [], [], set()
    for i, st in enumerate(steps):
        eligible = st['kind'] == 'head' and all(h.name.startswith('drom') for h in st['heads'])
        reads, writes = set(), set()
        if eligible:
            eligible = not torch.count_nonzero(st['nrows']).item()
            eligible = eligible and not torch.count_nonzero(st['integer_counter']).item()
            for name in ('k', 'v', 'a', 'b'):
                eligible = eligible and st[name][0][-1].item() == 0
            for name in ('q', 'z'):
                ix = st[name][1][:int(st[name][0][-1].item())]
                eligible = eligible and bool(torch.all(ix < ex.D).item())
                reads.update(ix.cpu().tolist())
            writes = set(st['odim'][st['ow'] != 0].cpu().tolist())
        if not eligible or reads & written:
            if len(current) > 1:
                runs.append(current)
            current, written = [], set()
        if eligible:
            current.append(i)
            written.update(writes)
    if len(current) > 1:
        runs.append(current)
    if not runs:
        raise ValueError('No independent consecutive data-ROM sublayers.')
    batches = {}
    for indices in runs:
        selected = [steps[i] for i in indices]
        packed = {}
        for name in ('q', 'k', 'v', 'z', 'a', 'b'):
            pointers, indices_, weights, offset = [], [], [], 0
            for st in selected:
                rp, ix, w = st[name]
                count = int(rp[-1].item())
                pointers.append(rp[:-1] + offset)
                indices_.append(ix[:count])
                weights.append(w[:count])
                offset += count
            pointers.append(torch.tensor([offset], device=ex.dev, dtype=torch.int32))
            ix, w = torch.cat(indices_), torch.cat(weights)
            if not ix.numel():
                ix = torch.zeros(1, device=ex.dev, dtype=torch.int32)
                w = torch.zeros(1, device=ex.dev)
            packed[name] = [torch.cat(pointers), ix, w]
        for name in ('bconst', 'odim', 'ow', 'nrows'):
            packed[name] = torch.cat([st[name] for st in selected])
        head_offset, kh, groups = 0, [], []
        for g, st in enumerate(selected):
            kh.append(st['khead'] + head_offset)
            head_offset += len(st['heads'])
            groups.extend([g] * st['V'])
        packed['khead'] = torch.cat(kh)
        packed['groups'] = torch.tensor(groups, device=ex.dev, dtype=torch.int32)
        S = torch.cat([st['S'] for st in selected], dim=1)
        hist = torch.stack([st['hist'][0] for st in selected])
        y = torch.zeros(len(selected), ex.D, device=ex.dev)
        offset = 0
        for g, st in enumerate(selected):
            st['S'] = S[:, offset:offset + st['V']]
            st['hist'] = hist[g:g + 1]
            offset += st['V']
        batches[indices[0]] = dict(indices=indices, packed=packed, S=S, hist=hist, y=y,
            V=offset, widths={k:max(st[k] for st in selected) for k in ('KQ','KK','KV','KZ','QHOT')})
    def fused_step(x):
        begin = 0
        try:
            for first, b in batches.items():
                ex.steps = steps[begin:first]
                original(x)
                p = b['packed']
                _parallel_heads[(b['V'], 1)](x, b['hist'], b['S'], b['y'], ex.D, b['V'],
                    *p['q'], *p['k'], *p['v'], *p['z'], *p['a'], *p['b'],
                    p['bconst'], p['odim'], p['ow'], p['khead'], p['nrows'], p['groups'],
                    **b['widths'], enable_fp_fusion=False)
                _ordered_apply[(triton.cdiv(ex.D, 1024),)](x, b['y'], b['hist'], ex.D,
                    len(b['indices']), BLOCK=1024, enable_fp_fusion=False)
                begin = b['indices'][-1] + 1
            ex.steps = steps[begin:]
            return original(x)
        finally:
            ex.steps = steps
    ex.step = fused_step
    return dict(batches=batches, scratch_y=next(iter(batches.values()))['y'],
                scratch_buffers=[b['y'] for b in batches.values()])


import importlib.util
from pathlib import Path
_feedback_path = Path(__file__).resolve().parents[1] / 'dmc-feedback-fusion/fusion.py'
_spec = importlib.util.spec_from_file_location('accepted_feedback_fusion', _feedback_path)
_feedback = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_feedback)
install_feedback = _feedback.install_feedback


def install(fr):
    info = install_rom(fr.ex)
    install_feedback(fr)
    return info
