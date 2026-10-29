"""Exact fp32 state and BF16 read contract for graph integer counters."""
import numpy as np


def eval_counter(node, m, chip, check, carry, ni, kernel):
    import torch
    from chipformer.q35 import ir

    assert len(node.vheads()) == 1 and set(node.q) == set(node.k) == {0}
    assert not node.a.terms
    q = ir.conv(node.q[0], m, check, carry, (ni, 'q', 0))
    k = ir.conv(node.k[0], m, check, carry, (ni, 'k', 0))
    v = np.zeros((len(m), ir.HD), dtype=np.float32)
    for col, ch in node.v.items():
        v[:, col] = ir.conv(ch, m, check, carry, (ni, 'v', 0, col))
    gates = ir.ev(node.beta, m)
    ir.require(np.isin(gates, [0, 195/512, 1]), 'integer counter beta selector', gates)
    ir.require(np.isin(q, [0, 128]) & np.isin(k, [0, 128]), 'integer counter q/k', q)
    initial = (node.init or [{}])[0]
    state = (carry[ni, 'S', 0] if carry is not None and (ni, 'S', 0) in carry
             else {r: np.asarray(row, dtype=np.float32).copy() for r, row in initial.items()})
    row = state.setdefault(0, np.zeros(ir.HD, dtype=np.float32))
    out = np.zeros_like(v, dtype=np.float64)
    for t in range(len(m)):
        old = row.copy()
        beta = np.float32(1 if gates[t] == 1 else 2.**-44 if gates[t] else 0)
        if k[t]:
            row[:] = row + (v[t] - row) * beta
        count = row[2] / 181
        ir.require(np.array([count == int(count) and 0 <= count <= 127]),
                   'integer counter state range', row[2:3])
        wanted = (0 if gates[t] == 1 else old[2] + (181 if gates[t] else 0)) if k[t] else old[2]
        ir.require(row[2:3] == wanted, 'integer counter fp32 update', row[2:3])
        ir.require(row[:2] == 181 * 2**17, 'integer counter reference columns', row[:2])
        if q[t]:
            h = (torch.from_numpy(row.copy()) / (128**0.5)).to(torch.bfloat16).float()
            normed = (h * torch.rsqrt(h.square().mean() + ir.RMS_EPS)).to(torch.bfloat16)
            out[t] = normed.double().numpy()
            ir.require(out[t, 2:3] == count / 16384, 'integer counter BF16 read', out[t, 2:3])
    if carry is not None:
        carry[ni, 'S', 0] = state
    for col, dim, weight in node.out:
        val = out[:, col] * np.maximum(ir.ev(node.z[col], m), 0) * weight
        ir.require(ir.is_bf16(val), 'integer counter output', val)
        m[:, dim] = m[:, dim] + val if node.into else val


def install():
    from chipformer.q35 import ir
    if getattr(ir.eval_head, 'integer_counters', False):
        return
    original = ir.eval_head

    def evaluate(node, m, chip, check, carry=None, ni=0, kernel=False):
        if node.levels == 'stock-integer-counter':
            return eval_counter(node, m, chip, check, carry, ni, kernel)
        return original(node, m, chip, check, carry, ni, kernel)

    evaluate.integer_counters = True
    ir.eval_head = evaluate
