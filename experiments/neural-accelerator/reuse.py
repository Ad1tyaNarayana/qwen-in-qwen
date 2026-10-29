"""Local residual allocation with a separate, exact clear before each handover."""
from dataclasses import dataclass

from chipformer.dsl.graph import Expr
from chipformer.q35.emit import reads
from chipformer.q35.ir import Chan, Dim, FFN, Head


@dataclass
class Lifetime:
    born: int
    last: int
    last_write: int


def expressions(value):
    if isinstance(value, Expr):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from expressions(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from expressions(item)


def reuse(c):
    """Allocate physical dimensions without changing any recurrent table or head shape.

    Observers, output codes, lag sources and tiny counter outputs remain pinned.
    A reused slot is cleared in an earlier MLP, never in the new writer's MLP.
    """
    original = c.dims
    lives = {d: Lifetime(0 if d in c.token_fns else dim.level, dim.level, dim.level)
             for d, dim in enumerate(original) if d}
    pinned = set(c.keep)
    for expr in expressions([c.probe, c.out_fam, c.out_pay]):
        pinned.update(expr.terms)
    lag_sources = set()
    tiny_outputs = set()
    for node in c.nodes:
        for d in reads(node):
            if d:
                lives[d].last = max(lives[d].last, node.level)
        if isinstance(node, Head):
            channels = [*node.q.values(), *node.k.values()]
            for vh in node.vheads():
                channels += list(vh.v.values())
                if node.tiny:
                    tiny_outputs.update(d for _, d, _ in vh.out)
            for channel in channels:
                if any(channel.taps[1:]):
                    lag_sources.update(channel.expr.terms)
    pinned |= lag_sources | tiny_outputs
    owners = []
    slots = {0: 0}
    units = dict(c.units)
    handovers = []
    for d in sorted(lives, key=lambda d: (lives[d].born, d)):
        born = lives[d].born
        clear = born-2 if born % 2 == 0 else born-1
        chosen = None
        if clear >= 2 and units.get(clear, 0) < c.target.intermediate:
            for slot, old in enumerate(owners):
                life = lives[old]
                if old not in pinned and life.last <= clear and life.last_write < clear:
                    chosen = slot
                    handovers.append(dict(old=old, new=d, slot=slot+1, last=life.last,
                                          last_write=life.last_write, clear=clear, born=born))
                    units[clear] = units.get(clear, 0)+1
                    owners[slot] = d
                    break
        if chosen is None:
            chosen = len(owners)
            owners.append(d)
        slots[d] = chosen+1
    assert all(h['last'] <= h['clear'] < h['born'] and h['old'] not in lag_sources
               for h in handovers)

    def expr(e):
        return Expr({slots[d]: weight for d, weight in e.terms.items()})

    def remap(value):
        if isinstance(value, Expr):
            return expr(value)
        if isinstance(value, list):
            return [remap(item) for item in value]
        if isinstance(value, tuple):
            return tuple(remap(item) for item in value)
        if isinstance(value, dict):
            return {key: remap(item) for key, item in value.items()}
        return value

    def chans(channels):
        return {key: Chan(expr(ch.expr), ch.taps) for key, ch in channels.items()}

    for node in c.nodes:
        if isinstance(node, FFN):
            node.terms = [(coef, expr(a), expr(b)) for coef, a, b in node.terms]
            node.out = slots[node.out]
            node.into = True
        else:
            node.q, node.k = chans(node.q), chans(node.k)
            for vh in [node, *node.vx]:
                vh.v = chans(vh.v)
                vh.a = expr(vh.a)
                vh.z = {col: expr(e) for col, e in vh.z.items()}
                vh.beta = None if vh.beta is None else expr(vh.beta)
                vh.out = [(col, slots[d], weight) for col, d, weight in vh.out]
                vh.into = True
    for h in handovers:
        physical = h['slot']
        c.nodes.append(FFN(f"reuse.clear.{h['old']}.{h['new']}",
                           [(-1, Expr({physical: 1}), Expr({0: 1}))],
                           physical, h['clear'], True))
    # All original MLP readers precede the clear in the IR. In stock execution they
    # read the same sublayer input, before its residual update clears the slot.
    c.nodes.sort(key=lambda node: node.level)
    c.units = units
    c.token_fns = {slots[d]: fn for d, fn in c.token_fns.items()}
    c.full_fns = {slots[d] for d in c.full_fns}
    for name in ('one', 'fam', 'pay', 'kk', 'final', 'code', 'owner', 'out_fam', 'out_pay', 'probe'):
        setattr(c, name, remap(getattr(c, name)))
    c.keep = [slots[d] for d in c.keep]
    c.dims = [original[0]] + [Dim(f'reuse.slot{slot+1}',
        'token' if slot+1 in c.token_fns else 'ffn', lives[owner].last_write)
        for slot, owner in enumerate(owners)]
    c.reuse_report = dict(logical_dims=len(original), physical_dims=len(c.dims),
        clears=len(handovers), pinned=len(pinned), pinned_lag_sources=len(lag_sources),
        pinned_tiny_outputs=len(tiny_outputs), handovers=handovers,
        lifetimes=[dict(dim=d, name=original[d].name, born=life.born,
                        last=life.last, last_write=life.last_write, pinned=d in pinned)
                   for d, life in lives.items()])
    return c
