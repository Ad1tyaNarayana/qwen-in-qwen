"""Shared counter graph with exact tiny-beta fp32 accumulation and BF16 reads.

counter_ir and compact interpret this graph's counter level explicitly.
The ordinary stock emitter consumes its projections without a special case.
"""

COUNTER = '''    def counter(self, name, reset, step):
        from integer_counters import feature
        reset = feature(self, reset, name + ".reset")
        step = feature(self, step, name + ".step")
        one = self.one
        ((x,),) = thead(self, name, {0: one}, {0: one},
                        [{"data": {2: (181 * 2**37) * (one-reset)}, "ref": (181 * one)/128,
                          "beta": reset + (195 * step)/512}], tiny=2.0**-90)
        self.tables[name] = self.nodes[-1]
        self.nodes[-1].levels = "stock-integer-counter"
        return x
'''

THERMO = '''    def thermo(self, x, reset, n, name):
        one = self.one
        return [self.node([(1, one, x-m), (-1, one, x-m-1)], f"{name}>{m}")
                for m in range(n)]
'''


def rewrite(source, core):
    import inspect
    for method, replacement in ((core.RvCore.counter, COUNTER), (core.RvCore.thermo, THERMO)):
        original = inspect.getsource(method)
        assert source.count(original) == 1
        source = source.replace(original, replacement)
    return source


def feature(circuit, expr, name):
    terms = [(weight, circuit.token_fns[dim], dim in circuit.full_fns)
             for dim, weight in expr.terms.items() if dim]
    constant = expr.terms.get(0, 0)
    return circuit.token_feature('counter.' + name,
        lambda token: constant + sum(weight * fn(token if full else token & 0xffff)
                                     for weight, fn, full in terms), full=True)


def snapshot(circuit, machine):
    import numpy as np
    from chipformer.q35.ir import HD, REF_COLS
    for name, count in (('count', machine.n), ('j', 0), ('vcnt', 0)):
        row = np.zeros(HD)
        row[list(REF_COLS)] = 181 * 2**17
        row[2] = 181 * count
        circuit.tables[name].init = [{0: row}]
