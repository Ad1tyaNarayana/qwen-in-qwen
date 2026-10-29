"""Unfused Executor2 with the graph's fp32/BF16 integer-counter contract."""
import hashlib
import importlib.util
from pathlib import Path
import sys


def executor_class(directory):
    from chipformer.rv import tritonexec
    source = Path(tritonexec.__file__).read_text()
    replacements = [
        ('khead, nrows,\n', 'khead, nrows, integer_counter,\n'),
        ('    if (kmax > 0.0) & (bv == 1.0):\n        tl.store(Sb + krow * 128 + ch, v)',
         '''    integer = tl.load(integer_counter + vh)
    if integer:
        beta = tl.where(bv == 1.0, 1.0, tl.where(bv == 195.0/512.0, 2.0**-44, 0.0))
        if kmax > 0.0:
            old = tl.load(Sb + krow * 128 + ch)
            delta = (v - old) * beta
            tl.store(Sb + krow * 128 + ch, old + delta)
    elif (kmax > 0.0) & (bv == 1.0):
        tl.store(Sb + krow * 128 + ch, v)'''),
        ('    wv = tl.load(ow + vh * 128 + ch)',
         '''    if integer:
        counter_read = (o * (128.0**-0.5)).to(tl.bfloat16).to(tl.float32)
        variance = tl.sum(counter_read * counter_read, axis=0) / 128.0
        out = (counter_read * tl.rsqrt(variance + 1.0e-6)).to(tl.bfloat16).to(tl.float32)
    wv = tl.load(ow + vh * 128 + ch)'''),
        ('"nrows": t(nrows), "S":',
         '"nrows": t(nrows), "integer_counter": t([heads[h].levels == "stock-integer-counter" for h, _ in vh]), "S":'),
        ('st["nrows"], KQ=', 'st["nrows"], st["integer_counter"], KQ='),
        ('QHOT=st["QHOT"])', 'QHOT=st["QHOT"], enable_fp_fusion=False)'),
    ]
    for before, after in replacements:
        assert source.count(before) == 1, before
        source = source.replace(before, after)
    path = directory / 'unified_tritonexec.py'
    path.write_text(source)
    name = 'unified_tritonexec_' + hashlib.sha256(source.encode()).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.Executor2


def freerun_class(directory):
    from chipformer.rv import fastexec
    Executor2 = executor_class(directory)
    source = Path(fastexec.__file__).read_text()
    old = 'nf, nb = len(C.OUT_NAMES), C.NPAY'
    assert source.count(old) == 1 and source.count('enumerate(C.OUT_NAMES)') == 1
    source = source.replace(old, 'output_names = list(S.FAMILIES) + S.CONTROL_NAMES\n        nf, nb = len(output_names), C.NPAY')
    source = source.replace('enumerate(C.OUT_NAMES)', 'enumerate(output_names)')
    source = source.replace('from chipformer.rv.tritonexec import Executor2', 'from compact import UnifiedExecutor2 as Executor2')
    global UnifiedExecutor2
    UnifiedExecutor2 = Executor2
    path = directory / 'unified_fastexec.py'
    path.write_text(source)
    spec = importlib.util.spec_from_file_location('unified_fastexec', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.FreeRun
