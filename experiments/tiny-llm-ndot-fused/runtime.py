"""Unified W=4 fixtures, with reference seeding limited to the labelled window."""
from pathlib import Path
import sys
import types
import copy
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/tiny-llm-ndot'))
import worker


def build(out, inputs):
    args = types.SimpleNamespace(repo=ROOT, upstream=ROOT.parent/'.upstream/qwenputer',
        out=out, inputs=inputs, original_circuit=False)
    import json
    fr, info, spec, meta = worker.build(args, json.loads((inputs/'reference.json').read_text()))
    meta['circuit_snapshot'] = 'unified constructor.snapshot'
    return fr, info, spec, meta


def fixtures(fr, inputs, tokens):
    from chipformer.rv import elf, spec
    import reference
    boot = reference.Machine.from_program(elf.program(inputs/'engine.elf'))
    boot.maximum = 4
    m = copy.deepcopy(boot)
    offset = 0
    # The retained program's first NDOT begins at token 18,616.
    # Select a later record boundary inside its repeated matvec loop.
    while offset < 32768:
        offset += len(m.record())
    seed = copy.deepcopy(m)
    expected = []
    while len(expected) < tokens:
        expected.extend(m.record())
    all_tokens = np.memmap(inputs/'reference.tokens.u32', dtype='<u4', mode='r')
    assert np.array_equal(np.asarray(expected[:tokens],dtype='<u4'),all_tokens[offset:offset+tokens])
    return [('boot',boot,all_tokens[:tokens]),
            ('ndot_matvec_reference_seeded',seed,np.asarray(expected[:tokens],dtype='<u4'))], offset


def reset(fr, machine, fusion=None):
    from constructor import snapshot
    from chipformer.rv import spec
    snapshot(fr.ex.c, machine)
    fr.load(0,spec.tok('GO'))
    fr.ex.reset()
    fr.ex.y.zero_()
    fr.tok.zero_()
    if fusion:
        for y in fusion['scratch_buffers']:
            y.zero_()
