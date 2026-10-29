"""Unit-test cache extraction on synthetic snapshots, without claiming stock execution."""
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

from prepare import ROOT


def main():
    import torch
    prep_spec = importlib.util.spec_from_file_location('observer_memory_prepare', ROOT / 'experiments/memory-map/prepare.py')
    prep = importlib.util.module_from_spec(prep_spec)
    prep_spec.loader.exec_module(prep)
    scratch = Path(tempfile.mkdtemp(prefix='observer-', dir=ROOT / '.cache/neural-accelerator'))
    prep.prepare(ROOT / '.upstream/qwenputer', scratch, 16, 8)
    sys.path.insert(0, str(scratch))
    from chipformer.rv import hf
    from chipformer.q35.ir import A_BIT
    from constructor import build, snapshot
    import reference
    from stock_m3 import workload, state_observer
    circuit = build(4)
    program, _ = workload(0, 7)
    machine = reference.Machine.from_program(program)
    machine.x[8] = 0xdeadbeef
    snapshot(circuit, machine)
    cache = SimpleNamespace(layers=[SimpleNamespace(recurrent_states={0: torch.zeros(1, 32, 128, 128)})
                                  for _ in range(circuit.n_layers())])
    hf.inject(circuit, cache)
    check = state_observer(circuit)
    expected = (tuple(machine.x), tuple(machine.ram), machine.anchor, machine.n, machine.pc)
    check(cache, expected, 0)
    slots = hf.head_slots(circuit)
    locations = {id(circuit.nodes[i]): slot for i, slot in slots.items()}
    for name, register, column in (('regs.rd', 8, circuit.reg_cols[0][0]), ('pc', 0, 2), ('ram0', 8, 2)):
        layer, head = locations[id(circuit.tables[name])]
        word = cache.layers[layer].recurrent_states[0][0, head * 2, register, column]
        original = word.clone()
        word.add_(A_BIT if word.item() == 0 else -A_BIT)
        try:
            check(cache, expected, 0)
        except AssertionError:
            print(f'detected synthetic {name} corruption')
        else:
            raise AssertionError(f'missed synthetic {name} corruption')
        finally:
            word.copy_(original)
    check(cache, expected, 0)
    print('synthetic cache observer checks pass; no model execution claim')


if __name__ == '__main__':
    main()
