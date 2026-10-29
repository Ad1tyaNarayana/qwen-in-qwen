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
    from cache_diagnosis import differences
    mirror = SimpleNamespace(layers=[SimpleNamespace(
        recurrent_states={0: layer.recurrent_states[0].clone()},
        conv_states={0: torch.zeros(1, 8192, 4)}) for layer in cache.layers])
    for layer in cache.layers:
        layer.conv_states = {0: torch.zeros(1, 8192, 4)}
    assert differences(circuit, cache, mirror) == []
    layer, head = locations[id(circuit.tables['j'])]
    coordinate = (0, head * 2, 0, 2)
    cache.layers[layer].recurrent_states[0][coordinate] = 64
    mirror.layers[layer].recurrent_states[0][coordinate] = torch.tensor(0x4280000c, dtype=torch.int32).view(torch.float32)
    delta, = differences(circuit, cache, mirror)
    assert delta['node'] == 'j' and delta['coordinate'] == list(coordinate)
    assert (delta['cached_bits'], delta['chunked_bits']) == ('0x42800000', '0x4280000c')
    assert not delta['signed_zero'] and delta['value_column'] in delta['output_columns']
    cache.layers[layer].recurrent_states[0][coordinate] = 0.0
    mirror.layers[layer].recurrent_states[0][coordinate] = -0.0
    delta, = differences(circuit, cache, mirror)
    assert delta['signed_zero'] and delta['chunked_bits'] == '0x80000000'
    print('cache diagnosis retains readable coordinates, exact bits, and signed zero')
    integer = build(4, stock_integer_counters=True)
    machine.n = 17
    snapshot(integer, machine)
    integer_cache = SimpleNamespace(layers=[SimpleNamespace(recurrent_states={0: torch.zeros(1, 32, 128, 128)})
                                          for _ in range(integer.n_layers())])
    hf.inject(integer, integer_cache)
    integer_check = state_observer(integer)
    expected = (tuple(machine.x), tuple(machine.ram), machine.anchor, machine.n, machine.pc)
    integer_check(integer_cache, expected, 0)
    integer_slots = {id(integer.nodes[i]): slot for i, slot in hf.head_slots(integer).items()}
    layer, head = integer_slots[id(integer.tables['count'])]
    integer_cache.layers[layer].recurrent_states[0][0, head*2, 0, 2].add_(1)
    try:
        integer_check(integer_cache, expected, 0)
    except AssertionError as error:
        assert error.args[0][0] == 'PC count'
    else:
        raise AssertionError('missed integer counter corruption')
    print('integer counter observer accepts exact n=17 and rejects a one-unit state error')
    print('synthetic cache observer checks pass; no model execution claim')


if __name__ == '__main__':
    main()
