"""Independently regenerate and verify bounded stock token files on the CPU."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys

from prepare import ROOT
from stock_m3 import workload


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--actual', type=Path)
    p.add_argument('--legacy-inputs', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    prep_spec = importlib.util.spec_from_file_location('verify_m4_prepare', ROOT / 'experiments/memory-map/prepare.py')
    prep = importlib.util.module_from_spec(prep_spec)
    prep_spec.loader.exec_module(prep)
    prep.prepare(ROOT / '.upstream/qwenputer', a.out / 'scratch', 64, 4096)
    sys.path.insert(0, str(a.out / 'scratch'))
    from chipformer.rv import asm, elf, ref, spec
    import reference
    from oracle import execute, serialize
    _, source = workload(0, 7)
    program = asm.assemble(source.replace('li s3, 128', 'li s3, 2048'))
    ndot = reference.Machine.from_program(program)
    ndot.maximum = 4
    stories = ref.Machine.from_program(elf.program(a.legacy_inputs / 'stories/engine.elf'))
    results = []
    for name, machine in (('ndot', ndot), ('stories260K', stories)):
        tokens, commits, dots = [], 0, 0
        while len(tokens) < 8192:
            word = machine.rom[(machine.pc-spec.ROM_BASE)//4]
            independent = None
            if name == 'ndot' and machine.n != spec.N_MAX and reference.decode(word, 4).op == 'NDOT4':
                outcome = execute(word, machine.x, struct.pack('<'+'I'*len(machine.ram), *machine.ram), machine.drom, 4)
                independent = serialize(outcome, word, machine.pc)
                dots += 1
            record = machine.record()
            if independent is not None:
                assert record == independent and tuple(machine.x) == outcome.regs
            tokens.extend(record)
            if len(tokens) <= 8191 and record[-1] != spec.tok('HALT'):
                commits += 1
            assert not machine.halted
        raw = struct.pack('<'+'I'*8192, *tokens[:8192])
        (a.out / f'{name}.reference.tokens.u32').write_bytes(raw)
        result = dict(guest=name, tokens=8192, expected_observable_commits=commits,
                      independent_dot_checks=dots, token_sha256=hashlib.sha256(raw).hexdigest())
        if name == 'stories260K':
            inputs = a.legacy_inputs / 'stories'
            meta = json.loads((inputs / 'reference.json').read_text())
            historical = json.loads(Path(__file__).with_name('legacy-stories.json').read_text())
            assert hashlib.sha256((inputs / 'engine.elf').read_bytes()).hexdigest() == historical['elf_sha256']
            recorded = (inputs / 'reference.tokens.u32').read_bytes()
            assert hashlib.sha256(recorded).hexdigest() == meta['sha256'] == historical['original']['trace_sha256']
            assert raw == recorded[:8192*4]
            result['historical_original_reference_prefix_exact'] = True
        if a.actual:
            actual = (a.actual / f'{name}.tokens.u32').read_bytes()
            assert raw == actual, name
            result['actual_stock_tokens_exact'] = True
        results.append(result)
    (a.out / 'report.json').write_text(json.dumps(dict(status='pass', actual_checked=a.actual is not None, runs=results), indent=2)+'\n')


if __name__ == '__main__':
    main()
