"""M2 gate. Failures are retained and stop wider or stock-generation work."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import struct
import tempfile
import traceback
from prepare import ROOT, prepare


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--width', type=int, choices=[4, 8], default=4)
    args = p.parse_args()
    scratch = Path(tempfile.mkdtemp(prefix='ir-', dir=ROOT / '.cache/neural-accelerator'))
    manifest = prepare(scratch)
    import numpy as np
    import torch
    from chipformer.q35.ir import Head, evaluate
    from chipformer.q35.tiny import pick_layout
    from chipformer.rv import core, spec as S
    import reference
    from constructor import build, snapshot
    from oracle import execute
    from test_isa import encoding
    torch.set_num_threads(4)
    c = build(args.width)
    baseline = core.build()
    stats = c.stats()
    stats.update(logical_value_heads=sum(len(n.vheads()) for n in c.nodes if isinstance(n, Head)),
                 hidden=pick_layout(len(c.dims)).hidden)
    additions = {key: stats[key]-baseline.stats()[key]
                 for key in ('layers', 'dims', 'heads', 'ffn_units')}
    additions['logical_value_heads'] = stats['logical_value_heads']-sum(
        len(n.vheads()) for n in baseline.nodes if isinstance(n, Head))
    additions['hidden'] = stats['hidden']-pick_layout(len(baseline.dims)).hidden
    del baseline
    report = dict(status='running', width=args.width, stats=stats, additions=additions,
                  source_manifest=c.source_manifest, prepare_manifest=manifest,
                  checks=[], torch=torch.__version__, numpy=np.__version__,
                  transformers=__import__('transformers').__version__,
                  full_map=True, stock_generate=False)
    output = Path(__file__).parent / f'ir-w{args.width}.json'

    def save():
        output.write_text(json.dumps(report, indent=2)+'\n')

    def read(bits, matrix):
        return [sum(float(v)*matrix[0, d] for d, v in bit.terms.items()) for bit in bits]

    rng = random.Random(314159)
    ram = bytearray(rng.randbytes(65536))
    drom = rng.randbytes(4194304)
    cases = [('last-drom-bank', args.width, 4194304-args.width, 65536-2*args.width, 0x7fffffff),
             ('first-span', 4, 0, 0, 0xffffffff)]
    if args.width == 8:
        cases += [('tail-upper-stripes', 4, 4, 8, 0x80000000)]
    try:
        save()
        for name, width, wo, ao, acc in cases:
            regs = [0]*32
            regs[1:4] = [S.DROM_BASE+wo, S.RAM_BASE+ao, acc]
            regs[17] = 93
            word = encoding(width)
            machine = reference.Machine([word, 0x73]+[0]*(S.ROM_WORDS-2),
                                        list(struct.unpack('<16384I', ram)), regs.copy(), 0, drom=drom)
            machine.maximum = args.width
            expected = execute(word, regs, ram, drom, args.width)
            record = machine.record()
            machine = reference.Machine([word, 0x73]+[0]*(S.ROM_WORDS-2),
                                        list(struct.unpack('<16384I', ram)), regs.copy(), 0, drom=drom)
            snapshot(c, machine)
            report['active_case'] = name
            save()
            matrix = evaluate(c, [S.tok('GO')]+record, check=True, kernel=True)
            predictions = core.predictions(c, matrix)
            assert predictions == record+[S.tok('HALT')], (predictions, record+[S.tok('HALT')])
            weights = read(c.probe['ndot_w'], matrix)
            activations = read(c.probe['ndot_a'], matrix)
            want_w = [(drom[wo+i] >> b) & 1 for i in range(width) for b in range(8)]
            want_a = [(ram[ao+i//8] >> (i%8)) & 1 for i in range(16*width)]
            assert weights[:8*width] == want_w
            assert activations[:16*width] == want_a
            result = read(c.probe['ndot'], matrix)
            assert result == [(expected.regs[3] >> b) & 1 for b in range(32)]
            report['checks'].append(dict(case=name, tokens=len(matrix), operands_exact=True,
                                          result_bits_exact=True, records_exact=True, kernel=True))
            print(json.dumps(report['checks'][-1]), flush=True)
            save()
        from chipformer.rv.asm import assemble
        asm = ['_start:', f'li x1, {S.DROM_BASE}', f'li x2, {S.RAM_BASE}', 'li x3, 0x7fffffff']
        for row in range(32):
            for n in ((4,) if args.width == 4 else (4, 8)):
                asm += [f'.word {encoding(n)}', f'addi x1, x1, {n}', f'addi x2, x2, {2*n}']
        for offset in (0, 32, 64, 4064, 4096, 4128, 8160, 8192, 2097152, 4194272):
            asm += [f'li x1, {S.DROM_BASE+offset}', f'.word {encoding(args.width)}']
        for offset in (0, 8, 16, 48, 64, 112, 128, 16368, 16384, 32752, 32768, 65520):
            aligned = offset//(2*args.width)*(2*args.width)
            asm += [f'li x2, {S.RAM_BASE+aligned}', f'.word {encoding(args.width)}']
        asm += [f'li x1, {S.DROM_BASE}']
        for offset in range(0, 64, 4):
            asm += [f'li x2, {S.RAM_BASE+offset}', f'li x5, {rng.getrandbits(32)}',
                    'sw x5, 0(x2)', 'lw x6, 0(x2)', 'sb x5, 1(x2)', 'lw x6, 0(x2)',
                    'sh x5, 2(x2)', 'lw x6, 0(x2)', f'li x2, {S.RAM_BASE+(offset//(2*args.width))*(2*args.width)}',
                    f'.word {encoding(args.width)}']
        asm += ['li x17, 93', 'ecall']
        program = assemble('\n'.join(asm))
        program.rodata = drom
        machine = reference.Machine.from_program(program)
        machine.maximum = args.width
        machine.ram = list(struct.unpack('<16384I', ram))
        snapshot(c, machine)
        inputs, wanted_tokens, expectations = [], [], []
        current = S.tok('GO')
        while not machine.halted:
            word = machine.rom[(machine.pc-S.ROM_BASE)//4]
            dot_width = int(reference.decode(word).op[4:]) if reference.decode(word).op.startswith('NDOT') else 0
            old_regs = machine.x.copy()
            old_ram = struct.pack('<16384I', *machine.ram) if dot_width else None
            refresh = machine.n == S.N_MAX
            record = machine.record()
            for predicted in record:
                expectation = None
                if dot_width and not refresh:
                    outcome = execute(word, old_regs, old_ram, drom, args.width)
                    assert outcome.fault is None
                    expectation = dict(reg=outcome.regs[(word >> 7) & 31], width=dot_width,
                        w=drom[old_regs[1]-S.DROM_BASE:old_regs[1]-S.DROM_BASE+dot_width],
                        a=old_ram[old_regs[2]-S.RAM_BASE:old_regs[2]-S.RAM_BASE+2*dot_width])
                inputs.append(current)
                wanted_tokens.append(predicted)
                expectations.append(expectation)
                current = predicted
        report['active_case'] = 'dependent-dot-and-partial-store-sequence'
        save()
        carry = {}
        checked_dots = 0
        for start in range(0, len(inputs), 64):
            matrix = evaluate(c, inputs[start:start+64], check=True, kernel=True, carry=carry)
            predicted = core.predictions(c, matrix)
            assert predicted == wanted_tokens[start:start+64], (start, predicted, wanted_tokens[start:start+64])
            for position, expectation in enumerate(expectations[start:start+64]):
                if expectation is None:
                    continue
                row = matrix[position:position+1]
                width = expectation['width']
                assert read(c.probe['ndot_w'][:8*width], row) == [
                    (v >> bit) & 1 for v in expectation['w'] for bit in range(8)]
                assert read(c.probe['ndot_a'][:16*width], row) == [
                    (v >> bit) & 1 for v in expectation['a'] for bit in range(8)]
                assert read(c.probe['ndot'], row) == [(expectation['reg'] >> bit) & 1 for bit in range(32)]
                checked_dots += 1
            print(json.dumps(dict(sequence_tokens_checked=min(start+64, len(inputs)), total=len(inputs))), flush=True)
        report['checks'].append(dict(case='dependent-dot-and-partial-store-sequence',
                                     instructions=machine.steps, tokens=len(inputs), dot_positions=checked_dots,
                                     records_exact=True, operands_exact=True, result_bits_exact=True, kernel=True))
        controls = []
        regs = [0]*32
        regs[1:4] = [S.DROM_BASE, S.RAM_BASE, 0x12345678]
        regs[17] = 93
        dot_word = encoding(args.width)

        def fresh(words=None):
            words = words or [dot_word, 0x73]
            return reference.Machine(words+[0]*(S.ROM_WORDS-len(words)),
                list(struct.unpack('<16384I', ram)), regs.copy(), 0, drom=drom)

        for name in ('memory-bit', 'accumulator-bit', 'pc-bit'):
            machine = fresh()
            if name == 'memory-bit':
                corrupted = bytearray(drom)
                corrupted[0] ^= 1
                machine.drom = bytes(corrupted)
            elif name == 'accumulator-bit':
                machine.x[3] ^= 1
            else:
                machine.anchor ^= 1
            snapshot(c, machine)
            row = evaluate(c, [S.tok('GO')], check=True, kernel=True)
            if name == 'memory-bit':
                actual = read(c.probe['ndot_w'], row)[0]
                wanted = drom[0] & 1
                boundary = 'DROM operand bit 0 before arithmetic'
            elif name == 'accumulator-bit':
                actual = read(c.probe['C'], row)[0]
                wanted = regs[3] & 1
                boundary = 'old destination register bit 0'
            else:
                actual = read(c.probe['w'], row)[0]
                wanted = 0
                boundary = 'PC word bit 0 before instruction decode'
            assert actual != wanted, (name, actual, wanted)
            controls.append(dict(control=name, expected_boundary=boundary, expected=wanted, observed=actual, detected=True))
        machine = fresh()
        machine.x[5] = machine.ram[0] ^ 1
        store = 0x23 | 2 << 12 | 2 << 15 | 5 << 20
        machine.rom[:3] = [store, dot_word, 0x73]
        snapshot(c, machine)
        head = c.tables['ram0']
        old_beta = head.beta
        from chipformer.dsl.graph import Expr
        head.beta = Expr()
        try:
            row = evaluate(c, [S.tok('GO'), S.tok('ST0')], check=True, kernel=True)[1:2]
        finally:
            head.beta = old_beta
        actual = read(c.probe['ndot_a'], row)[0]
        wanted = machine.x[5] & 1
        assert actual != wanted
        controls.append(dict(control='write-gate-bit', expected_boundary='RAM operand bit 0 immediately after SW',
                             expected=wanted, observed=actual, detected=True))
        machine = fresh()
        snapshot(c, machine)
        row = evaluate(c, [S.tok('GO')], check=True, kernel=True)
        wanted = core.predictions(c, row)[0]
        dim = next(iter(c.code[0].terms))
        row[0, dim] *= -1
        actual = core.predictions(c, row)[0]
        assert actual != wanted
        controls.append(dict(control='token-family-bit', expected_boundary='first generated result token',
                             expected=wanted, observed=actual, detected=True))
        report['corruption_controls'] = controls
        report['status'] = 'partial'
        report['remaining'] = 'Full memory enumeration, store dependencies, corruption controls, and stock checks are still required.'
    except Exception as error:
        report['status'] = 'fail'
        report['failure_type'] = type(error).__name__
        report['failure'] = str(error)
        report['traceback'] = traceback.format_exc()
        report['stop'] = 'M2 gate failed. Do not run W=8 or M3.'
        save()
        raise
    save()


if __name__ == '__main__':
    main()
