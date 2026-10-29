"""M2 gate. Failures are retained and stop wider or stock-generation work."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import struct
import tempfile
import traceback
from prepare import ROOT, prepare


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--width', type=int, choices=[4, 8], default=4)
    p.add_argument('--resume-controls', type=Path)
    p.add_argument('--chunk-size', type=int, default=2048)
    p.add_argument('--unreduced', action='store_true')
    p.add_argument('--out', type=Path, default=Path(__file__).parent)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
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
    c = build(args.width, reduced=not args.unreduced)
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
    output_name = f'ir-w{args.width}-before-reduction.json' if args.unreduced else f'ir-w{args.width}.json'
    output = args.out / output_name
    completed_faults = set()
    if args.resume_controls:
        prior = json.loads(args.resume_controls.read_text())
        assert prior['width'] == args.width and prior['source_manifest'] == c.source_manifest
        assert len(prior['corruption_controls']) == 5 and all(x['detected'] for x in prior['corruption_controls'])
        report['corruption_controls'] = prior['corruption_controls']
        report['checks'] = [x for x in prior['checks'] if 'fault' in x]
        completed_faults = {x['case'] for x in report['checks']}
        report['resumed_controls'] = dict(path=str(args.resume_controls),
            sha256=hashlib.sha256(args.resume_controls.read_bytes()).hexdigest())
    report['test_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report['chunk_size'] = args.chunk_size
    if hasattr(c, 'reuse_report'):
        liveness = args.out / f'liveness-w{args.width}.json'
        liveness.write_text(json.dumps(c.reuse_report, indent=2)+'\n')
        report['reuse'] = {key: value for key, value in c.reuse_report.items()
                           if key not in ('lifetimes', 'handovers')}
        report['reuse']['evidence'] = liveness.name

    def save():
        payload = json.dumps(report, indent=2)+'\n'
        with output.open('w') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())

    def control_result(**result):
        result['detected'] = bool(result['detected'])
        for key in ('expected', 'observed'):
            result[key] = int(result[key])
        report.setdefault('corruption_controls', []).append(result)
        save()
        print(json.dumps(result), flush=True)
        assert result['detected'], result

    node_indices = {id(node): i for i, node in enumerate(c.nodes)}

    def neural_ram(carry):
        from chipformer.q35.ir import A_BIT
        words = []
        for wi in range(S.RAM_WORDS):
            h = wi & 15 | (wi >> 12) << 4
            state = carry[node_indices[id(c.tables[f'ram{h}'])], 'S', wi >> 4 & 1]
            row = state.get(wi >> 5 & 127, np.zeros(128))
            bits = (row[2:34]+row[34:66])/A_BIT
            assert np.isin(bits, [0, 1]).all(), (wi, bits)
            words.append(sum(int(bit) << b for b, bit in enumerate(bits)))
        return struct.pack('<16384I', *words)

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
        regs = [0]*32
        regs[1:4] = [S.DROM_BASE, S.RAM_BASE, 0x12345678]
        regs[17] = 93
        dot_word = encoding(args.width)

        def fresh(words=None):
            words = words or [dot_word, 0x73]
            return reference.Machine(words+[0]*(S.ROM_WORDS-len(words)),
                list(struct.unpack('<16384I', ram)), regs.copy(), 0, drom=drom)

        if not args.resume_controls:
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
                control_result(control=name, expected_boundary=boundary, expected=wanted, observed=actual, detected=actual != wanted)
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
            control_result(control='write-gate-bit', expected_boundary='RAM operand bit 0 immediately after SW',
                           expected=wanted, observed=actual, detected=actual != wanted)
            machine = fresh()
            snapshot(c, machine)
            row = evaluate(c, [S.tok('GO')], check=True, kernel=True)
            wanted = core.predictions(c, row)[0]
            family_bit = c.out_fam[0]
            assert len(family_bit.terms) == 1
            dim, coefficient = next(iter(family_bit.terms.items()))
            old = float(coefficient)*row[0, dim]
            assert old in (0, 1)
            row[0, dim] = (1-old)/float(coefficient)
            assert read([family_bit], row) == [1-old]
            actual = core.predictions(c, row)[0]
            control_result(control='token-family-bit', expected_boundary='first generated result token',
                           expected=wanted, observed=actual, detected=actual != wanted)
        faults = []
        for a in range(args.width):
            for b in range(2*args.width):
                if a or b:
                    faults.append((f'alignment-{a}-{b}', dot_word, S.DROM_BASE+a, S.RAM_BASE+b))
        for address in (0, S.ROM_BASE, S.RAM_BASE, S.DROM_BASE-args.width,
                        S.DROM_BASE+4194304, 0x20000, 0xfffffff8):
            faults.append((f'weight-region-{address:x}', dot_word, address, S.RAM_BASE))
        for address in (0, S.ROM_BASE, S.DROM_BASE, S.RAM_BASE-2*args.width,
                        S.RAM_BASE+65536, 0x10000, 0xfffffff0):
            faults.append((f'activation-region-{address:x}', dot_word, S.DROM_BASE, address))
        for f3 in range(2 if args.width == 8 else 1, 8):
            faults.append((f'illegal-width-{f3}', (dot_word & ~(7 << 12)) | f3 << 12,
                           S.DROM_BASE, S.RAM_BASE))
        for kind in range(3):
            for address in (S.ROM_BASE, S.DROM_BASE, S.DROM_BASE+4194304-4):
                store_word = 0x23 | kind << 12 | 1 << 15 | 5 << 20
                faults.append((f'readonly-{kind}-{address:x}', store_word, address, S.RAM_BASE))
        if args.width == 8:
            # The wider circuit must retain the narrower instruction's fault contract.
            for a in range(4):
                for b in range(8):
                    if a or b:
                        faults.append((f'ndot4-alignment-{a}-{b}', encoding(4), S.DROM_BASE+a, S.RAM_BASE+b))
            for address in (0, S.ROM_BASE, S.RAM_BASE, S.DROM_BASE-4,
                            S.DROM_BASE+4194304, 0x20000, 0xfffffffc):
                faults.append((f'ndot4-weight-region-{address:x}', encoding(4), address, S.RAM_BASE))
            for address in (0, S.ROM_BASE, S.DROM_BASE, S.RAM_BASE-8,
                            S.RAM_BASE+65536, 0x10000, 0xfffffff8):
                faults.append((f'ndot4-activation-region-{address:x}', encoding(4), S.DROM_BASE, address))
        for name, word, a, b in faults:
            if name in completed_faults:
                continue
            machine = fresh()
            machine.anchor = 5
            machine.rom[5] = word
            machine.x[1:3] = [a, b]
            expected = (__import__('oracle').Outcome(tuple(machine.x), 5, 0)
                        if word & 127 == 0x23 else execute(word, machine.x, ram, drom, args.width))
            assert expected.fault is not None
            record = __import__('oracle').serialize(expected, word, S.ROM_BASE+20)
            snapshot(c, machine)
            report['active_case'] = name
            save()
            fault_carry = {}
            rows = evaluate(c, [S.tok('GO')]+record[:-1], check=True, kernel=True, carry=fault_carry)
            predicted = core.predictions(c, rows)
            assert predicted == record, (name, predicted, record)
            assert read(c.probe['w'], rows[:1]) == [(5 >> bit) & 1 for bit in range(16)]
            assert neural_ram(fault_carry) == ram, name
            from chipformer.q35.ir import A_BIT
            for port in ('rs1', 'rs2', 'rd'):
                state = fault_carry[node_indices[id(c.tables[f'regs.{port}'])], 'S', 0]
                for reg in range(1, 32):
                    row = state.get(reg, np.zeros(128))
                    bits = [sum(row[col] for col in c.reg_cols[bit])/A_BIT for bit in range(32)]
                    assert np.isin(bits, [0, 1]).all(), (name, port, reg, bits)
                    value = sum(int(bit) << b for b, bit in enumerate(bits))
                    assert value == expected.regs[reg], (name, port, reg, value, expected.regs[reg])
            report['checks'].append(dict(case=name, fault=expected.fault, fault_pc=S.ROM_BASE+20,
                                         records_exact=True, no_partial_register_write=True,
                                         all_ram_bytes_unchanged=True, kernel=True))
            save()
            print(json.dumps(report['checks'][-1]), flush=True)
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
                if args.width == 8:
                    # Reset before NDOT8 instead of inheriting NDOT4's half-span advance.
                    asm += [f'li x1, {S.DROM_BASE+16*row+(4 if n == 4 else 0)}',
                            f'li x2, {S.RAM_BASE+32*row+(8 if n == 4 else 0)}']
                asm += [f'.word {encoding(n)}', f'addi x1, x1, {n}', f'addi x2, x2, {2*n}']
        for offset in (0, 32, 64, 4064, 4096, 4128, 8160, 8192, 2097152, 4194272):
            asm += [f'li x1, {S.DROM_BASE+offset}', f'.word {encoding(args.width)}']
        for offset in (0, 8, 16, 48, 64, 112, 128, 16368, 16384, 32752, 32768, 65520):
            aligned = offset//(2*args.width)*(2*args.width)
            asm += [f'li x2, {S.RAM_BASE+aligned}', f'.word {encoding(args.width)}']
        # Enumerate routing factors independently, plus both edges of each bank.
        for offset in range(0, 8192, args.width):
            asm += [f'li x1, {S.DROM_BASE+offset}', f'.word {encoding(args.width)}']
        for bank in range(512):
            for edge in (0, 8192-args.width):
                asm += [f'li x1, {S.DROM_BASE+8192*bank+edge}', f'.word {encoding(args.width)}']
        for offset in range(0, 65536, 2*args.width):
            asm += [f'li x2, {S.RAM_BASE+offset}', f'.word {encoding(args.width)}']
        if args.width == 8:
            # Enumerate every routing factor of the retained narrower operation too.
            asm += [f'li x2, {S.RAM_BASE}']
            for offset in range(0, 8192, 4):
                asm += [f'li x1, {S.DROM_BASE+offset}', f'.word {encoding(4)}']
            for bank in range(512):
                for edge in (0, 8192-4):
                    asm += [f'li x1, {S.DROM_BASE+8192*bank+edge}', f'.word {encoding(4)}']
            for offset in range(0, 65536, 8):
                asm += [f'li x2, {S.RAM_BASE+offset}', f'.word {encoding(4)}']
        asm += [f'li x1, {S.DROM_BASE}']
        for offset in range(0, 64, 4):
            asm += [f'li x2, {S.RAM_BASE+offset}', f'li x5, {rng.getrandbits(32)}',
                    'sw x5, 0(x2)', 'lw x6, 0(x2)', 'sb x5, 1(x2)', 'lw x6, 0(x2)',
                    'sh x5, 2(x2)', 'lw x6, 0(x2)', f'li x2, {S.RAM_BASE+(offset//(2*args.width))*(2*args.width)}',
                    f'.word {encoding(args.width)}']
        for kind, size in (('sb', 1), ('sh', 2), ('sw', 4)):
            for lane in range(0, 2*args.width, size):
                asm += [f'li x2, {S.RAM_BASE}', f'li x5, {rng.getrandbits(32)}',
                        f'{kind} x5, {lane}(x2)', f'.word {encoding(args.width)}']
        if args.width == 8:
            for kind, size in (('sb', 1), ('sh', 2), ('sw', 4)):
                for lane in range(0, 8, size):
                    asm += [f'li x2, {S.RAM_BASE}', f'li x5, {rng.getrandbits(32)}',
                            f'{kind} x5, {lane}(x2)', f'.word {encoding(4)}']
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
        for start in range(0, len(inputs), args.chunk_size):
            matrix = evaluate(c, inputs[start:start+args.chunk_size], check=True, kernel=True, carry=carry)
            predicted = core.predictions(c, matrix)
            assert predicted == wanted_tokens[start:start+args.chunk_size], (start, predicted, wanted_tokens[start:start+args.chunk_size])
            for position, expectation in enumerate(expectations[start:start+args.chunk_size]):
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
            print(json.dumps(dict(sequence_tokens_checked=min(start+args.chunk_size, len(inputs)), total=len(inputs))), flush=True)
        report['checks'].append(dict(case='dependent-dot-and-partial-store-sequence',
                                     instructions=machine.steps, tokens=len(inputs), dot_positions=checked_dots,
                                     records_exact=True, operands_exact=True, result_bits_exact=True, kernel=True))
        final_ram = neural_ram(carry)
        assert final_ram == struct.pack('<16384I', *machine.ram)
        report['checks'].append(dict(case='all-byte-final-neural-ram', bytes=len(final_ram),
                                     sha256=hashlib.sha256(final_ram).hexdigest(), exact=True))
        save()
        report['status'] = 'pass'
        report['remaining'] = 'Stock execution is M3 and has not been attempted.'
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
