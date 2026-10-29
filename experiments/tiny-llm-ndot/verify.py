"""Compare native and software-reference engines without changing their states."""
import argparse
import functools
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/neural-accelerator'))
from prepare import prepare


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original', type=Path, required=True)
    p.add_argument('--ndot', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', runs=[], independent_dot_checks=0)
    def save():
        (a.out / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
    save()
    try:
        prepare(a.out / 'scratch')
        from chipformer.rv import elf, ref, spec
        import reference
        from oracle import execute
        sys.path.insert(0, str(ROOT / 'experiments/tiny-llm-rv32'))
        from verify import records, text
        ref.decode = functools.lru_cache(None)(ref.decode)
        reference.decode = functools.lru_cache(None)(reference.decode)
        outputs = [subprocess.check_output([str(path.resolve() / 'native'), str(path.resolve() / 'model.bin')])
                   for path in (a.original, a.ndot)]
        assert outputs[0] == outputs[1], 'native output differs'
        assert len(records(outputs[0])) == 64
        assert (a.original / 'model.bin').read_bytes() == (a.ndot / 'model.bin').read_bytes(), 'quantized model differs'
        report.update(native_exact=True, native_records=records(outputs[0]), decoded_text=text(outputs[0]),
                      model_sha256=sha((a.ndot / 'model.bin').read_bytes()))
        machines = []
        for name, directory, machine_type in [('original', a.original, ref.Machine), ('ndot', a.ndot, reference.Machine)]:
            meta = json.loads((directory / 'build.json').read_text())
            m = machine_type.from_program(elf.program(directory / 'engine.elf'))
            if name == 'ndot':
                m.maximum = 4
            symbols = meta['elf']['symbols']
            tokens = 0
            start_i = start_t = 0
            positions = []
            run = dict(guest=name, status='running', positions=positions)
            report['runs'].append(run)
            started = time.perf_counter()
            while not m.halted and m.steps < 300000000:
                refresh = m.n == spec.N_MAX
                if not refresh:
                    if m.pc == symbols['position_begin']:
                        start_i, start_t = m.steps, tokens
                    if m.pc == symbols['position_end']:
                        positions.append(dict(position=m.x[10], instructions=m.steps-start_i, trace_tokens=tokens-start_t))
                        print(json.dumps(dict(guest=name, **positions[-1])), flush=True)
                        run.update(instructions=m.steps, tokens=tokens, elapsed_seconds=time.perf_counter()-started)
                        save()
                expected = None
                if name == 'ndot' and not refresh and report['independent_dot_checks'] < 1024:
                    word = m.rom[(m.pc-spec.ROM_BASE)//4]
                    if reference.decode(word, 4).op == 'NDOT4':
                        expected = execute(word, m.x, struct.pack('<16384I', *m.ram), m.drom, 4)
                        assert expected.fault is None
                record = m.record()
                tokens += len(record)
                if expected is not None:
                    assert tuple(m.x) == expected.regs
                    report['independent_dot_checks'] += 1
            assert m.halted and m.fault is None and m.exit_code == 0
            actual = bytes(m.out)
            (a.out / f'{name}-output.bin').write_bytes(actual)
            assert actual == outputs[0], (name, 'software/native output differs')
            assert len(positions) == 64
            ram = struct.pack('<16384I', *m.ram)
            (a.out / f'{name}-ram.bin').write_bytes(ram)
            run.update(status='pass', instructions=m.steps, tokens=tokens, native_output_exact=True,
                       output_sha256=sha(actual), ram_sha256=sha(ram), registers=m.x,
                       pc=m.pc, anchor=m.anchor, count=m.n, fault=m.fault, exit_code=m.exit_code,
                       elapsed_seconds=time.perf_counter()-started)
            machines.append(m)
            save()
        old, new = machines
        report['instruction_savings'] = report['runs'][0]['instructions'] - report['runs'][1]['instructions']
        report['trace_token_savings'] = report['runs'][0]['tokens'] - report['runs'][1]['tokens']
        report['final_architecture'] = dict(
            register_differences=[dict(register=i, original=x, ndot=y) for i,(x,y) in enumerate(zip(old.x,new.x)) if x != y],
            pc_equal=old.pc == new.pc, original_pc=old.pc, ndot_pc=new.pc,
            ram_words_different=sum(x != y for x,y in zip(old.ram,new.ram)),
            exit_equal=old.exit_code == new.exit_code, fault_equal=old.fault == new.fault)
        state = report['final_architecture']
        save()
        assert not state['register_differences'] and state['pc_equal'] and state['ram_words_different'] == 0, ('final architectural state differs', state)
        report['status'] = 'pass'
    except Exception as error:
        import traceback
        report.update(status='fail', error=repr(error), traceback=traceback.format_exc())
        print(report['traceback'], flush=True)
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
