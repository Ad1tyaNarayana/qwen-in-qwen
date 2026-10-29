"""Replay unchanged Doom and stories fixtures through old and extended references."""
import argparse
import array
import functools
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from prepare import ROOT, prepare


def main():
    p = argparse.ArgumentParser()
    p.add_argument('fixture', choices=['doom', 'stories'])
    args = p.parse_args()
    scratch = Path(tempfile.mkdtemp(prefix='legacy-', dir=ROOT / '.cache/neural-accelerator'))
    prepare(scratch)
    from chipformer.rv import elf, ref
    import reference
    if args.fixture == 'doom':
        path = ROOT / '.cache/dmc-continuing/legacy-build/doom.elf'
        expected = json.loads((ROOT / 'experiments/one-instr-design/counts-legacy.json').read_text())
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected['elf_sha256']
    else:
        path = ROOT / '.cache/tiny-llm/neural-stories/engine.elf'
        expected = json.loads(path.with_name('reference.json').read_text())
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected['elf_sha256']
    ref.decode = functools.lru_cache(None)(ref.decode)
    reference.decode = functools.lru_cache(None)(reference.decode)
    program = elf.program(path)
    results = []
    for cls in (ref.Machine, reference.Machine):
        machine = cls.from_program(program)
        digest, buffer, tokens = hashlib.sha256(), array.array('I'), 0
        while not machine.halted and machine.steps <= expected['instructions']:
            record = machine.record()
            buffer.extend(record)
            tokens += len(record)
            if len(buffer) >= 65536:
                if sys.byteorder != 'little':
                    buffer.byteswap()
                digest.update(buffer.tobytes())
                del buffer[:]
        if sys.byteorder != 'little':
            buffer.byteswap()
        digest.update(buffer.tobytes())
        ram = array.array('I', machine.ram)
        if sys.byteorder != 'little':
            ram.byteswap()
        result = dict(instructions=machine.steps, tokens=tokens,
                      trace_sha256=digest.hexdigest(), final_ram_sha256=hashlib.sha256(ram.tobytes()).hexdigest(),
                      output_sha256=hashlib.sha256(machine.out).hexdigest(),
                      pc=machine.pc, regs=machine.x, fault=machine.fault,
                      exit_code=machine.exit_code, halted=machine.halted)
        assert machine.halted and machine.fault is None and machine.exit_code == 0
        assert (machine.steps, tokens) == (expected['instructions'], expected['tokens'])
        results.append(result)
        print(json.dumps(dict(fixture=args.fixture, machine=cls.__module__, **result)), flush=True)
    assert results[0] == results[1]
    if args.fixture == 'stories':
        for field, historical in [('trace_sha256', 'sha256'), ('final_ram_sha256', 'final_ram_sha256'),
                                  ('output_sha256', 'output_sha256')]:
            assert results[1][field] == expected[historical]
    output = dict(status='pass', fixture=args.fixture, original=results[0], extended=results[1],
                  elf_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  comparison='complete replay through HALT, all tokens hashed, final architectural state exact')
    (Path(__file__).parent / f'legacy-{args.fixture}.json').write_text(json.dumps(output, indent=2)+'\n')


if __name__ == '__main__':
    main()
