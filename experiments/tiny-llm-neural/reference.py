"""Stream an independent software trace and verify native guest output."""
import argparse
import array
import functools
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build', type=Path, required=True)
    p.add_argument('--max-instructions', type=int, default=20000000)
    a = p.parse_args()
    out = a.build.resolve()
    meta = json.loads((out / 'build.json').read_text())
    module = importlib.util.spec_from_file_location('memory_prepare', ROOT / 'experiments/memory-map/prepare.py')
    prep = importlib.util.module_from_spec(module)
    module.loader.exec_module(prep)
    package = out / 'reference-package'
    manifest = prep.prepare(ROOT / '.upstream/qwenputer', package, 64, 4096)
    sys.path.insert(0, str(package))
    from chipformer.rv import elf, ref, spec
    assert spec.RAM_WORDS * 4 == 65536 and spec.DROM_WORDS * 4 == 4194304
    ref.decode = functools.lru_cache(None)(ref.decode)
    machine = ref.Machine.from_program(elf.program(out / 'engine.elf'))
    native = subprocess.check_output([str(out / 'native'), str(out / 'model.bin')])
    trace = out / 'reference.tokens.u32'
    if trace.exists():
        raise FileExistsError(trace)
    syms = meta['elf']['symbols']
    count = 0
    buffer = array.array('I')
    assert buffer.itemsize == 4
    digest = hashlib.sha256()
    chunks, positions = [], []
    start_i = start_t = 0
    min_sp = spec.SP_INIT
    started = time.perf_counter()
    with trace.open('xb') as f:
        def flush():
            if sys.byteorder != 'little':
                buffer.byteswap()
            raw = buffer.tobytes()
            f.write(raw)
            digest.update(raw)
            chunks.append(dict(end_token=count, sha256=hashlib.sha256(raw).hexdigest()))
            del buffer[:]
        while not machine.halted and machine.steps < a.max_instructions:
            if machine.n != spec.N_MAX:
                if machine.pc == syms['position_begin']:
                    start_i, start_t = machine.steps, count
                if machine.pc == syms['position_end']:
                    positions.append(dict(position=machine.x[10], instructions=machine.steps-start_i,
                                          trace_tokens=count-start_t, end_token=count))
                    print(json.dumps(positions[-1]), flush=True)
            record = machine.record()
            count += len(record)
            buffer.extend(record)
            min_sp = min(min_sp, machine.x[2])
            if len(buffer) >= 65536:
                flush()
        if buffer:
            flush()
    assert machine.halted and machine.fault is None and machine.exit_code == 0
    assert bytes(machine.out) == native
    (out / 'reference-output.bin').write_bytes(machine.out)
    ram = array.array('I', machine.ram)
    if sys.byteorder != 'little':
        ram.byteswap()
    (out / 'reference-ram.bin').write_bytes(ram.tobytes())
    report = dict(kind='independent Mac software reference, never a neural input after boot',
                  tokens=count, sha256=digest.hexdigest(), chunks=chunks,
                  elf_sha256=hashlib.sha256((out / 'engine.elf').read_bytes()).hexdigest(),
                  positions=positions, instructions=machine.steps, stack_bytes=spec.SP_INIT-min_sp,
                  minimum_sp=min_sp, native_output_equal=True, output=native.decode(),
                  output_sha256=hashlib.sha256(native).hexdigest(),
                  final_ram_sha256=hashlib.sha256(ram.tobytes()).hexdigest(),
                  elapsed_seconds=time.perf_counter()-started, prepare_manifest=manifest,
                  prompt_tokens=meta['prompt_tokens'], steps=meta['steps'])
    (out / 'reference.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('chunks','prepare_manifest')}, indent=2))


if __name__ == '__main__':
    main()
