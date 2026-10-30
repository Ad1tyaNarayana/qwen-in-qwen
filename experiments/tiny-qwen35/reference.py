"""Run the independent NDOT reference from boot and retain trace and final state."""
import argparse
import array
import functools
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/neural-accelerator'))
from prepare import prepare


def run(out, limit):
    out = out.resolve()
    meta = json.loads((out/'build.json').read_text())
    assert hashlib.sha256((out/'engine.elf').read_bytes()).hexdigest() == meta['elf']['sha256']
    manifest = prepare(out/'reference-package')
    from chipformer.rv import elf, ref, spec
    import reference
    ref.decode = functools.lru_cache(None)(ref.decode)
    reference.decode = functools.lru_cache(None)(reference.decode)
    machine = reference.Machine.from_program(elf.program(out/'engine.elf'))
    machine.maximum = 4
    native = subprocess.check_output([meta['native'],str(out/'model.bin')])
    digest = hashlib.sha256()
    chunks, positions = [], []
    count = begin_i = begin_t = 0
    minimum_sp = spec.SP_INIT
    started = time.monotonic()
    buffer = array.array('I')
    assert buffer.itemsize == 4
    ndots = 0
    with (out/'reference.tokens.u32').open('xb') as stream:
        def flush():
            if sys.byteorder != 'little':
                buffer.byteswap()
            raw = buffer.tobytes()
            stream.write(raw)
            digest.update(raw)
            chunks.append(dict(end_token=count,sha256=hashlib.sha256(raw).hexdigest()))
            del buffer[:]
        while not machine.halted and machine.steps < limit:
            if machine.n != spec.N_MAX:
                pc = machine.pc
                if pc == meta['elf']['symbols']['position_begin']:
                    begin_i,begin_t = machine.steps,count
                if pc == meta['elf']['symbols']['position_end']:
                    positions.append(dict(position=machine.x[10],instructions=machine.steps-begin_i,
                                          trace_tokens=count-begin_t,end_token=count,
                                          projected_minutes_at_3735=(count-begin_t)/3735/60))
                    (out/'reference-progress.json').write_text(json.dumps(positions,indent=2)+'\n')
                    print(json.dumps(positions[-1]),flush=True)
                word = machine.rom[(pc-spec.ROM_BASE)//4]
                if word & 127 == 0x0b and word >> 25 == 0x20:
                    ndots += 1
            record = machine.record()
            count += len(record)
            buffer.extend(record)
            minimum_sp = min(minimum_sp,machine.x[2])
            if len(buffer) >= 65536:
                flush()
        if buffer:
            flush()
    ram = struct.pack('<16384I',*machine.ram)
    (out/'reference-ram.bin').write_bytes(ram)
    (out/'reference-output.bin').write_bytes(machine.out)
    complete = machine.halted and machine.fault is None and machine.exit_code == 0
    result = dict(status='pass' if complete and bytes(machine.out) == native else 'fail',
                  kind='independent RV32IM+NDOT4 reference from ELF boot',
                  native_output_exact=bytes(machine.out) == native,complete=complete,
                  fault=machine.fault,exit_code=machine.exit_code,
                  tokens=count,instructions=machine.steps,ndot4_instructions=ndots,
                  sha256=digest.hexdigest(),chunks=chunks,positions=positions,
                  stack_bytes=spec.SP_INIT-minimum_sp,minimum_sp=minimum_sp,
                  stack_clearance=minimum_sp-meta['elf']['symbols']['__scratch_end'],
                  elf_sha256=meta['elf']['sha256'],final_ram_sha256=hashlib.sha256(ram).hexdigest(),
                  registers=machine.x,final_registers=machine.x,steps=meta['steps'],
                  prompt_tokens=meta['prompt_tokens'],output=bytes(machine.out).decode(errors='replace'),
                  output_sha256=hashlib.sha256(machine.out).hexdigest(),
                  elapsed_seconds=time.monotonic()-started,prepare_manifest=manifest)
    (out/'reference.json').write_text(json.dumps(result,indent=2)+'\n')
    assert result['status'] == 'pass',result
    assert len(positions) == meta['steps']
    assert result['stack_bytes'] <= 1024 and result['stack_clearance'] >= 1024
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build',type=Path,required=True)
    p.add_argument('--max-instructions',type=int,default=2000000000)
    a = p.parse_args()
    run(a.build,a.max_instructions)
