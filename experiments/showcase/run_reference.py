"""Export an independent scalar/ndot software trace, final state and native output."""
import argparse
import array
import functools
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2]


def run(out, budget=2000000):
    sys.path.insert(0,str(ROOT/'experiments/neural-accelerator'))
    from prepare import prepare
    prepare(out/f'reference-package-{time.time_ns()}')
    from chipformer.rv import elf, ref, spec
    # This script has a distinct import name so it cannot shadow the ndot Machine.
    import importlib.util
    module_spec=importlib.util.spec_from_file_location('ndot_reference',ROOT/'experiments/neural-accelerator/reference.py')
    ndot=importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(ndot)
    ref.decode=functools.lru_cache(None)(ref.decode)
    ndot.decode=functools.lru_cache(None)(ndot.decode)
    machine=ndot.Machine.from_program(elf.program(out/'engine.elf'))
    machine.maximum=4
    native=subprocess.check_output([str(out/'native')])
    count=0
    started=time.perf_counter()
    minimum_sp=spec.SP_INIT
    ndots=0
    with (out/'reference.tokens.u32').open('xb') as f:
        buf=array.array('I')
        while not machine.halted and machine.steps<budget:
            before=machine.steps
            pc=machine.pc
            record=machine.record()
            if machine.steps>before:
                index=(pc-spec.ROM_BASE)//4
                if 0<=index<len(machine.rom):
                    ndots += (machine.rom[index]&0x7f)==0x0b
            buf.extend(record);count+=len(record)
            minimum_sp=min(minimum_sp,machine.x[2])
            if len(buf)>=65536:
                if sys.byteorder!='little':buf.byteswap()
                f.write(buf.tobytes());del buf[:]
        if sys.byteorder!='little':buf.byteswap()
        f.write(buf.tobytes())
    assert machine.halted and machine.fault is None and machine.exit_code==0
    assert bytes(machine.out)==native
    ram=array.array('I',machine.ram)
    if sys.byteorder!='little':ram.byteswap()
    (out/'reference-ram.bin').write_bytes(ram.tobytes())
    (out/'reference-output.bin').write_bytes(native)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    report=dict(tokens=count,instructions=machine.steps,sha256=sha(out/'reference.tokens.u32'),
        elf_sha256=sha(out/'engine.elf'),final_registers=machine.x,
        final_ram_sha256=sha(out/'reference-ram.bin'),output_sha256=hashlib.sha256(native).hexdigest(),
        output=native.decode(),native_output_equal=True,halted=True,exit_code=0,fault=None,
        ndot_instructions=ndots,stack_bytes=spec.SP_INIT-minimum_sp,
        elapsed_seconds=time.perf_counter()-started,prompt_tokens=[],steps=1)
    (out/'reference.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('output','final_registers')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('build',type=Path)
    p.add_argument('--budget',type=int,default=2000000)
    a=p.parse_args();run(a.build.resolve(),a.budget)
