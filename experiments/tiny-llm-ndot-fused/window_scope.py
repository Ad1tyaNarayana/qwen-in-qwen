"""Describe independent instruction coverage of the reference-seeded matvec window."""
import functools
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/neural-accelerator'))
from prepare import prepare
inputs=ROOT/'.cache/tiny-llm/ndot5-20261030'
prepare(ROOT/'.cache/tiny-llm/fused-window-scope')
from chipformer.rv import elf,spec,ref
import reference
ref.decode=functools.lru_cache(None)(ref.decode)
reference.decode=functools.lru_cache(None)(reference.decode)
m=reference.Machine.from_program(elf.program(inputs/'engine.elf'));m.maximum=4
offset=0
while offset<32768:
    offset+=len(m.record())
begin=offset
pc=m.pc
ndot=0
instructions=m.steps
while offset<begin+65536:
    if m.n!=spec.N_MAX:
        ins=reference.decode(m.rom[(m.pc-spec.ROM_BASE)//4],4)
        ndot+=ins.op=='NDOT4'
    offset+=len(m.record())
report=dict(kind='Independent reference coverage, equivalence and performance fixture only; never autonomous',
            start_token=begin,end_token=offset,requested_tokens=65536,start_pc=hex(pc),end_pc=hex(m.pc),
            ndot4_instructions=ndot,retired_instructions=m.steps-instructions,
            elf_sha256=hashlib.sha256((inputs/'engine.elf').read_bytes()).hexdigest())
Path(__file__).with_name('evidence').joinpath('window-scope.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
