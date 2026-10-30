"""Count actual retired opcode families and first NDOT trace offset independently."""
from collections import Counter
import functools
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/neural-accelerator'))
from prepare import prepare
prepare(ROOT/'.cache/showcase'/f'scope-package-{time.time_ns()}')
from chipformer.rv import elf,ref
import reference
ref.decode=functools.lru_cache(None)(ref.decode)
reference.decode=functools.lru_cache(None)(reference.decode)
report={}
for demo in ['sha256','life','tictactoe','mnist','mnist-scalar']:
    out=ROOT/'.cache/showcase'/demo
    machine=reference.Machine.from_program(elf.program(out/'engine.elf'));machine.maximum=4
    counts=Counter();tokens=0;first=None;prefix_dots=0
    prefix=16384 if demo=='mnist' else 4096
    while not machine.halted:
        before=machine.steps;pc=machine.pc
        record=machine.record()
        if machine.steps>before:
            word=machine.rom[(pc-0x40000)//4]
            op=word&127;counts[f'0x{op:02x}']+=1
            if op==0x0b:
                if first is None:first=tokens
                if tokens+len(record)<=prefix:prefix_dots+=1
        tokens+=len(record)
    original=json.loads((out/'reference.json').read_text())
    assert tokens==original['tokens'] and machine.steps==original['instructions']
    report[demo]=dict(retired_opcodes=dict(counts),first_ndot_trace_offset=first,
        stock_prefix_tokens=prefix,ndot_instructions_completed_in_stock_prefix=prefix_dots,
        instructions=machine.steps,trace_tokens=tokens)
(ROOT/'experiments/showcase/evidence/workload-scope.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
