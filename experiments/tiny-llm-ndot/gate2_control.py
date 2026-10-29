"""Account for all stores selected by the final NDOT shift-selector branch."""
import argparse
import functools
import hashlib
import json
from pathlib import Path
import struct
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/neural-accelerator'))
from prepare import prepare

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
args.out.mkdir(parents=True,exist_ok=False)
prepare(args.out/'scratch')
from chipformer.rv import elf,ref,spec
import reference
ref.decode=functools.lru_cache(None)(ref.decode)
reference.decode=functools.lru_cache(None)(reference.decode)
build=ROOT/'.cache/tiny-llm/ndot64'
meta=json.loads((build/'build.json').read_text())
liveness=json.loads((ROOT/'experiments/tiny-llm-ndot/evidence/gate2-liveness.json').read_text())
run=next(r for r in liveness['runs'] if r['guest']=='ndot')
root=next(r for r in run['temporaries'] if r['register']==28)
assert hashlib.sha256((build/'engine.elf').read_bytes()).hexdigest()==run['elf_sha256']
retained=ROOT/'.cache/tiny-llm/ndot-verify64'
old=(retained/'original-ram.bin').read_bytes()
new=(retained/'ndot-ram.bin').read_bytes()
m=reference.Machine.from_program(elf.program(build/'engine.elf'))
m.maximum=4
branches=[]
stores=[]
while not m.halted:
    pc=m.pc
    active=m.steps>=root['last_write']['step']
    refresh=m.n==spec.N_MAX
    if active:
        assert m.x[28]==root['last_write']['value']
    if active and not refresh and pc==0x401ac:
        ins=reference.decode(m.rom[(pc-spec.ROM_BASE)//4],4)
        assert ins.op=='BGE' and ins.rs1==28 and ins.rs2==0
        taken=ref.sx(m.x[28],32)>=0
        assert not taken
        branches.append(dict(step=m.steps+1,pc=hex(pc),value=m.x[28],taken=taken))
    if active and not refresh and pc==0x40120:
        ins=reference.decode(m.rom[(pc-spec.ROM_BASE)//4],4)
        assert ins.op=='SW' and ins.rs1==31 and ins.rs2==30
        address=(m.x[31]+ins.imm)&0xffffffff
        offset=address-spec.RAM_BASE
        assert 0<=offset<65536-8192
        value=struct.pack('<I',m.x[30])
        assert value==old[offset:offset+4]==new[offset:offset+4]
        stores.append(dict(step=m.steps+1,pc=hex(pc),address=hex(address),value=m.x[30]))
    m.record()
assert m.x==run['registers'] and m.x[2]==0x01010000
assert struct.pack('<16384I',*m.ram)==new
assert bytes(m.out)==(retained/'ndot-output.bin').read_bytes()
assert m.fault is None and m.exit_code==0
base=meta['elf']['symbols']['logits']
assert len(branches)==len(stores)==512
assert [int(s['address'],16) for s in stores]==list(range(base,base+512*4,4))
report=dict(status='pass',root=root['last_write'],final_sp=hex(m.x[2]),
    branch_site='0x401ac',store_site='0x40120',branches=branches,stores=stores,
    all_512_stored_values_equal_both_final_non_stack_images=True,
    retained_final_registers_ram_and_output_exact=True,
    first_address=hex(base),last_address=hex(base+511*4),
    scope='Final x28 is only used by the matvec rescaling branch. Every selected logit store is byte-exact non-stack RAM; downstream logit hash, argmax and output are covered by exact 64-position output comparison.')
(args.out/'gate2-control.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('branches','stores')},indent=2),flush=True)
