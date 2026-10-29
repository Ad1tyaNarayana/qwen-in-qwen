"""Combine executed liveness, control-use and exact 64-position evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out',type=Path,default=HERE/'evidence/gate2-corrected.json')
args=parser.parse_args()
paths=[HERE/'evidence'/name for name in ('gate2-liveness.json','gate2-control.json','gate2-native-recheck.json')]
live,control,native=[json.loads(path.read_text()) for path in paths]
assert all(r['status']=='pass' for r in (live,control,native))
assert native['positions']==64 and len(native['records'])==64
assert all(row[4]==0 for row in native['records'])
assert live['differing_registers']==[28,29,30,31]
assert all(int(run['final_sp'],16)==0x01010000 for run in live['runs'])
assert all(d['below_both_final_sp'] for d in live['stack_differences'])
ndot=next(run for run in live['runs'] if run['guest']=='ndot')
assert control['root']==next(t for t in ndot['temporaries'] if t['register']==28)['last_write']
assert len(control['branches'])==len(control['stores'])==512
retained=ROOT/'.cache/tiny-llm/ndot-verify64'
old=(retained/'original-ram.bin').read_bytes()
new=(retained/'ndot-ram.bin').read_bytes()
assert old[:57344]==new[:57344]
for store in control['stores']:
    offset=int(store['address'],16)-0x01000000
    assert 0<=offset<57344
    assert struct.pack('<I',store['value'])==old[offset:offset+4]==new[offset:offset+4]
assert (retained/'original-output.bin').read_bytes()==(retained/'ndot-output.bin').read_bytes()
assert hashlib.sha256((retained/'ndot-output.bin').read_bytes()).hexdigest()==native['output_sha256']
for run,folder in zip(live['runs'],('ndot-original64','ndot64')):
    assert hashlib.sha256((ROOT/'.cache/tiny-llm'/folder/'engine.elf').read_bytes()).hexdigest()==run['elf_sha256']
    assert run['exit_abi']['registers']==[10,17] and run['exit_abi']['values']==[0,93]
    assert all(not t['ecalls'] for t in run['temporaries'])
result=dict(status='pass',gate=2,rule='Exact defined observations, non-stack RAM, and final-use liveness with covered data/address/control stores',
    final_sp='0x01010000',stack_differences=[d['address'] for d in live['stack_differences']],
    differing_temporaries=[28,29,30,31],exit_abi_registers=[10,17],
    direct_store_sites=['0x40110','0x40120'],direct_store_address='0x0100aa54',
    control_branch='0x401ac',control_stores=512,control_address_range=['0x0100a258','0x0100aa54'],
    observations_64_exact=True,non_stack_ram_exact=True,
    proof_sha256={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
    neural_rule='Own-binary trace, final RAM and registers must match exactly without a dead-state exception')
args.out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
