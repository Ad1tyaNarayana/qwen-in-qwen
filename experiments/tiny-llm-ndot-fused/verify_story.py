"""Independently compare exported GPU trace, observed text and final state on Mac."""
import argparse
import hashlib
import json
import sys
import re
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--actual',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    meta=json.loads((a.inputs/'reference.json').read_text())
    report=json.loads((a.actual/'main/report.json').read_text())
    assert json.loads((a.actual/'sequence.json').read_text())['complete']
    for name in ('prefix','restore','full'):
        assert json.loads((a.actual/f'{name}.exit.json').read_text())['exit_code']==0
    assert report['complete'] and report['mismatches']==0
    actual=np.memmap(a.actual/'main/actual.tokens.u32',dtype='<u4',mode='r')
    expected=np.memmap(a.inputs/'reference.tokens.u32',dtype='<u4',mode='r')
    assert len(actual)==len(expected)==meta['tokens']==report['generated_tokens']
    mismatches=0
    for offset in range(0,len(actual),1048576):
        mismatches+=int(np.count_nonzero(actual[offset:offset+1048576]!=expected[offset:offset+1048576]))
    assert mismatches==0
    assert sha(a.actual/'main/actual.tokens.u32')==meta['sha256']==report['actual_trace_sha256']
    assert (a.actual/'main/actual-ram.bin').read_bytes()==(a.inputs/'reference-ram.bin').read_bytes()
    regs=json.loads((a.actual/'main/actual-registers.json').read_text())
    assert set(regs)=={'regs.rs1','regs.rs2','regs.rd'}
    assert all(x==meta['final_registers'] for x in regs.values())
    observed=(a.actual/'main/actual-output.bin').read_bytes()
    sys.path.insert(0,str(a.inputs/'reference-package'))
    from chipformer.rv import spec
    byte_base,_=spec.FAMILIES['BYTE']
    extracted=bytearray()
    for offset in range(0,len(actual),1048576):
        chunk=actual[offset:offset+1048576]
        selected=chunk[(chunk>=byte_base)&(chunk<byte_base+256)]
        extracted.extend((selected-byte_base).astype('u1').tobytes())
    assert bytes(extracted)==observed
    pieces=re.findall(rb'@[0-9]+ [0-9]+ [0-9]+ [0-9]+ [0-9]+\n>(.*?)\n(?=@|$)',observed,re.S)
    assert len(pieces)==meta['steps']
    story_text=b''.join(pieces).decode()
    assert observed==(a.inputs/'reference-output.bin').read_bytes()
    assert sha(a.actual/'main/actual-output.bin')==meta['output_sha256']
    chunks=[json.loads(x) for x in (a.actual/'main/chunks.jsonl').read_text().splitlines()]
    for chunk in chunks:
        end=chunk['end_offset'];begin=end-chunk['amount']
        assert hashlib.sha256(actual[begin:end].tobytes()).hexdigest()==chunk['sha256']
    prefix=json.loads((a.actual/'main/prefix-report.json').read_text())
    restored=json.loads((a.actual/'main/restore-report.json').read_text())
    assert prefix['generated_tokens']==65536
    assert restored['started_from_checkpoint_offset']==65536 and restored['generated_tokens']==65792
    assert report['started_from_checkpoint_offset']==65792
    checkpoint=json.loads((a.actual/'checkpoint-metadata.json').read_text())
    assert checkpoint['offset']==len(actual)
    assert checkpoint['compute_seconds']==report['compute_seconds']
    assert checkpoint['token']==[spec.tok('HALT')]
    root=Path(__file__).resolve().parents[2]
    for relative,digest in checkpoint['binding']['scripts'].items():
        assert sha(root/relative.replace('\\','/'))==digest,relative
    seconds=report['compute_seconds']
    summary=dict(status='pass',tokens=len(actual),mismatches=mismatches,positions=meta['steps'],
        prompt='Once',story_text=story_text,output=observed.decode(),output_source=report['output_source'],
        compute_seconds=seconds,tokens_per_second=len(actual)/seconds,min_per_position=seconds/60/meta['steps'],
        final_ram_bytes=65536,all_three_register_copies_equal=True,actual_checkpoint_restore_offsets=[65536,65792],
        trace_sha256=sha(a.actual/'main/actual.tokens.u32'),ram_sha256=sha(a.actual/'main/actual-ram.bin'),
        checkpoint_metadata=json.loads((a.actual/'checkpoint-metadata.json').read_text()),
        sources={str(path):sha(path) for path in (a.inputs/'reference.json',a.actual/'main/report.json')},
        checks='All actual trace tokens, literal final RAM bytes, 96 register values, guest output, chunk hashes and actual checkpoint restore boundaries',
        autonomous=True,reference_state_after_boot=False)
    a.out.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='checkpoint_metadata'},indent=2))


if __name__=='__main__':
    main()
