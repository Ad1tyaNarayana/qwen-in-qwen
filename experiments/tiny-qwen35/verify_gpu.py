"""Independently validate raw neural and stock artifacts, with exact comparisons."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import subprocess

import numpy as np

from pack import ROOT, HERE


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def trace(actual, expected, count):
    a = np.memmap(actual,dtype='<u4',mode='r')
    b = np.memmap(expected,dtype='<u4',mode='r')
    assert len(a) == count and len(b) >= count
    mismatches = sum(int(np.count_nonzero(a[i:min(i+65536,count)] != b[i:min(i+65536,count)]))
                     for i in range(0,count,65536))
    assert mismatches == 0,mismatches
    return a


def verify(program, actual, source_revision=None):
    revision = (subprocess.check_output(['git', 'rev-parse', '--verify', source_revision+'^{commit}'],
                cwd=ROOT, text=True).strip() if source_revision else None)

    def source_digest(relative):
        relative = relative.replace('\\', '/')
        if revision:
            return hashlib.sha256(subprocess.check_output(['git', 'show', revision+':'+relative], cwd=ROOT)).hexdigest()
        return digest(ROOT/relative)

    reference = json.loads((program/'reference.json').read_text())
    assert reference['status'] == 'pass'
    assert json.loads((actual/'sequence.json').read_text(encoding='utf-8-sig'))['complete']
    for name in ['prefix','restore','full','stock']:
        assert json.loads((actual/(name+'.exit.json')).read_text(encoding='utf-8-sig'))['exit_code'] == 0
    main = actual/'main'
    report = json.loads((main/'report.json').read_text())
    accepted_cpu = json.loads((ROOT/'experiments/tiny-llm-ndot/evidence/gpu.json').read_text())
    assert report['constructor']['generated_sha256'] == accepted_cpu['constructor']['generated_sha256']
    assert report['complete'] and report['halted'] and report['mismatches'] == 0
    tokens = trace(main/'actual.tokens.u32',program/'reference.tokens.u32',reference['tokens'])
    assert digest(main/'actual.tokens.u32') == reference['sha256'] == report['actual_trace_sha256']
    assert (main/'actual-ram.bin').read_bytes() == (program/'reference-ram.bin').read_bytes()
    registers = json.loads((main/'actual-registers.json').read_text())
    assert set(registers) == {'regs.rs1','regs.rs2','regs.rd'}
    assert all(values == reference['final_registers'] for values in registers.values())
    sys.path.insert(0,str(program/'reference-package'))
    from chipformer.rv import spec
    byte_base,_ = spec.FAMILIES['BYTE']
    observed = bytearray()
    for i in range(0,len(tokens),1048576):
        selected = tokens[i:i+1048576]
        selected = selected[(selected >= byte_base)&(selected < byte_base+256)]
        observed.extend((selected-byte_base).astype('u1').tobytes())
    assert bytes(observed) == (main/'actual-output.bin').read_bytes() == (program/'reference-output.bin').read_bytes()
    assert hashlib.sha256(observed).hexdigest() == reference['output_sha256']
    pieces = re.split(rb'^@\d+ \d+ \d+ \d+ \d+\n',observed,flags=re.M)[1:]
    assert len(pieces) == reference['steps']
    text = b''.join(p[1:-1] for p in pieces).decode(errors='replace')
    prefix = json.loads((main/'prefix-report.json').read_text())
    restored = json.loads((main/'restore-report.json').read_text())
    assert prefix['generated_tokens'] == restored['started_from_checkpoint_offset'] == 65536
    assert restored['generated_tokens'] == 65792
    recovery = None
    if report['started_from_checkpoint_offset'] != 65792:
        import torch
        recovery = json.loads((actual/'recovery-checkpoint.json').read_text())
        saved_path = actual/'pre-shutdown/checkpoint.pt'
        assert digest(saved_path) == recovery['checkpoint_sha256']
        saved = torch.load(saved_path,map_location='cpu',weights_only=True)
        assert saved['offset'] == recovery['offset'] == report['started_from_checkpoint_offset']
        assert saved['compute_seconds'] == recovery['compute_seconds'] < report['compute_seconds']
        assert saved['binding'] == recovery['binding'] == checkpoint_binding(report,reference)
        assert saved['tok'].tolist() == recovery['token'] == [int(tokens[saved['offset']-1])]
        before = actual/'pre-shutdown/actual.tokens.u32'
        assert digest(before) == recovery['actual_trace_sha256']
        before_tokens = trace(before,program/'reference.tokens.u32',recovery['actual_trace_tokens'])
        assert np.array_equal(before_tokens[:saved['offset']],tokens[:saved['offset']])
        assert recovery['replay_tokens'] == len(before_tokens)-saved['offset']
    checkpoint = json.loads((actual/'checkpoint-metadata.json').read_text())
    assert checkpoint['offset'] == reference['tokens']
    assert checkpoint['token'] == [spec.tok('HALT')]
    assert checkpoint['compute_seconds'] == report['compute_seconds']
    assert checkpoint['binding']['elf'] == reference['elf_sha256']
    assert checkpoint['binding']['reference'] == reference['sha256']
    previous = 0
    for chunk in map(json.loads,(main/'chunks.jsonl').read_text().splitlines()):
        end = chunk['end_offset']
        assert end-previous == chunk['amount']
        assert hashlib.sha256(tokens[previous:end].tobytes()).hexdigest() == chunk['sha256']
        previous = end
    assert previous == reference['tokens']
    for name,sha in report['scripts'].items():
        assert source_digest(name) == sha,name
    stock = json.loads((actual/'stock/report.json').read_text())
    assert stock['status'] == 'pass' and stock['generated_tokens'] >= 16384 and stock['mismatches'] == 0
    assert stock['allocator_cap_bytes'] <= 15*2**30
    assert stock['peak_allocated_bytes'] <= stock['allocator_cap_bytes']
    assert stock['free_vram_guard_bytes'] >= 512*2**20 and stock['minimum_free_bytes'] >= 512*2**20
    assert stock['stock_generate'] and stock['initial_snapshot_only'] and not stock['layer_offload_used']
    assert stock['constructor']['generated_sha256'] == report['constructor']['generated_sha256']
    assert stock['elf_sha256'] == reference['elf_sha256']
    assert stock['worker_sha256'] == digest(ROOT/'experiments/tiny-llm-ndot/stock_prefix.py')
    trace(actual/'stock/actual.tokens.u32',program/'reference.tokens.u32',stock['generated_tokens'])
    result = dict(status='pass',neural_tokens=len(tokens),neural_mismatches=0,
                  positions=reference['steps'],observed_text=text,output_source='actual GPU BYTE records',
                  final_ram_bytes_exact=65536,final_register_copies_exact=3,
                  checkpoint_restore_offsets=[65536,65792],
                  checkpoint_sha256=checkpoint['checkpoint_sha256'],
                  autonomous=True,reference_state_after_boot=False,
                  same_cpu_constructor_as_accepted_llama_run=True,
                  compute_seconds=report['compute_seconds'],
                  measured_minutes_per_position=report['compute_seconds']/60/reference['steps'],
                  stock_tokens=stock['generated_tokens'],stock_mismatches=0,
                  stock_minimum_free_bytes=stock['minimum_free_bytes'],
                  actual_trace_sha256=digest(main/'actual.tokens.u32'),
                  actual_ram_sha256=digest(main/'actual-ram.bin'),
                  shutdown_recovery=recovery,
                  timing_scope='Retained generation calls; excludes shutdown, setup, IO, waits and replayed work after the last pre-shutdown checkpoint')
    for name,value in [('neural-validation.json',result),('gpu.json',report),('stock.json',stock)]:
        (HERE/'evidence'/name).write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    return result


def checkpoint_binding(report,reference):
    return dict(elf=reference['elf_sha256'],reference=reference['sha256'],
                scripts=report['scripts'],mode=report['mode'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--program',type=Path,required=True)
    p.add_argument('--actual',type=Path,required=True)
    p.add_argument('--source-revision', help='Resolve historical script bindings at this commit')
    a = p.parse_args()
    verify(a.program,a.actual,a.source_revision)
