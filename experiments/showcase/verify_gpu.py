"""Independently accept every actual GPU token, all RAM and each register copy."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent


def verify(actual,demo,spec,source_revision=None):
    revision = (subprocess.check_output(['git', 'rev-parse', '--verify', source_revision+'^{commit}'],
                cwd=ROOT, text=True).strip() if source_revision else None)

    def source_digest(relative):
        relative = relative.replace('\\', '/')
        if revision:
            return hashlib.sha256(subprocess.check_output(['git', 'show', revision+':'+relative], cwd=ROOT)).hexdigest()
        return hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()

    fixture=ROOT/'.cache/showcase'/demo
    reference=json.loads((fixture/'reference.json').read_text())
    build=json.loads((fixture/'build.json').read_text())
    weights=json.loads((HERE/'evidence/weights.json').read_text())
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha(fixture/'engine.elf')==reference['elf_sha256']==build['elf_sha256']
    assert sha(fixture/'reference.tokens.u32')==reference['sha256']
    assert sha(fixture/'drom.bin')==build['drom_sha256']
    expected=np.fromfile(fixture/'reference.tokens.u32',dtype='<u4')
    run=actual/demo/'fused'
    report=json.loads((run/'report.json').read_text())
    observed=np.fromfile(run/'actual.tokens.u32',dtype='<u4')
    assert len(observed)==len(expected)==reference['tokens']
    assert np.array_equal(observed,expected)
    assert (run/'actual-ram.bin').read_bytes()==(fixture/'reference-ram.bin').read_bytes()
    registers=json.loads((run/'actual-registers.json').read_text())
    assert set(registers)=={'regs.rs1','regs.rs2','regs.rd'}
    assert all(regs==reference['final_registers'] for regs in registers.values())
    decoded=bytes(value for family,value in map(lambda x:spec.family(int(x)),observed) if family=='BYTE' and value<256)
    assert decoded==(run/'actual-output.bin').read_bytes()==(fixture/'reference-output.bin').read_bytes()
    assert report['complete'] and report['halted'] and report['mismatches']==0
    assert report['peak_allocated_bytes']<=8*2**30
    assert report['tokens_per_second']==len(expected)/report['compute_seconds']
    assert report['network_weight_sha256']==weights['sha256']
    assert {k:v for k,v in report['constructor'].items() if k!='base_sha256'}=={k:v for k,v in weights['constructor'].items() if k!='base_sha256'}
    prefix=json.loads((run/'prefix-report.json').read_text())
    restore=json.loads((run/'restore-report.json').read_text())
    assert prefix['started_from_checkpoint_offset']==0
    assert restore['started_from_checkpoint_offset']==min(65536,len(expected))
    assert report['started_from_checkpoint_offset']==min(65792,len(expected))
    chunks=[json.loads(line) for line in (run/'chunks.jsonl').read_text().splitlines()]
    end=0
    for chunk in chunks:
        assert chunk['end_offset']==end+chunk['amount']
        assert hashlib.sha256(observed[end:chunk['end_offset']].tobytes()).hexdigest()==chunk['sha256']
        end=chunk['end_offset']
    assert end==len(expected)
    assert report['compute_seconds']==chunks[-1]['compute_seconds']>0
    assert report['scripts']['experiments/showcase/gpu_worker.py']==sha(HERE/'gpu_worker.py')
    for relative,digest in report['scripts'].items():
        assert source_digest(relative)==digest,relative
    stock=actual/demo/'stock'
    stockreport=json.loads((stock/'report.json').read_text())
    stocktokens=np.fromfile(stock/'actual.tokens.u32',dtype='<u4')
    target=min(16384 if demo=='mnist' else 4096,len(expected))
    assert len(stocktokens)==target and np.array_equal(stocktokens,expected[:target])
    assert stockreport['status']=='pass' and stockreport['mismatches']==0
    assert stockreport['minimum_free_bytes']>=512*2**20
    assert stockreport['allocator_cap_bytes']<=15*2**30
    assert stockreport['peak_allocated_bytes']<=15*2**30
    assert stockreport['peak_reserved_bytes']<=15*2**30
    assert not stockreport['layer_offload_used'] and not stockreport['reference_tokens_used_as_inputs']
    assert stockreport['model']['network_weight_sha256']==weights['sha256']
    assert stockreport['model']['wrapper_sha256']==sha(HERE/'gpu_stock.py')
    assert stockreport['worker_sha256']==sha(ROOT/'experiments/tiny-llm-ndot/stock_prefix.py')
    assert stockreport['elf_sha256']==reference['elf_sha256']
    assert stockreport['generated_tokens']==target
    assert stockreport['actual_sha256']==sha(stock/'actual.tokens.u32')
    for stage in ['prefix','restore','full','stock']:
        assert json.loads((actual/f'{demo}-{stage}.exit.json').read_text(encoding='utf-8-sig'))['exit_code']==0
    evidence=HERE/demo/'evidence';evidence.mkdir(parents=True,exist_ok=True)
    import shutil
    for name,path in [('gpu-report.json',run/'report.json'),('stock-report.json',stock/'report.json')]:shutil.copyfile(path,evidence/name)
    result=dict(status='pass',instructions=reference['instructions'],trace_tokens=len(expected),mismatches=0,
        final_ram_bytes_checked=65536,all_three_register_copies_equal=True,output=decoded.decode(),
        output_source='BYTE records independently decoded from actual self-fed GPU trace',
        generation_seconds=report['compute_seconds'],tokens_per_second=report['tokens_per_second'],
        stock_hf_prefix_tokens_checked=len(stocktokens),network_weight_sha256=weights['sha256'],
        actual_trace_sha256=sha(run/'actual.tokens.u32'),final_ram_sha256=sha(run/'actual-ram.bin'),
        elf_sha256=reference['elf_sha256'],drom_sha256=build['drom_sha256'],
        checkpoint_restore_offsets=[restore['started_from_checkpoint_offset'],report['started_from_checkpoint_offset']],
        actual_evidence_path=str(run.relative_to(ROOT)),stock_minimum_free_bytes=stockreport['minimum_free_bytes'])
    (evidence/'neural-validation.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('actual',type=Path);p.add_argument('--source-revision', help='Resolve historical script bindings at this commit');p.add_argument('--demo',nargs='+',default=['sha256','life','tictactoe','mnist','mnist-scalar']);a=p.parse_args()
    sys.path.insert(0,str(ROOT/'.cache/showcase/sha256'))
    # Prepare an independent clean spec with the fixed map, never GPU state.
    sys.path.insert(0,str(ROOT/'experiments/neural-accelerator'))
    from prepare import prepare
    import time
    prepare(ROOT/'.cache/showcase'/f'validation-package-{time.time_ns()}')
    from chipformer.rv import spec
    for demo in a.demo:print(json.dumps(verify(a.actual.resolve(),demo,spec,a.source_revision),indent=2))
