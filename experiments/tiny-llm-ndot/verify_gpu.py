"""Independently validate exported M5 GPU files on the Mac."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_trace(actual, expected, count):
    observed = np.memmap(actual, mode='r', dtype='<u4')
    wanted = np.memmap(expected, mode='r', dtype='<u4')
    assert len(observed) == count and len(wanted) >= count
    wrong = sum(int(np.count_nonzero(observed[i:min(i+65536,count)] != wanted[i:min(i+65536,count)])) for i in range(0,count,65536))
    assert wrong == 0, wrong
    return observed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--actual',type=Path,required=True)
    parser.add_argument('--main-only',action='store_true')
    parser.add_argument('--baselines-only',action='store_true')
    parser.add_argument('--evidence',type=Path,required=True)
    args = parser.parse_args()
    args.evidence.mkdir(parents=True,exist_ok=True)
    fixture = ROOT/'.cache/tiny-llm/ndot5-20261030'
    original = ROOT/'.cache/tiny-llm/neural-stories'
    reference = json.loads((fixture/'reference.json').read_text())
    directory = args.actual/'main'
    gpu = json.loads((directory/'report.json').read_text())
    assert gpu['complete'] and gpu['halted'] and gpu['mismatches']==0
    actual = check_trace(directory/'actual.tokens.u32',fixture/'reference.tokens.u32',reference['tokens'])
    assert sha(directory/'actual.tokens.u32') == reference['sha256']
    assert (directory/'actual-ram.bin').read_bytes() == (fixture/'reference-ram.bin').read_bytes()
    regs = json.loads((directory/'actual-registers.json').read_text())
    assert len(regs)==3 and all(values==reference['final_registers'] for values in regs.values())
    sys.path.insert(0,str(fixture/'reference-package'))
    from chipformer.rv import spec
    base,size = spec.FAMILIES['BYTE']
    raw = (actual[(actual>=base)&(actual<base+min(size,256))]-base).astype('u1').tobytes()
    assert raw == (directory/'actual-output.bin').read_bytes() == (fixture/'reference-output.bin').read_bytes()
    chunks = re.split(rb'^@\d+ \d+ \d+ \d+ \d+\n',raw,flags=re.M)[1:]
    text = b''.join(c[1:-1] for c in chunks).decode()
    assert text == 'Once upon a time,'
    for name,digest in gpu['scripts'].items():
        assert sha(ROOT/name.replace('\\','/'))==digest, name
    prefix=json.loads((directory/'prefix-report.json').read_text())
    resume=json.loads((directory/'resume-report.json').read_text())
    assert prefix['generated_tokens']==resume['started_from_checkpoint_offset']==65536
    assert resume['generated_tokens']==gpu['started_from_checkpoint_offset']==65792
    timing=[json.loads(line) for line in (directory/'chunks.jsonl').read_text().splitlines()]
    last=0
    last_seconds=0.0
    for chunk in timing:
        end=chunk['end_offset']
        assert end-last==chunk['amount']
        assert chunk['compute_seconds']>=last_seconds
        last_seconds=chunk['compute_seconds']
        assert hashlib.sha256(actual[last:end].tobytes()).hexdigest()==chunk['sha256']
        last=end
    assert last==reference['tokens'] and timing[-1]['compute_seconds']==gpu['compute_seconds']
    if args.main_only:
        summary=dict(status='pass',text=text,tokens=reference['tokens'],mismatches=0,
            final_ram_bytes_exact=65536,final_register_copies_exact=3,
            final_registers=reference['final_registers'],tokens_per_second=gpu['tokens_per_second'],
            compute_seconds=gpu['compute_seconds'],measured_minutes_per_position=gpu['compute_seconds']/5/60)
        for name,value in [('neural-validation.json',summary),('gpu.json',gpu),('reference5.json',reference)]:
            (args.evidence/name).write_text(json.dumps(value,indent=2)+'\n')
        print(json.dumps(summary,indent=2))
        return
    exits=list(args.actual.glob('*.exit.json'))
    assert len(exits)==(18 if args.baselines_only else 19), len(exits)
    assert all(json.loads(path.read_text(encoding='utf-8-sig'))['exit_code']==0 for path in exits)
    baseline={}
    for label,inputs in [('original-unfused',original),('unified-original',original),('unified-ndot',fixture)]:
        runs=[]
        for index in range(5):
            folder=args.actual/f'{label}-{index}'
            report=json.loads((folder/'report.json').read_text())
            assert report['generated_tokens']==65536 and report['mismatches']==0 and 'error' not in report
            check_trace(folder/'actual.tokens.u32',inputs/'reference.tokens.u32',65536)
            runs.append(dict(tokens_per_second=report['tokens_per_second'],compute_seconds=report['compute_seconds']))
        rates=[r['tokens_per_second'] for r in runs]
        baseline[label]=dict(runs=runs,median_tokens_per_second=statistics.median(rates),
            range_tokens_per_second=[min(rates),max(rates)],
            median_microseconds_per_transition=1e6/statistics.median(rates))
    orig=json.loads((original/'reference.json').read_text())
    baseline['original_unfused_projected_minutes_per_position']=orig['tokens']/baseline['original-unfused']['median_tokens_per_second']/5/60
    baseline['unified_original_projected_minutes_per_position']=orig['tokens']/baseline['unified-original']['median_tokens_per_second']/5/60
    if args.baselines_only:
        (args.evidence/'baseline.json').write_text(json.dumps(baseline,indent=2)+'\n')
        print(json.dumps(baseline,indent=2))
        return
    stock_dir=args.actual/'stock'
    stock=json.loads((stock_dir/'report.json').read_text())
    assert stock['status']=='pass' and stock['generated_tokens']>=16384 and stock['mismatches']==0
    assert stock['allocator_cap_bytes']<=15*2**30 and stock['free_vram_guard_bytes']>=512*2**20
    assert stock['minimum_free_bytes']>=512*2**20
    assert stock['elf_sha256']==reference['elf_sha256']
    assert stock['constructor']['generated_sha256']==gpu['constructor']['generated_sha256']
    assert stock['worker_sha256']==sha(Path(__file__).with_name('stock_prefix.py'))
    assert stock['peak_allocated_bytes']<=stock['allocator_cap_bytes']
    assert stock['stock_generate'] and stock['initial_snapshot_only'] and not stock['layer_offload_used']
    check_trace(stock_dir/'actual.tokens.u32',fixture/'reference.tokens.u32',stock['generated_tokens'])
    checkpoint=json.loads((args.actual/'checkpoint-metadata.json').read_text())
    assert checkpoint['offset']==reference['tokens'] and checkpoint['token']==[spec.tok('HALT')]
    assert checkpoint['binding']['elf']==reference['elf_sha256']
    assert checkpoint['binding']['reference']==reference['sha256']
    summary=dict(status='pass',text=text,tokens=reference['tokens'],mismatches=0,
        final_ram_bytes_exact=65536,final_register_copies_exact=3,final_registers=reference['final_registers'],
        tokens_per_second=gpu['tokens_per_second'],compute_seconds=gpu['compute_seconds'],
        measured_minutes_per_position=gpu['compute_seconds']/5/60,
        stock_tokens=stock['generated_tokens'],stock_mismatches=0,
        baseline=baseline,phase1_fused_historical_minutes_per_position=14.06,
        timing_scope=gpu['timed_work'],fusion='not attempted')
    for name,value in [('validation.json',summary),('gpu.json',gpu),('reference5.json',reference),('stock.json',stock),('baseline.json',baseline)]:
        (args.evidence/name).write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
