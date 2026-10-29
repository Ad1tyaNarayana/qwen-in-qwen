"""Summarize exact cache evidence without promoting readable differences to passes."""
import argparse
import hashlib
import json
from pathlib import Path


def summarize(path):
    report = json.loads(path.read_text())
    assert report['status'] == 'diagnosed', report.get('error')
    chunks = sorted({c['chunk_size'] for c in report['comparisons']})
    result = dict(source=path.name, kind=report['kind'], generated_tokens=report['generated_tokens'],
                  checked_commits=report['checked_commits'], token_sha256=report['generated_token_sha256'],
                  all_predictions_exact=True, chunk_sizes={})
    for chunk in chunks:
        checks = [c for c in report['comparisons'] if c['chunk_size'] == chunk]
        different = [c for c in checks if c['differences']]
        result['chunk_sizes'][str(chunk)] = dict(
            comparisons=len(checks), boundaries_with_differences=len(different),
            first_difference=different[0] if different else None, final=checks[-1],
            max_numerical_differences=max(c['numerical_differences'] for c in checks),
            max_absolute_difference=max((abs(d['cached']-d['chunked']) for c in checks for d in c['differences']), default=0))
    result['prefix_refinement'] = report.get('prefix_refinement', [])
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--accelerator', type=Path, required=True)
    p.add_argument('--original', type=Path, required=True)
    p.add_argument('--kernel', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--fixed', type=Path)
    p.add_argument('--m3', type=Path)
    p.add_argument('--m4', type=Path)
    a = p.parse_args()
    accelerator, original = summarize(a.accelerator), summarize(a.original)
    assert accelerator['kind'] == 'accelerator' and original['kind'] == 'original'
    kernel = json.loads(a.kernel.read_text())
    cells = {(d['layer'], d['value_head'], d['key_row'], d['value_column'], d['node'])
             for path in (a.accelerator, a.original)
             for c in json.loads(path.read_text())['comparisons'] for d in c['differences']}
    report = dict(status='FAIL', acceptance_case='readable data differs; no exception applies',
                  baseline_exception_acceptance=dict(a=False, b='not attempted because a fails', c=True),
                  accelerator=accelerator, original=original, differing_cells=sorted(cells),
                  readability_proof='counter keys select row 0 on reset and row 1 otherwise; both rows are queried on every non-reset token; column 2 is the counter data output; reference ballast is in columns 0 and 1',
                  mechanism='fp32 cumulative log-decay subtraction changes exp of log-half; even without reset, cumulative log-half rounding differs; one-hot key normalization is exact and beta saturates to one',
                  kernel_source_sha256=kernel['source_sha256'],
                  rejected_fixes=['beta=0.5 with no decay loses the live value by chunk-sum cancellation',
                                  'removing reset decay still leaves cumulative log-half rounding'],
                  script_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                 for name in ('cache_diagnosis.py', 'counter_kernel_diagnosis.py',
                                              'summarize_cache_diagnosis.py', 'stock.py', 'stock_m3.py',
                                              'constructor.py', 'reuse.py', 'oracle.py', 'reference.py')},
                  baseline_constructor_was_unchanged=True, M3='not summarized', M4='not attempted',
                  M5='requires M3 and M4, aligned stories260K ndot guest, per-dot prefix and all 64 integer predictions/logit hashes/decoded bytes/KV clips/final architectural state')
    if a.fixed:
        fixed = json.loads(a.fixed.read_text())
        assert fixed['status'] == 'diagnosed' and fixed['acceptance'] == 'exact'
        assert not any(c['differences'] for c in fixed['comparisons'])
        report.update(status='fixed', acceptance_case='constructor fixed; exact state, no exception',
                      fixed=summarize(a.fixed), counter_encoding='181*n; beta=2^-44; write=181*2^44; reference=181*2^17',
                      deviation='opt-in stock-only counter; legacy IR evaluator and compact backend do not support it')
    if a.m3:
        m3 = json.loads(a.m3.read_text())
        report['M3'] = dict(status=m3['status'], runs=m3['runs'], controls=m3['controls'])
    if a.m4:
        m4 = json.loads(a.m4.read_text())
        report['M4'] = dict(status=m4['status'], runs=m4['runs'], memory=m4.get('final_memory'), error=m4.get('error'))
    a.out.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
