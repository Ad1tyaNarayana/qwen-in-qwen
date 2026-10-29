"""Independent Mac acceptance of the unified circuit's Gate 1 artifacts."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--actual', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--m2', type=Path, required=True)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--stock', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    report = dict(status='running', gate=1, runs=[])
    try:
        weights = json.loads(a.weights.read_text())
        m2 = json.loads(a.m2.read_text())
        gpu = json.loads((a.actual / 'run/report.json').read_text())
        exit_report = json.loads((a.actual / 'exit.json').read_text(encoding='utf-8-sig'))
        assert weights['status'] == m2['status'] == gpu['status'] == 'pass'
        assert exit_report['exit_code'] == 0
        assert len(weights['tensors']) == 897 and all(x['unequal_words'] == 0 for x in weights['tensors'])
        generated = weights['graph_generated_sha256']
        assert m2['source_manifest']['generated_sha256'] == gpu['constructor']['generated_sha256'] == generated
        accepted = json.loads(Path(__file__).with_name('stock-m4.json').read_text())
        assert gpu['constructor']['base_sha256'] == accepted['constructor']['base_sha256']
        for field in ('generated_sha256', 'arithmetic_sha256', 'reuse_sha256'):
            assert weights['constructor'][field] == accepted['constructor'][field], field
        assert len(gpu['runs']) == 4
        for name, count, stock_name in [('ndot', 8192, 'ndot'), ('stories-stock', 8192, 'stories260K'),
                                        ('doom', 262144, None), ('stories', 262144, None)]:
            actual_path = a.actual / 'run' / name / 'actual.tokens.u32'
            actual = np.fromfile(actual_path, dtype='<u4')
            if stock_name:
                path = a.stock / (stock_name + '.tokens.u32')
                binding = next(x for x in accepted['runs'] if x['guest'] == stock_name)
                assert sha(path) == binding['token_sha256']
            else:
                path = a.inputs / name / 'reference.tokens.u32'
                binding = json.loads(path.with_name('reference.json').read_text())
                assert sha(path) == binding['sha256']
            expected = np.memmap(path, mode='r', dtype='<u4')[:count]
            assert len(actual) == count
            wrong = np.flatnonzero(actual != expected)
            assert len(wrong) == 0, (name, int(wrong[0]))
            report['runs'].append(dict(guest=name, tokens=count, mismatches=0, sha256=sha(actual_path)))
        report.update(status='pass', generated_sha256=generated, bf16_tensors=897,
                      weight_report_sha256=sha(a.weights), m2_report_sha256=sha(a.m2),
                      gpu_report_sha256=sha(a.actual / 'run/report.json'),
                      verification='Mac comparison of actual GPU bytes against stock exports and independent legacy traces')
    except Exception as error:
        report.update(status='fail', error=repr(error))
        raise
    finally:
        a.out.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
