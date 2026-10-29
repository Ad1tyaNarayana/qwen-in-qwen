"""Independently compare exported GPU legacy tokens on the local machine."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--actual', type=Path, required=True)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    results = []
    for directory in sorted(a.actual.glob('w*-*')):
        if not directory.is_dir() or not (directory / 'report.json').exists():
            continue
        report = json.loads((directory / 'report.json').read_text())
        actual = np.fromfile(directory / 'actual.tokens.u32', dtype='<u4')
        reference = np.memmap(a.inputs / report['fixture'] / 'reference.tokens.u32', dtype='<u4', mode='r')
        assert len(actual) == report['generated_tokens'] == report['checked_tokens']
        wrong = np.flatnonzero(actual != reference[:len(actual)])
        digest = hashlib.sha256(actual.tobytes()).hexdigest()
        result = dict(fixture=report['fixture'], width=report['width'],
                      generated_tokens=len(actual), mismatches=len(wrong), actual_sha256=digest,
                      worker_status=report['status'], backend=report['backend'])
        if len(wrong):
            i = int(wrong[0])
            result['first_mismatch'] = dict(offset=i, actual=int(actual[i]), expected=int(reference[i]))
        if report['status'] == 'pass':
            assert len(actual) >= 262144 and not len(wrong)
            assert digest == report['actual_sha256'] == report['expected_prefix_sha256']
        results.append(result)
    assert results
    a.out.write_text(json.dumps(dict(results=results, verification='independent local token-file comparison'), indent=2) + '\n')
    print(a.out.read_text())


if __name__ == '__main__':
    main()
