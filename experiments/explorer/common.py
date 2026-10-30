"""Shared paths and the pinned, isolated compiler package."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = ROOT / '.cache/explorer'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def compiler():
    CACHE.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT / 'experiments/neural-accelerator'))
    from prepare import prepare
    scratch = Path(tempfile.mkdtemp(prefix='package-', dir=CACHE))
    prepare(scratch)
    from constructor import build
    return build(4)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, separators=(',', ':'), allow_nan=False).encode()
    if path.suffix == '.gz':
        path.write_bytes(gzip.compress(raw, mtime=0))
    else:
        path.write_bytes(raw)
