"""Prepare a private expanded-map package, retaining the clean-pin checks."""
import hashlib
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def prepare(scratch):
    module_spec = importlib.util.spec_from_file_location(
        'expanded_map_prepare', ROOT / 'experiments/memory-map/prepare.py')
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    manifest = module.prepare(ROOT / '.upstream/qwenputer', scratch, 64, 4096)
    for relative in ('rv/ref.py', 'q35/ir.py', 'q35/prims.py'):
        path = scratch / 'chipformer' / relative
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest[relative] = dict(original_sha256=digest, scratch_sha256=digest, replacements=[])
    sys.path.insert(0, str(scratch))
    return manifest
