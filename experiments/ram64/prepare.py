"""Copy the pinned upstream package to ignored scratch space and widen standalone RAM."""
import hashlib
from pathlib import Path
import shutil
import subprocess

REVISION = 'ca13310e2f9f788d1dd3feeea8213c1b0c23913c'
PATCHES = {
    'rv/spec.py': ('RAM_BASE, RAM_WORDS = 0x0, 1 << 13', 'RAM_BASE, RAM_WORDS = 0x0, 1 << 14'),
    'rv/core.py': ('EA[15:32]], "in_ram")', 'EA[16:32]], "in_ram")'),
}


def prepare(upstream, scratch, ram_kib):
    revision = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != REVISION:
        raise ValueError(f'Expected upstream {REVISION}, got {revision}')
    changes = subprocess.check_output(['git', '-C', str(upstream), 'status', '--porcelain'], text=True)
    if changes:
        raise ValueError('Upstream has local changes. Use a clean pinned checkout.')
    source = upstream / 'experimental/riscv/chipformer'
    destination = scratch / 'chipformer'
    if destination.exists():
        raise ValueError(f'Scratch package already exists. Choose a fresh --scratch directory: {destination}')
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    manifest = {}
    for relative, (before, after) in PATCHES.items():
        original = (source / relative).read_bytes()
        text = original.decode()
        if text.count(before) != 1:
            raise ValueError(f'Expected one patch site in {relative}')
        patched = text.replace(before, after) if ram_kib == 64 else text
        (destination / relative).write_text(patched)
        manifest[relative] = {
            'original_sha256': hashlib.sha256(original).hexdigest(),
            'scratch_sha256': hashlib.sha256(patched.encode()).hexdigest(),
            'replacements': [[before, after]] if ram_kib == 64 else [],
        }
    return manifest
