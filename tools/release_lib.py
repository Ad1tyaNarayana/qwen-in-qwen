"""Shared release paths, streaming hashes, and safe archive handling."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'release/bundle-manifest.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def members(archive):
    seen = set()
    for item in archive.getmembers():
        path = PurePosixPath(item.name)
        if (not item.isfile() or path.is_absolute() or '..' in path.parts
                or '\\' in item.name or ':' in item.name
                or item.name in seen or str(path) != item.name):
            raise ValueError(f'Unsafe or duplicate archive entry: {item.name}')
        seen.add(item.name)
        yield item


def unpack_verified(bundle, destination):
    manifest = json.loads(MANIFEST.read_text())
    if sha(bundle) != manifest['archive_sha256']:
        raise ValueError('Bundle SHA-256 differs from tracked manifest')
    with tarfile.open(bundle, 'r:gz') as archive:
        entries = list(members(archive))
        if {m.name for m in entries} != set(manifest['files']):
            raise ValueError('Bundle member set differs from tracked allowlist')
        for item in entries:
            expected = manifest['files'][item.name]
            if item.size != expected['bytes']:
                raise ValueError(f'Wrong size: {item.name}')
            target = destination / item.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(item) as source, target.open('wb') as output:
                import shutil
                shutil.copyfileobj(source, output)
            if sha(target) != expected['sha256']:
                raise ValueError(f'Bundle entry SHA-256 mismatch: {item.name}')
