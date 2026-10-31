"""Scan tracked files and optional release archives for prohibited release content."""
import argparse
import re
import subprocess
import tarfile
from pathlib import Path
from release_lib import ROOT, members

LIMIT = 100 * 1024 * 1024
BUNDLE_LIMIT = 256 * 1024 * 1024
SECRETS = [rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
           rb'\bgh[pousr]_[A-Za-z0-9]{30,}\b', rb'\bAKIA[0-9A-Z]{16}\b',
           rb'\bsk-[A-Za-z0-9_-]{40,}\b']


def problems(name, size, prefix, limit):
    path = Path(name.lower())
    errors = []
    if size > limit:
        errors.append(f'file exceeds {limit} bytes')
    if path.suffix in {'.wad', '.iwad', '.pwad', '.nor', '.rgb565', '.rgba', '.rgb'} or prefix[:4] in {b'IWAD', b'PWAD'}:
        errors.append('game data or pixel dump')
    # Doom reports/hashes/source are allowed; executable, trace, state and images are not.
    doom = any('doom' in part or part.startswith('dmc-') for part in path.parts)
    if doom and (path.suffix in {'.elf', '.bin', '.u32', '.pt', '.png', '.jpg', '.bmp', '.raw'}
                 or 'frame' in path.name and path.suffix not in {'.py', '.md', '.json', '.jsonl', '.txt', '.svg'}):
        errors.append('Doom-derived executable, trace, state or imagery')
    if any(re.search(pattern, prefix) for pattern in SECRETS):
        errors.append('likely secret')
    return [f'{name}: {error}' for error in errors]


def scan_tree(root):
    errors = []
    paths = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).split(b'\0')
    for raw in paths:
        if not raw:
            continue
        name = raw.decode()
        path = root / name
        if path.is_symlink():
            target = path.readlink()
            if target.is_absolute() or re.match(r'^[A-Za-z]:[\\/]', str(target)):
                errors.append(f'{name}: absolute symlink')
            elif not path.exists() or not path.resolve().is_relative_to(root.resolve()):
                errors.append(f'{name}: broken or escaping symlink')
            # Apply the filename policy even to links.
            errors.extend(problems(name, 0, b'', LIMIT))
        elif path.is_file():
            errors.extend(problems(name, path.stat().st_size, path.read_bytes(), LIMIT))
        else:
            errors.append(f'{name}: tracked file missing')
    return errors


def scan_bundle(path):
    errors = []
    with tarfile.open(path, 'r:gz') as archive:
        total = 0
        for item in members(archive):
            total += item.size
            if total > 2 * 1024**3:
                raise ValueError('Archive expands beyond 2 GiB')
            if item.size > BUNDLE_LIMIT:
                errors.append(f'{item.name}: bundle file exceeds size limit')
                continue
            errors.extend(problems(item.name, item.size, archive.extractfile(item).read(), BUNDLE_LIMIT))
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--bundle', type=Path)
    args = parser.parse_args()
    errors = scan_tree(args.root)
    if args.bundle:
        try:
            errors.extend(scan_bundle(args.bundle))
        except (ValueError, tarfile.TarError) as error:
            errors.append(str(error))
    for error in errors:
        print('FAIL', error)
    print(f'check_release: {"FAIL" if errors else "PASS"} ({len(errors)} violations)')
    return bool(errors)


if __name__ == '__main__':
    raise SystemExit(main())
