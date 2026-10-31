"""Build the explicit allowlist of non-Doom verifier inputs, never whole cache trees."""
import argparse
import gzip
import hashlib
import io
import re
import json
import tarfile
from pathlib import Path
from release_lib import ROOT, MANIFEST, sha
from check_release import scan_bundle


def allowlist(cache):
    files = set()

    def add(directory, names):
        for name in names.split():
            files.add(directory + '/' + name)

    def package(directory):
        # Only Python source from the frozen independent reference package.
        for p in (cache / directory / 'reference-package').rglob('*.py'):
            files.add(str(p.relative_to(cache)))

    reference = 'engine.elf reference.json reference.tokens.u32 reference-ram.bin reference-output.bin'
    actual = 'actual.tokens.u32 actual-ram.bin actual-registers.json actual-output.bin report.json prefix-report.json restore-report.json chunks.jsonl'
    for name in ['neural-stories', 'ndot5-20261030', 'ndot-story38-20261030']:
        d = 'tiny-llm/' + name
        add(d, reference + ' model.bin')
        package(d)
    d = 'tiny-llm/ndot-story38-20261030-actual'
    add(d, 'sequence.json prefix.exit.json restore.exit.json full.exit.json checkpoint-metadata.json')
    add(d + '/main', actual)
    d = 'checkpoint-integrity/actual-20261030a'
    add(d, 'sequence.json mutations/report.json')
    for name in ['scalar', 'ndot']:
        add(d + '/' + name, actual)
        for stage in ['prefix', 'restore', 'full']:
            add(d, f'{name}-{stage}.exit.json')
    add(d + '/scalar', 'full-initial-failure-report.json')
    d = 'tiny-qwen35/trained-program'
    add(d, reference)
    package(d)
    d = 'tiny-qwen35/actual-covariance-20261030'
    add(d, 'sequence.json prefix.exit.json restore.exit.json full.exit.json stock.exit.json checkpoint-metadata.json recovery-checkpoint.json pre-shutdown/checkpoint.pt pre-shutdown/actual.tokens.u32 stock/report.json stock/actual.tokens.u32')
    add(d + '/main', actual)
    for demo in ['sha256', 'life', 'tictactoe', 'mnist', 'mnist-scalar']:
        add('showcase/' + demo, reference + ' build.json drom.bin')
    add('showcase/tictactoe', 'moves.h')
    for demo in ['mnist', 'mnist-scalar']:
        add('showcase/' + demo, 'model.h')
    add('showcase/mnist-model', 'model.h integer.npz test-images.bin test-labels.bin test-predictions.bin training.json')
    d = 'showcase/actual-20261030a'
    for demo in ['sha256', 'life', 'tictactoe', 'mnist', 'mnist-scalar']:
        add(d + '/' + demo + '/fused', actual)
        add(d + '/' + demo + '/stock', 'report.json actual.tokens.u32')
        for stage in ['prefix', 'restore', 'full', 'stock']:
            add(d, f'{demo}-{stage}.exit.json')
    return sorted(files)


PUBLIC_METADATA = {
    'checkpoint-integrity/actual-20261030a/scalar/full-initial-failure-report.json',
    *(f'showcase/{demo}/build.json' for demo in
      ['sha256', 'life', 'tictactoe', 'mnist', 'mnist-scalar']),
}


def public_metadata(raw):
    """Redact home paths in new distribution copies, preserving all checked fields."""
    pattern = re.compile(r"/Users/[^/\\\s'\"]+|[A-Za-z]:[/\\]+Users[/\\]+[^/\\\s'\"]+")
    count = 0

    def visit(value):
        nonlocal count
        if isinstance(value, str):
            value, changes = pattern.subn('/REDACTED_HOME', value)
            count += changes
            return value
        if isinstance(value, list):
            return [visit(x) for x in value]
        if isinstance(value, dict):
            return {visit(k): visit(v) for k, v in value.items()}
        return value

    result = visit(json.loads(raw))
    return ((json.dumps(result, indent=2)+'\n').encode() if count else raw), count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=ROOT / '.cache')
    args = parser.parse_args()
    output = ROOT / '.cache/release'
    output.mkdir(parents=True, exist_ok=True)
    bundle = output / 'qwenputer-evidence-v1.tar.gz'
    notices = {'LICENSE': ROOT/'LICENSE',
               'THIRD_PARTY_NOTICES.md': ROOT/'THIRD_PARTY_NOTICES.md',
               'QWENPUTER-LICENSE': ROOT/'vendor/qwenputer/LICENSE',
               'LLAMA2C-LICENSE': ROOT/'release/licenses/llama2.c.txt'}
    entries = sorted(allowlist(args.cache) + list(notices))

    def source_for(relative):
        return notices.get(relative, args.cache / relative)
    manifest = {}
    redacted = {}
    # Fail before writing when an allowlisted input is absent or a symlink.
    for relative in entries:
        source = source_for(relative)
        if source.is_symlink() or not source.is_file():
            raise ValueError(f'Missing regular allowlisted input: {relative}')
        manifest[relative] = dict(bytes=source.stat().st_size, sha256=sha(source))
        if relative in PUBLIC_METADATA:
            raw, count = public_metadata(source.read_bytes())
            if count:
                redacted[relative] = raw
                manifest[relative] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                    source_sha256=sha(source), redacted_home_paths=count)
        print(f'{manifest[relative]["bytes"]:12d} {relative}')
    with bundle.open('wb') as stream, gzip.GzipFile(fileobj=stream, mode='wb', mtime=0, filename='') as zipped:
        with tarfile.open(fileobj=zipped, mode='w') as archive:
            for relative in entries:
                source = source_for(relative)
                info = archive.gettarinfo(str(source), arcname=relative)
                info.uid = info.gid = info.mtime = 0
                info.uname = info.gname = ''
                info.mode = 0o644
                if relative in redacted:
                    info.size = len(redacted[relative])
                    archive.addfile(info, io.BytesIO(redacted[relative]))
                else:
                    with source.open('rb') as f:
                        archive.addfile(info, f)
    errors = scan_bundle(bundle)
    if errors:
        raise ValueError('\n'.join(errors))
    if bundle.stat().st_size > 2_000_000_000:
        raise ValueError('Bundle exceeds 2 GB budget')
    MANIFEST.parent.mkdir(exist_ok=True)
    MANIFEST.write_text(json.dumps(dict(format=1, archive=bundle.name,
        archive_bytes=bundle.stat().st_size, archive_sha256=sha(bundle), files=manifest,
        omitted={'Doom': 'Private game data and derived outputs are prohibited.',
                 'training': 'Multi-GB training and quantization sweeps are unnecessary for retained-output checks.',
                 'separate trained PyTorch guest checkpoint': 'Not needed to validate retained traces. The required quantized guest weights are embedded in the program ELF.'},
        metadata_note='Six bundle-only JSON copies redact home-directory paths. source_sha256 binds the unchanged private input; sha256 binds the distributed copy. Numerical fields and verifier assertions are unchanged.',
        checkpoint_note='pre-shutdown/checkpoint.pt is actual CPU circuit recurrent state, not trained guest weights. See THIRD_PARTY_NOTICES.md.'), indent=2) + '\n')
    print(f'Bundle: {bundle.name}: {bundle.stat().st_size} bytes; {len(entries)} entries; expanded {sum(x["bytes"] for x in manifest.values())} bytes')


if __name__ == '__main__':
    main()
