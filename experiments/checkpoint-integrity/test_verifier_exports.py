"""Acceptance and rejection controls on retained private story and Doom exports."""
import argparse
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def link_tree(source, target, omit=()):
    target.mkdir(parents=True)
    for path in source.iterdir():
        if path.name not in omit:
            (target / path.name).symlink_to(path.resolve(), target_is_directory=path.is_dir())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--story-only', action='store_true', help='Run the nine story controls; Doom requires private user-supplied exports')
    args = parser.parse_args()
    results = {}

    def check(label, action, accepted, error_text=None):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                action()
        except (AssertionError, FileNotFoundError) as error:
            if accepted or (error_text and error_text not in str(error)):
                raise
            results[label] = dict(accepted=False, error=f'{type(error).__name__}: {error}')
        else:
            assert accepted, f'{label}: altered export was accepted'
            results[label] = dict(accepted=True)
        print(f'{label}: PASS ({"accepted" if accepted else "rejected"})')

    story = load('story_verifier', 'experiments/tiny-llm-ndot-fused/verify_story.py')
    doom = load('doom_verifier', 'experiments/doom-showcase/verify_runs.py')
    source = ROOT / '.cache/tiny-llm/ndot-story38-20261030-actual'
    journal = (source / 'main/chunks.jsonl').read_text().splitlines()
    assert len(journal) > 2
    with tempfile.TemporaryDirectory(prefix='qwenputer-verifier-controls-') as temporary:
        temporary = Path(temporary)

        def verify_story(actual):
            sys.argv = ['verify_story.py', '--inputs', str(ROOT / '.cache/tiny-llm/ndot-story38-20261030'),
                        '--actual', str(actual), '--out', str(temporary / 'story-validation.json'),
                        '--source-revision', '017c39c']
            story.main()

        check('story_unmodified', lambda: verify_story(source), True)
        terminal = json.loads(journal[-1])
        terminal['compute_seconds'] += 1
        zero = json.loads(journal[0])
        zero['amount'] = 0
        bad_hash = json.loads(journal[0])
        bad_hash['sha256'] = '0' * 64
        decreased = json.loads(journal[1])
        decreased['compute_seconds'] = 0
        cases = {
            'empty_journal': ([], 'Chunk journal is empty'),
            'deleted_row': (journal[:1] + journal[2:], 'Chunk coverage is not contiguous'),
            'reordered_rows': ([journal[1], journal[0]] + journal[2:], 'Chunk coverage is not contiguous'),
            'terminal_time_mismatch': (journal[:-1] + [json.dumps(terminal)], 'Terminal chunk time differs'),
            'zero_amount': ([json.dumps(zero)] + journal[1:], 'Chunk coverage is not contiguous'),
            'missing_tail': (journal[:-1], 'does not cover the complete trace'),
            'altered_chunk_hash': ([json.dumps(bad_hash)] + journal[1:], None),
            'decreasing_time': ([journal[0], json.dumps(decreased)] + journal[2:], 'Chunk timer decreased'),
        }
        for label, (rows, error_text) in cases.items():
            target = temporary / label
            link_tree(source, target, {'main'})
            link_tree(source / 'main', target / 'main', {'chunks.jsonl'})
            (target / 'main/chunks.jsonl').write_text('\n'.join(rows) + ('\n' if rows else ''))
            check('story_' + label, lambda: verify_story(target), False, error_text)

        if not args.story_only:
            cache = doom.CACHE
            check('doom_ff_full_unmodified', lambda: doom.verify('ff-full'), True)
            check('doom_unified_unmodified', lambda: doom.verify('unified-full'), True)
            for label, omitted in [('missing_ram', {'final-gpu-ram.bin'}),
                                   ('missing_registers', {'actual-registers.json'}),
                                   ('altered_registers', {'actual-registers.json'})]:
                doom.CACHE = temporary / label
                target = doom.CACHE / 'export-unified-full'
                link_tree(cache / 'export-unified-full', target, omitted)
                if label == 'altered_registers':
                    registers = json.loads((cache / 'export-unified-full/actual-registers.json').read_text())
                    # Keep all three copies equal: equality among copies alone is insufficient.
                    for values in registers.values():
                        values[1] ^= 1
                    (target / 'actual-registers.json').write_text(json.dumps(registers))
                expected = {'missing_ram': 'final-gpu-ram.bin', 'missing_registers': 'actual-registers.json',
                            'altered_registers': 'Final registers differ from reference'}[label]
                check('doom_' + label, lambda: doom.verify('unified-full'), False, expected)
    args.out.write_text(json.dumps(dict(status='pass', source_revision='017c39c',
        verifier_sha256=dict(story=story.sha(Path(story.__file__)), doom=doom.sha(Path(doom.__file__))),
        scope='story only; Doom NOT checked' if args.story_only else 'story and Doom', results=results), indent=2) + '\n')
    print(f'All {len(results)} retained-export controls pass')


if __name__ == '__main__':
    main()
