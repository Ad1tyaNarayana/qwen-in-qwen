"""Check reduced full-map self-fed generation against independent legacy traces."""
import argparse
import array
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback
import types

from prepare import ROOT, prepare


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def fixtures(out):
    """Retain the original scalar trace and bind it to the historical full replay."""
    prepare(out / 'scratch')
    from chipformer.rv import elf, ref
    import functools
    ref.decode = functools.lru_cache(None)(ref.decode)
    for name, source in (
        ('doom', ROOT / '.cache/dmc-continuing/legacy-build/doom.elf'),
        ('stories', ROOT / '.cache/tiny-llm/neural-stories/engine.elf'),
    ):
        historical = json.loads((Path(__file__).parent / f'legacy-{name}.json').read_text())
        assert sha(source) == historical['elf_sha256']
        directory = out / name
        directory.mkdir()
        (directory / 'engine.elf').write_bytes(source.read_bytes())
        machine = ref.Machine.from_program(elf.program(source))
        count = 0
        digest = hashlib.sha256()
        with (directory / 'reference.tokens.u32').open('wb') as stream:
            buffer = array.array('I')
            while not machine.halted:
                record = machine.record()
                buffer.extend(record)
                count += len(record)
                if len(buffer) >= 65536 or machine.halted:
                    if sys.byteorder != 'little':
                        buffer.byteswap()
                    data = buffer.tobytes()
                    stream.write(data)
                    digest.update(data)
                    buffer = array.array('I')
        expected = historical['original']
        assert count == expected['tokens']
        assert digest.hexdigest() == expected['trace_sha256']
        assert machine.steps == expected['instructions']
        assert machine.fault is None and machine.exit_code == 0
        write_json(directory / 'reference.json', dict(
            fixture=name, tokens=count, sha256=digest.hexdigest(),
            elf_sha256=sha(source), original_reference=True,
            historical_evidence=f'legacy-{name}.json'))
        print(json.dumps(dict(fixture=name, tokens=count, sha256=digest.hexdigest())), flush=True)


def run(a):
    import numpy as np
    import torch
    report = dict(status='running', fixture=a.fixture, width=a.width,
                  requested_tokens=a.tokens, generated_tokens=0, checked_tokens=0,
                  mismatches=0, stock_hf=False, backend='unfused Executor2 / FreeRun',
                  boot_state_only=True, reference_tokens_used_as_inputs=False,
                  started_utc=datetime.now(timezone.utc).isoformat())
    write_json(a.out / 'report.json', report)
    watchdog = threading.Timer(a.max_seconds, lambda: os._exit(124))
    watchdog.daemon = True
    watchdog.start()
    try:
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.set_per_process_memory_fraction(8 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
        prep_spec = importlib.util.spec_from_file_location('legacy_memory_prepare', ROOT / 'experiments/memory-map/prepare.py')
        prep = importlib.util.module_from_spec(prep_spec)
        prep_spec.loader.exec_module(prep)
        manifest = prep.prepare(a.upstream, a.out / 'scratch', 64, 4096)
        sys.path.insert(0, str(a.out / 'scratch'))
        from chipformer.rv import elf, ref, spec, fastexec
        from constructor import build, snapshot
        inputs = a.inputs / a.fixture
        meta = json.loads((inputs / 'reference.json').read_text())
        assert sha(inputs / 'engine.elf') == meta['elf_sha256']
        assert sha(inputs / 'reference.tokens.u32') == meta['sha256']
        expected = np.memmap(inputs / 'reference.tokens.u32', mode='r', dtype='<u4')
        assert len(expected) == meta['tokens'] and len(expected) >= a.tokens
        circuit = build(a.width)
        snapshot(circuit, ref.Machine.from_program(elf.program(inputs / 'engine.elf')))
        source = Path(fastexec.__file__).read_text()
        old = 'nf, nb = len(C.OUT_NAMES), C.NPAY'
        assert source.count(old) == 1 and source.count('enumerate(C.OUT_NAMES)') == 1
        source = source.replace(old, 'output_names = list(S.FAMILIES) + S.CONTROL_NAMES\n        nf, nb = len(output_names), C.NPAY')
        source = source.replace('enumerate(C.OUT_NAMES)', 'enumerate(output_names)')
        module = types.ModuleType('ndot_standalone_freerun')
        exec(compile(source, str(a.out / 'standalone-freerun.py'), 'exec'), module.__dict__)
        (a.out / 'standalone-freerun.py').write_text(source)
        fr = module.FreeRun(circuit, 1, 'cuda', compile_=False, v2=True)
        fr.load(0, spec.tok('GO'))
        fr.ex.reset()
        fr.ex.y.zero_()
        fr.capture()
        fr.load(0, spec.tok('GO'))
        fr.ex.reset()
        fr.ex.y.zero_()
        fr.tok.zero_()
        report.update(geometry=circuit.stats(), constructor=circuit.source_manifest,
                      prepare_manifest=manifest, fixture_binding=meta,
                      worker_sha256=sha(Path(__file__)), torch=torch.__version__,
                      gpu=torch.cuda.get_device_name(),
                      explicit_zero_history_reset_after_capture=True)
        write_json(a.out / 'report.json', report)
        started = time.perf_counter()
        with (a.out / 'actual.tokens.u32').open('wb') as stream, torch.no_grad():
            for offset in range(0, a.tokens, 4096):
                amount = min(4096, a.tokens - offset)
                actual = fr.run(amount)[0].astype('<u4')
                stream.write(actual.tobytes())
                stream.flush()
                wrong = np.flatnonzero(actual != expected[offset:offset + amount])
                report.update(generated_tokens=offset + amount, checked_tokens=offset + amount,
                              mismatches=report['mismatches'] + len(wrong),
                              elapsed_seconds=time.perf_counter() - started)
                if len(wrong):
                    first = int(wrong[0])
                    report['first_mismatch'] = dict(offset=offset + first,
                        actual=int(actual[first]), expected=int(expected[offset + first]))
                    write_json(a.out / 'report.json', report)
                    raise AssertionError(report['first_mismatch'])
                write_json(a.out / 'report.json', report)
                print(json.dumps({k: report[k] for k in ('fixture', 'width', 'checked_tokens', 'mismatches')}), flush=True)
        report.update(status='pass', actual_sha256=sha(a.out / 'actual.tokens.u32'),
                      expected_prefix_sha256=hashlib.sha256(expected[:a.tokens].tobytes()).hexdigest())
        assert report['actual_sha256'] == report['expected_prefix_sha256']
    except Exception as error:
        report.update(status='fail', error=repr(error), traceback=traceback.format_exc())
        print(report['traceback'], flush=True)
    finally:
        report.update(finished_utc=datetime.now(timezone.utc).isoformat(),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
        write_json(a.out / 'report.json', report)
        watchdog.cancel()
    return 0 if report['status'] == 'pass' else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare-fixtures', action='store_true')
    p.add_argument('--inputs', type=Path)
    p.add_argument('--upstream', type=Path, default=ROOT / '.upstream/qwenputer')
    p.add_argument('--fixture', choices=['doom', 'stories'])
    p.add_argument('--width', type=int, choices=[4, 8], default=4)
    p.add_argument('--tokens', type=int, default=262144)
    p.add_argument('--max-seconds', type=int, default=10800)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    if a.prepare_fixtures:
        fixtures(a.out)
        return
    if a.inputs is None or a.fixture is None or a.tokens <= 0:
        p.error('--inputs, --fixture and positive --tokens are required')
    raise SystemExit(run(a))


if __name__ == '__main__':
    main()
