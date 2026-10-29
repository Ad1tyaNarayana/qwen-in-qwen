"""Gate 1 autonomous unfused prefixes, bound to actual stock-HF token files."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import traceback

from prepare import ROOT, prepare
from legacy_neural import write_json, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--stock', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', gate=1, runs=[], backend='unfused adapted Executor2 / FreeRun',
                  initial_snapshot_only=True, reference_inputs=False)
    write_json(a.out / 'report.json', report)
    try:
        import numpy as np
        import torch
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.set_per_process_memory_fraction(8*2**30 / torch.cuda.get_device_properties(0).total_memory)
        module_spec = importlib.util.spec_from_file_location('unified_memory_prepare', ROOT / 'experiments/memory-map/prepare.py')
        memory_prepare = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(memory_prepare)
        report['prepare_manifest'] = memory_prepare.prepare(ROOT.parent / '.upstream/qwenputer', a.out / 'scratch', 64, 4096)
        sys.path.insert(0, str(a.out / 'scratch'))
        from chipformer.rv import elf, spec
        from constructor import build, snapshot
        from compact import freerun_class
        from stock_m3 import workload
        import reference
        c = build(4)
        report.update(constructor=c.source_manifest, geometry=c.stats(), gpu=torch.cuda.get_device_name())
        _, source = workload(0, 7)
        from chipformer.rv.asm import assemble
        ndot = assemble(source.replace('li s3, 128', 'li s3, 2048'))
        for name, program, tokens, stock_name in (
            ('ndot', ndot, 8192, 'ndot'),
            ('stories-stock', elf.program(a.inputs / 'stories/engine.elf'), 8192, 'stories260K'),
            ('doom', elf.program(a.inputs / 'doom/engine.elf'), 262144, None),
            ('stories', elf.program(a.inputs / 'stories/engine.elf'), 262144, None),
        ):
            initial = reference.Machine.from_program(program)
            initial.maximum = 4
            if stock_name:
                accepted = json.loads(Path(__file__).with_name('stock-m4.json').read_text())
                binding = next(x for x in accepted['runs'] if x['guest'] == stock_name)
                path = a.stock / (stock_name + '.tokens.u32')
                assert sha(path) == binding['token_sha256']
            else:
                fixture = a.inputs / name
                meta = json.loads((fixture / 'reference.json').read_text())
                assert sha(fixture / 'engine.elf') == meta['elf_sha256']
                path = fixture / 'reference.tokens.u32'
                assert sha(path) == meta['sha256']
            expected = np.memmap(path, dtype='<u4', mode='r')[:tokens]
            snapshot(c, initial)
            directory = a.out / name
            directory.mkdir()
            FreeRun = freerun_class(directory)
            fr = FreeRun(c, 1, 'cuda', compile_=False, v2=True)
            fr.load(0, spec.tok('GO'))
            fr.ex.reset()
            fr.ex.y.zero_()
            fr.capture()
            fr.load(0, spec.tok('GO'))
            fr.ex.reset()
            fr.ex.y.zero_()
            fr.tok.zero_()
            run = dict(guest=name, tokens=0, mismatches=0, status='running', stock_comparison=bool(stock_name))
            report['runs'].append(run)
            with (directory / 'actual.tokens.u32').open('wb') as stream, torch.no_grad():
                for offset in range(0, tokens, 4096):
                    actual = fr.run(min(4096, tokens-offset))[0].astype('<u4')
                    stream.write(actual.tobytes())
                    stream.flush()
                    wrong = np.flatnonzero(actual != expected[offset:offset+len(actual)])
                    run.update(tokens=offset+len(actual), mismatches=int(len(wrong)))
                    if len(wrong):
                        i = int(wrong[0])
                        run['first_mismatch'] = dict(offset=offset+i, actual=int(actual[i]), expected=int(expected[offset+i]))
                        raise AssertionError(run['first_mismatch'])
                    write_json(a.out / 'report.json', report)
                    print(json.dumps(run), flush=True)
            run.update(status='pass', sha256=sha(directory / 'actual.tokens.u32'))
            del fr
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        report['status'] = 'pass'
    except Exception as error:
        report.update(status='fail', error=repr(error), traceback=traceback.format_exc())
        print(report['traceback'], flush=True)
    finally:
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(a.out / 'report.json', report)
    raise SystemExit(report['status'] != 'pass')


if __name__ == '__main__':
    main()
