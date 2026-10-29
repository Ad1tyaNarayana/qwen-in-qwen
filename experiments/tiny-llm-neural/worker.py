"""Self-fed circuit execution with actual-state checkpoints and exact trace checks."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time
import types


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace(temp, path):
    for attempt in range(100):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if attempt == 99:
                raise
            time.sleep(.05)


def atomic_json(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2)+'\n')
    replace(temp, path)


def build(a, meta):
    import torch
    import triton
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(8*2**30 / torch.cuda.get_device_properties(0).total_memory)
    prep_spec = importlib.util.spec_from_file_location('memory_prepare', a.repo / 'experiments/memory-map/prepare.py')
    prep = importlib.util.module_from_spec(prep_spec)
    prep_spec.loader.exec_module(prep)
    manifest = prep.prepare(a.upstream, a.out / 'scratch', 64, 4096)
    sys.path.insert(0, str(a.out / 'scratch'))
    from chipformer.rv import core, elf, ref, spec, fastexec
    assert spec.RAM_WORDS*4 == 65536 and spec.DROM_WORDS*4 == 4194304
    program = elf.program(a.inputs / 'engine.elf')
    circuit = core.build()
    core.snapshot(circuit, ref.Machine.from_program(program))
    source = Path(fastexec.__file__).read_text()
    patched = source.replace('nf, nb = len(C.OUT_NAMES), C.NPAY',
        'output_names = list(S.FAMILIES) + S.CONTROL_NAMES\n        nf, nb = len(output_names), C.NPAY').replace('enumerate(C.OUT_NAMES)', 'enumerate(output_names)')
    assert source != patched
    module = types.ModuleType('standalone_freerun')
    exec(compile(patched, str(a.out / 'standalone-freerun.py'), 'exec'), module.__dict__)
    (a.out / 'standalone-freerun.py').write_text(patched)
    fr = module.FreeRun(circuit, 1, 'cuda', compile_=False, v2=True)
    fusion_path = a.repo / 'experiments/dmc-feedback-fusion/fusion.py'
    fusion_spec = importlib.util.spec_from_file_location('stacked_fusion', fusion_path)
    fusion = importlib.util.module_from_spec(fusion_spec)
    fusion_spec.loader.exec_module(fusion)
    fused = fusion.install(fr)
    def reset():
        fr.load(0, spec.tok('GO'))
        fr.ex.reset()
        fr.ex.y.zero_()
        fr.tok.zero_()
        fused['scratch_y'].zero_()
    reset()
    fr.capture()
    reset()
    metadata = dict(mode='fp32 compact compiled Qwen3.5-layer circuit; stacked ROM and feedback fusion',
        stock_hf=False, torch=torch.__version__, triton=triton.__version__, gpu=torch.cuda.get_device_name(),
        prepare_manifest=manifest, ram_bytes=65536, drom_bytes=4194304, code_rom_bytes=262144,
        reference_tokens_used_as_inputs=False, reference_machine_run_called=False,
        boot_state_only=True, explicit_full_history_reset=True,
        scripts={str(p.relative_to(a.repo)):sha(p) for p in [Path(__file__), fusion_path,
            a.repo / 'experiments/dmc-rom-fusion/fusion.py']})
    return fr, fused, spec, metadata


def save(fr, fused, path, binding, offset, seconds, wall):
    import torch
    assert not torch.count_nonzero(fr.ex.y).item()
    assert not torch.count_nonzero(fused['scratch_y']).item()
    heads = [st for st in fr.ex.steps if st['kind'] == 'head']
    state = dict(binding=binding, offset=offset, compute_seconds=seconds, retained_worker_seconds=wall,
                 x=fr.x.detach().cpu().clone(), tok=fr.tok.detach().cpu().clone(),
                 heads=[{k:st[k].detach().cpu().clone() for k in ('S','hist')} for st in heads])
    temp = path.with_suffix('.tmp')
    torch.save(state, temp)
    replace(temp, path)


def restore(fr, fused, path, binding):
    import torch
    state = torch.load(path, map_location='cpu', weights_only=True)
    assert state['binding'] == binding
    heads = [st for st in fr.ex.steps if st['kind'] == 'head']
    assert len(heads) == len(state['heads'])
    for st, saved in zip(heads, state['heads']):
        for k in ('S','hist'):
            assert st[k].shape == saved[k].shape
            st[k].copy_(saved[k])
    fr.x.copy_(state['x'])
    fr.tok.copy_(state['tok'])
    fr.ex.y.zero_()
    fused['scratch_y'].zero_()
    return state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--stop-after', type=int)
    p.add_argument('--max-seconds', type=int, default=14400)
    a = p.parse_args()
    a.repo = a.repo.resolve()
    a.out.mkdir(parents=True, exist_ok=True)
    watchdog = threading.Timer(a.max_seconds, lambda: os._exit(124))
    watchdog.daemon = True
    watchdog.start()
    process_started = time.perf_counter()
    report = dict(complete=False)
    try:
        import numpy as np
        import torch
        meta = json.loads((a.inputs / 'reference.json').read_text())
        assert sha(a.inputs / 'engine.elf') == meta['elf_sha256']
        assert sha(a.inputs / 'reference.tokens.u32') == meta['sha256']
        expected = np.memmap(a.inputs / 'reference.tokens.u32', mode='r', dtype='<u4')
        assert len(expected) == meta['tokens']
        fr, fused, spec, metadata = build(a, meta)
        report.update(metadata)
        binding = dict(elf=meta['elf_sha256'], reference=meta['sha256'], scripts=metadata['scripts'])
        checkpoint = a.out / 'checkpoint.pt'
        trace = a.out / 'actual.tokens.u32'
        offset, seconds, previous_wall = 0, 0., 0.
        if checkpoint.exists():
            state = restore(fr, fused, checkpoint, binding)
            offset, seconds = state['offset'], state['compute_seconds']
            previous_wall = state['retained_worker_seconds']
            assert trace.stat().st_size >= offset*4
            with trace.open('r+b') as f:
                f.truncate(offset*4)
            if offset:
                actual = np.memmap(trace, mode='r', dtype='<u4')
                assert np.array_equal(actual, expected[:offset])
                del actual
        else:
            assert not trace.exists()
            trace.touch()
        initial_offset = offset
        report['started_from_checkpoint_offset'] = initial_offset
        start = time.perf_counter()
        last_checkpoint = offset
        target = min(a.stop_after or len(expected), len(expected))
        with trace.open('ab') as output, torch.no_grad():
            while offset < target:
                amount = min(8192, target-offset)
                t = time.perf_counter()
                generated = fr.run(amount)[0]
                seconds += time.perf_counter()-t
                wanted = expected[offset:offset+amount]
                if not np.array_equal(generated, wanted):
                    generated.astype('<u4').tofile(a.out / 'mismatch-chunk.u32')
                    wrong = np.flatnonzero(generated != wanted)
                    raise AssertionError(dict(offset=offset, mismatches=len(wrong), first=int(wrong[0]),
                        actual=int(generated[wrong[0]]), expected=int(wanted[wrong[0]])))
                output.write(generated.astype('<u4').tobytes())
                output.flush()
                offset += amount
                progress = dict(complete=False, generated_tokens=offset, total_tokens=len(expected), mismatches=0,
                    compute_seconds=seconds, tokens_per_second=offset/seconds,
                    checkpoint_offset=last_checkpoint, updated_utc=datetime.now(timezone.utc).isoformat(),
                    process_seconds=time.perf_counter()-process_started)
                atomic_json(a.out / 'heartbeat.json', progress)
                with (a.out / 'chunks.jsonl').open('a') as log:
                    log.write(json.dumps(dict(end_offset=offset, amount=amount, compute_seconds=seconds,
                        sha256=hashlib.sha256(generated.astype('<u4').tobytes()).hexdigest()))+'\n')
                if offset-last_checkpoint >= 262144:
                    save(fr, fused, checkpoint, binding, offset, seconds, previous_wall+time.perf_counter()-start)
                    last_checkpoint = offset
                    print(json.dumps(progress), flush=True)
        save(fr, fused, checkpoint, binding, offset, seconds, previous_wall+time.perf_counter()-start)
        sys.path.insert(0, str(a.repo / 'experiments/dmc-cuda/source/helpers'))
        from state import read_ram
        ram = read_ram(fr.ex, spec.RAM_BASE, 65536)
        (a.out / 'actual-ram.bin').write_bytes(ram)
        actual = np.memmap(trace, mode='r', dtype='<u4')
        observed = bytes(spec.family(int(t))[1] for t in actual if spec.family(int(t))[0] == 'BYTE'
                         and spec.family(int(t))[1] < 256)
        (a.out / 'actual-output.bin').write_bytes(observed)
        complete = offset == len(expected)
        if complete:
            assert int(fr.tok[0].item()) == spec.tok('HALT')
            assert sha(trace) == meta['sha256']
            assert hashlib.sha256(observed).hexdigest() == meta['output_sha256']
            assert hashlib.sha256(ram).hexdigest() == meta['final_ram_sha256']
        report.update(complete=complete, generated_tokens=offset, total_tokens=len(expected), mismatches=0,
            compute_seconds=seconds, tokens_per_second=offset/seconds,
            retained_worker_seconds=previous_wall+time.perf_counter()-start,
            process_wall_seconds=time.perf_counter()-process_started,
            output=observed.decode(errors='replace'), output_source='BYTE records from actual self-fed GPU trace',
            actual_trace_sha256=sha(trace), actual_ram_sha256=hashlib.sha256(ram).hexdigest(),
            halted=complete, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            timed_work='FreeRun graph generation including feedback, trace copies, synchronization and CPU transfer. Excludes setup, validation, IO, observation and checkpoints.')
    except Exception as error:
        import traceback
        report.update(error=repr(error), traceback=traceback.format_exc())
        print(report['traceback'], flush=True)
    finally:
        atomic_json(a.out / 'report.json', report)
        watchdog.cancel()
        print(json.dumps(report, indent=2), flush=True)
    if 'error' in report:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
