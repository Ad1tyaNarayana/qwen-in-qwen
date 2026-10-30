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

from checkpoint_integrity import seal, verify


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
    sys.path.insert(0, str(a.repo / 'experiments/neural-accelerator'))
    from chipformer.rv import elf, spec
    from constructor import build as constructor, snapshot
    from compact import freerun_class
    import reference
    initial = reference.Machine.from_program(elf.program(a.inputs / 'engine.elf'))
    initial.maximum = 4
    if a.original_circuit:
        from chipformer.rv import core, ref
        circuit = core.build()
        core.snapshot(circuit, ref.Machine.from_program(elf.program(a.inputs / 'engine.elf')))
    else:
        circuit = constructor(4)
        snapshot(circuit, initial)
    if a.original_circuit:
        from chipformer.rv import fastexec
        source = Path(fastexec.__file__).read_text()
        source = source.replace('nf, nb = len(C.OUT_NAMES), C.NPAY', 'output_names = list(S.FAMILIES) + S.CONTROL_NAMES\n        nf, nb = len(output_names), C.NPAY').replace('enumerate(C.OUT_NAMES)', 'enumerate(output_names)')
        module = types.ModuleType('original_standalone')
        exec(compile(source, '<original standalone FreeRun>', 'exec'), module.__dict__)
        FreeRun = module.FreeRun
    else:
        FreeRun = freerun_class(a.out)
    fr = FreeRun(circuit, 1, 'cuda', compile_=False, v2=True)
    def reset():
        fr.load(0, spec.tok('GO'))
        fr.ex.reset()
        fr.ex.y.zero_()
        fr.tok.zero_()
    reset()
    fr.capture()
    reset()
    metadata = dict(mode=('original full-map unfused Executor2 / FreeRun' if a.original_circuit else 'unified W=4 full-map unfused Executor2 / FreeRun'),
        stock_hf=False, torch=torch.__version__, triton=triton.__version__, gpu=torch.cuda.get_device_name(),
        constructor=getattr(circuit, 'source_manifest', None), geometry=circuit.stats(),
        prepare_manifest=manifest, ram_bytes=65536, drom_bytes=4194304, code_rom_bytes=262144,
        reference_tokens_used_as_inputs=False, reference_machine_run_called=False,
        boot_state_only=True, explicit_full_history_reset=True,
        scripts={str(p.relative_to(a.repo)):sha(p) for p in [Path(__file__),
            a.repo / 'experiments/tiny-llm-ndot/checkpoint_integrity.py',
            a.repo / 'experiments/neural-accelerator/constructor.py',
            a.repo / 'experiments/neural-accelerator/compact.py',
            a.repo / 'experiments/neural-accelerator/integer_counters.py',
            a.repo / 'experiments/neural-accelerator/counter_ir.py']})
    return fr, {'scratch_y':fr.ex.y, 'circuit':circuit}, spec, metadata


def read_registers(fr, circuit):
    import numpy as np
    from chipformer.q35.ir import A_BIT
    found = {}
    for step in fr.ex.steps:
        if step['kind'] != 'head':
            continue
        offset = 0
        for head in step['heads']:
            if head.name in ('regs.rs1', 'regs.rs2', 'regs.rd'):
                rows = step['S'][0, offset].detach().cpu().numpy()
                values = [0]
                for reg in range(1, 32):
                    bits = np.array([sum(rows[reg, col] for col in circuit.reg_cols[b])/A_BIT for b in range(32)])
                    assert np.isin(bits, [0, 1]).all(), (head.name, reg, bits.tolist())
                    values.append(sum(int(bit)<<b for b,bit in enumerate(bits)))
                found[head.name] = values
            offset += len(head.vheads())
    assert len(found) == 3
    assert found['regs.rs1'] == found['regs.rs2'] == found['regs.rd']
    return found


def save(fr, fused, path, binding, offset, seconds, wall):
    import torch
    assert not torch.count_nonzero(fr.ex.y).item()
    assert not torch.count_nonzero(fused['scratch_y']).item()
    heads = [st for st in fr.ex.steps if st['kind'] == 'head']
    state = dict(binding=binding, offset=offset, compute_seconds=seconds, retained_worker_seconds=wall,
                 x=fr.x.detach().cpu().clone(), tok=fr.tok.detach().cpu().clone(),
                 heads=[{k:st[k].detach().cpu().clone() for k in ('S','hist')} for st in heads])
    seal(state)
    temp = path.with_suffix('.tmp')
    torch.save(state, temp)
    replace(temp, path)


def restore(fr, fused, path, binding):
    import torch
    state = torch.load(path, map_location='cpu', weights_only=True)
    verify(state)
    assert state['binding'] == binding
    heads = [st for st in fr.ex.steps if st['kind'] == 'head']
    assert len(heads) == len(state['heads'])
    pairs = [(st[k], saved[k]) for st, saved in zip(heads, state['heads'])
             for k in ('S', 'hist')] + [(fr.x, state['x']), (fr.tok, state['tok'])]
    for target, saved in pairs:
        assert target.shape == saved.shape and target.dtype == saved.dtype
    for target, saved in pairs:
        target.copy_(saved)
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
    p.add_argument('--original-circuit', action='store_true')
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
        binding = dict(elf=meta['elf_sha256'], reference=meta['sha256'], scripts=metadata['scripts'], mode=metadata['mode'])
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
        if offset == len(expected):
            registers = read_registers(fr, fused['circuit'])
            atomic_json(a.out / 'actual-registers.json', registers)
            assert registers['regs.rs1'] == meta['final_registers']
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
