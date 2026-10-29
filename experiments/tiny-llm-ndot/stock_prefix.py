"""Bounded autonomous stock generation for the unified ndot stories binary."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import subprocess
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/neural-accelerator'))
from legacy_neural import write_json, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tokens', type=int, default=16384)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', generated_tokens=0, mismatches=0,
                  stock_generate=True, initial_snapshot_only=True,
                  reference_tokens_used_as_inputs=False, layer_offload_used=False,
                  worker_sha256=sha(Path(__file__)))
    path = args.out / 'report.json'
    write_json(path, report)
    watchdog = threading.Timer(10800, lambda: os._exit(124))
    watchdog.daemon = True
    watchdog.start()
    try:
        import numpy as np
        import torch
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(15*2**30/total)
        report.update(allocator_cap_bytes=15*2**30, free_vram_guard_bytes=512*2**20,
                      gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                      transformers=__import__('transformers').__version__, minimum_free_bytes=total)
        memory_stop = threading.Event()
        physical = dict(minimum_free_bytes=total, below_guard=False, error=None)
        def physical_memory():
            while not memory_stop.is_set():
                try:
                    value = subprocess.check_output(['nvidia-smi','--query-gpu=memory.free',
                        '--format=csv,noheader,nounits'],text=True).strip()
                    free = int(value)*2**20
                    physical['minimum_free_bytes'] = min(physical['minimum_free_bytes'],free)
                    if free < 512*2**20:
                        physical['below_guard'] = True
                except Exception as error:
                    physical['error'] = repr(error)
                memory_stop.wait(5)
        memory_thread = threading.Thread(target=physical_memory,daemon=True)
        memory_thread.start()
        def guard(required=0):
            if physical['below_guard'] or physical['error']:
                raise RuntimeError(physical)
            torch.cuda.empty_cache()
            free, _ = torch.cuda.mem_get_info()
            report['minimum_free_bytes'] = min(report['minimum_free_bytes'], free)
            if free-required < 512*2**20:
                raise RuntimeError(dict(free_bytes=free, required=required, guard=512*2**20))
        guard()
        module_spec = importlib.util.spec_from_file_location('ndot_stock_prepare', ROOT/'experiments/memory-map/prepare.py')
        prep = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(prep)
        report['prepare_manifest'] = prep.prepare(args.upstream, args.out/'scratch', 64, 4096)
        sys.path.insert(0, str(args.out/'scratch'))
        from chipformer.rv import elf, spec
        from constructor import build
        from stock import build_model, boot_cache
        import reference
        from transformers import StoppingCriteria, StoppingCriteriaList
        from transformers.generation.streamers import BaseStreamer
        circuit = build(4)
        report.update(constructor=circuit.source_manifest, geometry=circuit.stats())
        model, metadata = build_model(circuit, 'cuda', memory_guard=guard)
        report['model'] = metadata
        write_json(path, report)
        versions = [p._version for p in model.parameters()]
        binding = json.loads((args.inputs/'reference.json').read_text())
        assert sha(args.inputs/'engine.elf') == binding['elf_sha256']
        assert sha(args.inputs/'reference.tokens.u32') == binding['sha256']
        expected = np.memmap(args.inputs/'reference.tokens.u32',mode='r',dtype='<u4')[:args.tokens]
        assert len(expected) == args.tokens
        initial = reference.Machine.from_program(elf.program(args.inputs/'engine.elf'))
        initial.maximum = 4
        class Checker(BaseStreamer):
            def __init__(self):
                self.initial = True
                self.tokens = []
            def put(self, value):
                values = value.cpu().tolist()
                if self.initial:
                    self.initial = False
                    assert values == [[spec.VOCAB, spec.tok('GO')]]
                    return
                for token in values:
                    index = len(self.tokens)
                    self.tokens.append(token)
                    assert token == int(expected[index]), (index,token,int(expected[index]))
                if len(self.tokens)%1024 == 0:
                    report.update(generated_tokens=len(self.tokens), elapsed_seconds=time.perf_counter()-started)
                    write_json(path,report)
                    print(json.dumps(dict(tokens=len(self.tokens), elapsed_seconds=report['elapsed_seconds'])),flush=True)
            def end(self):
                pass
        class Guard(StoppingCriteria):
            def __call__(self,input_ids,scores,**kwargs):
                guard()
                return False
        checker = Checker()
        started = time.perf_counter()
        with torch.inference_mode():
            cache = boot_cache(model,circuit,initial)
            generated = model.generate(input_ids=torch.tensor([[spec.VOCAB,spec.tok('GO')]],device='cuda'),
                past_key_values=cache,max_new_tokens=args.tokens,do_sample=False,
                eos_token_id=spec.tok('HALT'),pad_token_id=spec.VOCAB,
                streamer=checker,stopping_criteria=StoppingCriteriaList([Guard()]))
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-started
        assert generated[0,2:].tolist() == checker.tokens == expected.tolist()
        assert versions == [p._version for p in model.parameters()]
        raw = np.asarray(checker.tokens,dtype='<u4').tobytes()
        (args.out/'actual.tokens.u32').write_bytes(raw)
        memory_stop.set()
        memory_thread.join()
        assert not physical['below_guard'] and physical['error'] is None, physical
        report.update(minimum_nvidia_free_bytes=physical['minimum_free_bytes'],
                      nvidia_sampling_interval_seconds=5)
        report['minimum_free_bytes'] = min(report['minimum_free_bytes'],physical['minimum_free_bytes'])
        report.update(status='pass',generated_tokens=len(checker.tokens),mismatches=0,
                      elapsed_seconds=elapsed,tokens_per_second=len(checker.tokens)/elapsed,
                      actual_sha256=hashlib.sha256(raw).hexdigest(), reference_sha256=binding['sha256'],
                      elf_sha256=binding['elf_sha256'], peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      timed_work='PAD prefill, initial snapshot injection and stock greedy generate, including guard and observer; excludes construction')
    except Exception as error:
        report.update(status='fail',error=repr(error),traceback=traceback.format_exc())
        print(report['traceback'],flush=True)
        if 'checker' in locals():
            report['generated_tokens'] = len(checker.tokens)
            (args.out/'actual.tokens.u32').write_bytes(np.asarray(checker.tokens,dtype='<u4').tobytes())
    finally:
        if 'memory_stop' in locals():
            memory_stop.set()
            memory_thread.join()
        if 'physical' in locals():
            report['physical_memory'] = physical
            report['minimum_free_bytes'] = min(report.get('minimum_free_bytes',total),physical['minimum_free_bytes'])
        if 'torch' in locals() and torch.cuda.is_initialized():
            stats = torch.cuda.memory_stats()
            report['allocator'] = dict(configuration=os.environ.get('PYTORCH_ALLOC_CONF','default'),
                allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved(),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                inactive_split_bytes=stats.get('inactive_split_bytes.all.current'))
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(path,report)
        watchdog.cancel()
    raise SystemExit(report['status']!='pass')


if __name__ == '__main__':
    main()
