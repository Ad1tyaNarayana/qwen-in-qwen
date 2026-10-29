"""Run a bounded self-fed boot prefix through the stock HF model on CUDA."""
import gc
import json
from pathlib import Path
import sys
import time
import argparse
import subprocess

ROOT = Path(__file__).resolve().parents[2]
import os
import threading
import importlib.util
parser = argparse.ArgumentParser()
parser.add_argument('--upstream', type=Path, required=True)
parser.add_argument('--inputs', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--max-tokens', type=int, default=100000)
parser.add_argument('--max-seconds', type=float, default=1500)
parser.add_argument('--allocator-gib', type=float, default=14)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
prep_spec = importlib.util.spec_from_file_location('memory_prepare', ROOT / 'experiments/memory-map/prepare.py')
prep = importlib.util.module_from_spec(prep_spec)
prep_spec.loader.exec_module(prep)
package = args.output / ('scratch-'+str(time.time_ns()))
prep.prepare(args.upstream, package, 64, 4096)
sys.path.insert(0, str(package))

import torch
from transformers import GenerationConfig, Qwen3_5ForCausalLM, Qwen3_5TextConfig
from chipformer.q35.emit import check_bf16, emit_layers
from chipformer.q35.tiny import BASE, CODE_GAIN, pick_layout
from chipformer.rv import core, elf, firmware, ref, spec
from chipformer.rv.tiny import embedding_matrix


def main():
    main_report = json.loads((ROOT.parent / 'artifacts/tiny-llm-neural/report.json').read_text())
    assert main_report['complete'] and main_report['halted'] and main_report['mismatches'] == 0
    args.device = 'cuda'
    args.firmware = args.inputs / 'engine.elf'
    watchdog = threading.Timer(args.max_seconds+300, lambda: os._exit(124))
    watchdog.daemon = True
    watchdog.start()
    free, total = torch.cuda.mem_get_info()
    nvidia_before = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free,memory.total,memory.used',
        '--format=csv,noheader,nounits'], text=True).strip()
    smi_free = int(nvidia_before.split(',')[0].strip())*2**20
    cap = int(args.allocator_gib*2**30)
    torch.cuda.set_per_process_memory_fraction(cap/total)
    process_started = time.perf_counter()
    memory_stop = threading.Event()
    memory = {'minimum_free_vram_bytes': min(free,smi_free), 'headroom_stop': False,
              'minimum_nvidia_free_bytes': smi_free}
    def sample_memory():
        samples = 0
        while not memory_stop.wait(1):
            available, _ = torch.cuda.mem_get_info()
            memory['minimum_free_vram_bytes'] = min(memory['minimum_free_vram_bytes'], available)
            samples += 1
            if samples % 5 == 0:
                value = subprocess.check_output(['nvidia-smi','--query-gpu=memory.free',
                    '--format=csv,noheader,nounits'],text=True).strip()
                physical_free = int(value)*2**20
                memory['minimum_nvidia_free_bytes'] = min(memory['minimum_nvidia_free_bytes'],physical_free)
                memory['minimum_free_vram_bytes'] = min(memory['minimum_free_vram_bytes'],physical_free)
                if physical_free < 1.25*2**30:
                    memory['headroom_stop'] = True
            if available < 1.25*2**30:
                memory['headroom_stop'] = True
    memory_thread = threading.Thread(target=sample_memory, daemon=True)
    memory_thread.start()
    from chipformer.q35.fast import exact_matmuls
    exact_matmuls()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(4)
    circuit = core.build()
    layout = pick_layout(len(circuit.dims))
    count = circuit.n_layers()
    vocabulary = -(-(spec.VOCAB + 1) // 64) * 64
    config = Qwen3_5TextConfig(**dict(BASE, hidden_size=layout.hidden, vocab_size=vocabulary,
        num_hidden_layers=count + 1,
        layer_types=["linear_attention"] * count + ["full_attention"],
        linear_num_key_heads=circuit.target.key_heads,
        linear_num_value_heads=circuit.target.value_heads,
        intermediate_size=circuit.target.intermediate, eos_token_id=spec.tok("HALT"),
        pad_token_id=spec.VOCAB, bos_token_id=spec.VOCAB))
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(config)
    parameters = sum(p.numel() for p in model.parameters())
    print(f"meta model has {parameters} parameters, {parameters * 2} bytes in bf16", flush=True)
    required = parameters*2 + 512*2**20
    if min(free,smi_free) < required + 2**30:
        raise RuntimeError(f'Insufficient VRAM headroom: CUDA free={free}, nvidia-smi free={smi_free}, needed={required+2**30}')
    from chipformer.q35 import emit
    import inspect
    namespace = dict(emit.__dict__)
    source = inspect.getsource(emit.emit_layers)
    source = source.replace('def emit_layers(', 'def iter_layers(')
    source = source.replace('            out[p + k] = t', '        yield {p + k:t for k,t in layer.items()}')
    source = source.replace('    return out', '')
    exec(compile(source, '<streamed original emit_layers>', 'exec'), namespace)
    (args.output / 'streamed-emitter.py').write_text(source)
    loaded = set()
    for weights in namespace['iter_layers'](circuit, layout, config.to_dict(), 0, 'model.', None, set(circuit.keep)):
        weights = {k:v.to(torch.bfloat16) for k,v in weights.items()}
        _, unexpected = model.load_state_dict(weights, strict=False, assign=True)
        assert not unexpected
        loaded.update(weights)
        del weights
    weights = {}
    gc.collect()
    weights["model.embed_tokens.weight"] = embedding_matrix(circuit, layout, vocabulary).to(torch.bfloat16)
    norm = torch.full((layout.hidden,), -1.0)
    for dim in circuit.keep:
        norm[layout.dim(dim)] = CODE_GAIN
    check_bf16("final norm", norm)
    weights["model.norm.weight"] = norm.to(torch.bfloat16)
    _, unexpected = model.load_state_dict(weights, strict=False, assign=True)
    assert not unexpected
    loaded.update(weights)
    for name in set(model.state_dict()) - loaded:
        if name == "lm_head.weight":
            continue
        assert name.startswith(f"model.layers.{count}.")
        parent, leaf = name.rsplit(".", 1)
        module = model.get_submodule(parent)
        original = getattr(module, leaf)
        setattr(module, leaf, torch.nn.Parameter(torch.zeros(original.shape, dtype=torch.bfloat16), requires_grad=False))
    model.tie_weights()
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
    model.model.rotary_emb = Qwen3_5TextRotaryEmbedding(config)
    model.eval()
    model.layout = layout
    model.generation_config = GenerationConfig(eos_token_id=spec.tok("HALT"),
                                               pad_token_id=spec.VOCAB, do_sample=False)
    del weights
    gc.collect()
    model = model.to(args.device)
    if args.device == "cuda":
        assert torch.cuda.is_available()
        torch.cuda.synchronize()
        print(f"CUDA model on {torch.cuda.get_device_name()}, allocated {torch.cuda.memory_allocated()} bytes", flush=True)
    program = elf.program(args.firmware)
    import numpy as np
    import hashlib
    metadata = json.loads((args.inputs / 'reference.json').read_text())
    assert hashlib.sha256(args.firmware.read_bytes()).hexdigest() == metadata['elf_sha256']
    trace = args.inputs / 'reference.tokens.u32'
    assert hashlib.sha256(trace.read_bytes()).hexdigest() == metadata['sha256']
    wanted = np.memmap(trace, mode='r', dtype='<u4')[:args.max_tokens].tolist()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    from transformers import StoppingCriteria, StoppingCriteriaList
    from transformers.generation.streamers import BaseStreamer
    class CheckedStreamer(BaseStreamer):
        def __init__(self):
            self.tokens = []
            self.initial = True
        def put(self, value):
            values = value.cpu().tolist()
            if self.initial:
                self.initial = False
                assert values == [[spec.VOCAB, spec.tok("GO")]], values
                return
            for value in values:
                index = len(self.tokens)
                self.tokens.append(value)
                if value != wanted[index]:
                    (output / "mismatch.json").write_text(json.dumps({"position": index, "actual": value, "expected": wanted[index]}))
                    raise AssertionError((index, value, wanted[index]))
            if len(self.tokens) % 8 == 0 or len(self.tokens) == len(wanted):
                torch.cuda.empty_cache()
                elapsed = time.perf_counter() - started
                checkpoint = {"generated_tokens": len(self.tokens), "target_tokens": len(wanted), "mismatches": 0,
                              "elapsed_seconds": elapsed, "tokens": self.tokens,
                              "output_hex": firmware.output_text(self.tokens).encode("latin-1").hex()}
                (output / "checkpoint.json").write_text(json.dumps(checkpoint))
                print(f"checkpoint {len(self.tokens)}/{len(wanted)} tokens in {elapsed:.3f}s", flush=True)
        def end(self):
            (output / "generated-tokens.json").write_text(json.dumps(self.tokens))
    class Deadline(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            return memory['headroom_stop'] or time.perf_counter() - started >= args.max_seconds
    streamer = CheckedStreamer()
    started = time.perf_counter()
    with torch.inference_mode():
        cache = firmware.load_cache(model, circuit, ref.Machine.from_program(program))
        print(f"PAD prefill and snapshot injection took {time.perf_counter() - started:.3f}s", flush=True)
        ids = torch.tensor([[spec.VOCAB, spec.tok("GO")]], device=args.device)
        result = model.generate(input_ids=ids, past_key_values=cache, max_new_tokens=len(wanted), do_sample=False,
                                eos_token_id=spec.tok("HALT"), pad_token_id=spec.VOCAB,
                                streamer=streamer, stopping_criteria=StoppingCriteriaList([Deadline()]))
        generated = result[0, 2:].tolist()
        assert generated == streamer.tokens
    if args.device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    assert generated == wanted[:len(generated)], (generated, wanted)
    report = {"mode": f"stock transformers Qwen3_5ForCausalLM, standalone compiled RISC-V CPU, bf16 {args.device.upper()}",
              "device": args.device,
              "cuda_runtime": torch.version.cuda,
              "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
              "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
              "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
              "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
              "parameters": parameters, "target_tokens": len(wanted), "completed_target": len(generated) == len(wanted), "generated_tokens": len(generated), "tokens": generated,
              "output": firmware.output_text(generated), "mismatches": 0,
              "elapsed_seconds_one_run": elapsed,
              "timed_work": "PAD prefill, state injection, and stock greedy generate; excludes model compilation",
              "torch": torch.__version__, "transformers": __import__("transformers").__version__}
    report['allocator_cap_bytes'] = cap
    report['free_vram_before_bytes'] = free
    report['nvidia_smi_before_free_total_used_mib'] = nvidia_before
    report['total_vram_bytes'] = total
    report['main_run_complete_before_hf'] = True
    report['ram_bytes'] = 4*spec.RAM_WORDS
    report['drom_bytes'] = 4*spec.DROM_WORDS
    report['chip_layers'] = count
    report['hidden_size'] = layout.hidden
    report['vram_sampling_interval_seconds'] = 1
    report['nvidia_smi_sampling_interval_seconds'] = 5
    report['unused_allocator_cache_release_interval_tokens'] = 8
    report['headroom_guard_bytes'] = int(1.25*2**30)
    memory_stop.set()
    memory_thread.join()
    report.update(memory)
    report['stop_reason'] = 'headroom_guard' if memory['headroom_stop'] else 'deadline_or_token_limit'
    report['cuda_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
    assert memory['minimum_free_vram_bytes'] >= 2**30, memory
    report['process_wall_seconds'] = time.perf_counter()-process_started
    report['layer_offload_used'] = False
    report['streamed_emitter'] = 'Original emitter computes each layer unchanged; BF16 weights assigned one layer at a time to avoid a full FP32 state dictionary.'
    report['scope'] = 'Bounded self-fed boot prefix. Does not reach transformer inference or establish complete LLM prediction through stock HF.'
    report['upstream_revision'] = prep.REVISION
    report['reference_trace_sha256'] = metadata['sha256']
    report['tokens_per_second'] = len(generated)/elapsed
    report['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report['streamed_emitter_sha256'] = hashlib.sha256(source.encode()).hexdigest()
    watchdog.cancel()
    if args.firmware:
        report["firmware_sha256"] = __import__("hashlib").sha256(args.firmware.read_bytes()).hexdigest()
        report["bounded_prefix"] = True
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k != 'tokens'}, indent=2))


if __name__ == "__main__":
    main()
