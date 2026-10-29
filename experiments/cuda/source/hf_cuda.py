"""Compile upstream RV weights and run a bounded stock Qwen3_5ForCausalLM smoke test."""
import gc
import json
from pathlib import Path
import sys
import time
import argparse
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".upstream/qwenputer/experimental/riscv"))

import torch
from transformers import GenerationConfig, Qwen3_5ForCausalLM, Qwen3_5TextConfig
from chipformer.q35.emit import check_bf16, emit_layers
from chipformer.q35.tiny import BASE, CODE_GAIN, pick_layout
from chipformer.rv import core, elf, firmware, ref, spec
from chipformer.rv.asm import assemble
from chipformer.rv.tiny import embedding_matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--firmware", type=Path)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-tokens", type=int, default=16)
    args = parser.parse_args()
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
    weights = emit_layers(circuit, layout, config.to_dict(), 0, "model.", None, set(circuit.keep))
    for name in list(weights):
        weights[name] = weights[name].to(torch.bfloat16)
    gc.collect()
    weights["model.embed_tokens.weight"] = embedding_matrix(circuit, layout, vocabulary).to(torch.bfloat16)
    norm = torch.full((layout.hidden,), -1.0)
    for dim in circuit.keep:
        norm[layout.dim(dim)] = CODE_GAIN
    check_bf16("final norm", norm)
    weights["model.norm.weight"] = norm.to(torch.bfloat16)
    missing, unexpected = model.load_state_dict(weights, strict=False, assign=True)
    assert not unexpected
    for name in missing:
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
    program = elf.program(args.firmware) if args.firmware else assemble("_start: li a0, 72\nli a7, 1\necall\nli a7, 93\necall")
    reference = ref.Machine.from_program(program)
    wanted = reference.run()[1:]
    if args.firmware:
        wanted = wanted[:args.max_tokens]
    started = time.perf_counter()
    with torch.inference_mode():
        generated = firmware.run(model, circuit, ref.Machine.from_program(program), max_new=len(wanted) if args.firmware else len(wanted) + 2)
    if args.device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    assert generated == wanted, (generated, wanted)
    report = {"mode": f"stock transformers Qwen3_5ForCausalLM, standalone compiled RISC-V CPU, bf16 {args.device.upper()}",
              "device": args.device,
              "cuda_runtime": torch.version.cuda,
              "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
              "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
              "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
              "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
              "parameters": parameters, "generated_tokens": len(generated), "tokens": generated,
              "output": firmware.output_text(generated), "mismatches": 0,
              "elapsed_seconds_one_run": elapsed,
              "timed_work": "PAD prefill, state injection, and stock greedy generate; excludes model compilation",
              "torch": torch.__version__, "transformers": __import__("transformers").__version__}
    report["upstream_revision"] = subprocess.check_output([r"C:\Program Files\Git\cmd\git.exe", "-C", str(ROOT / ".upstream/qwenputer"), "rev-parse", "HEAD"], text=True).strip()
    if args.firmware:
        report["firmware_sha256"] = __import__("hashlib").sha256(args.firmware.read_bytes()).hexdigest()
        report["bounded_prefix"] = True
    output = args.output or ROOT / ("artifacts/hf-game-prefix" if args.firmware else "artifacts/hf-smoke")
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
