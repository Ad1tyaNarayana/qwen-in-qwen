"""Compile upstream RV weights and run a bounded stock Qwen3_5ForCausalLM smoke test."""
import gc
import json
from pathlib import Path
import sys
import time
import argparse
import subprocess

ROOT = Path(__file__).resolve().parent


import torch
from transformers import GenerationConfig, Qwen3_5ForCausalLM, Qwen3_5TextConfig



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, default=ROOT.parents[1])
    parser.add_argument('--upstream', type=Path)
    parser.add_argument('--scratch', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--ram-kib', type=int, choices=[32, 64], default=64)
    args = parser.parse_args()
    from prepare import prepare
    repo = args.repo.resolve()
    upstream = args.upstream or repo / '.upstream/qwenputer'
    scratch = args.scratch or repo / '.cache' / f'ram{args.ram_kib}'
    output = args.out or repo / 'artifacts' / f'ram{args.ram_kib}'
    manifest = prepare(upstream.resolve(), scratch.resolve(), args.ram_kib)
    sys.path.insert(0, str(scratch.resolve()))
    from chipformer.q35.emit import check_bf16, emit_layers
    from chipformer.q35.tiny import BASE, CODE_GAIN, pick_layout
    from chipformer.rv import core, elf, firmware, ref, spec
    from chipformer.rv.asm import assemble
    from chipformer.rv.tiny import embedding_matrix
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
    boundary = args.ram_kib * 1024
    first, last, sentinel = boundary // 2, boundary - 4, boundary // 2 - 4
    program = assemble(f"""_start:
li a7, 1
li t0, {first}
li t1, 65
sw t1, 0(t0)
lw a0, 0(t0)
ecall
li t0, {last}
li t1, 66
sw t1, 0(t0)
lw a0, 0(t0)
ecall
lw a0, 0(zero)
ecall
li t0, {sentinel}
lw a0, 0(t0)
ecall
li a7, 93
ecall
""")
    def machine():
        m = ref.Machine.from_program(program)
        m.ram[0] = 88
        m.ram[sentinel // 4] = 89
        return m
    reference = machine()
    wanted = reference.run()[1:]
    started = time.perf_counter()
    generated = firmware.run(model, circuit, machine(), max_new=len(wanted) + 2)
    elapsed = time.perf_counter() - started
    assert generated == wanted, (generated, wanted)
    report = {"mode": "stock transformers Qwen3_5ForCausalLM, standalone compiled RISC-V CPU, bf16 CPU",
              "parameters": parameters, "generated_tokens": len(generated), "tokens": generated,
              "output": firmware.output_text(generated), "mismatches": 0,
              "elapsed_seconds_one_run": elapsed,
              "timed_work": "PAD prefill, state injection, and stock greedy generate; excludes model compilation",
              "torch": torch.__version__, "transformers": __import__("transformers").__version__}
    from chipformer.q35.ir import Head, HD
    report['upstream_revision'] = 'ca13310e2f9f788d1dd3feeea8213c1b0c23913c'
    report['ram_bytes'] = spec.RAM_WORDS * 4
    report['ram_key_heads'] = sum(k.startswith('ram') for k in circuit.tables)
    report['chip_layers'] = count
    report['hidden_size'] = layout.hidden
    report['logical_heads'] = sum(isinstance(n, Head) for n in circuit.nodes)
    report['logical_value_heads'] = sum(len(n.vheads()) for n in circuit.nodes if isinstance(n, Head))
    report['logical_recurrent_fp32_bytes'] = report['logical_value_heads'] * HD * HD * 4
    report['allocated_recurrent_fp32_bytes'] = count * circuit.target.value_heads * HD * HD * 4
    report['addresses_written'] = [first, last]
    report['alias_sentinels_read'] = {hex(0): 88, hex(sentinel): 89}
    assert report['output'] == 'ABXY'
    report['ram_ranges_exercised'] = 'upper half first word and last aligned word, with lower half alias sentinels'
    report['is_doom'] = False
    report['standalone_only'] = True
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2))
    (output / "patch-manifest.json").write_text(json.dumps(manifest, indent=2))
    (output / "program.u32").write_bytes(b''.join(w.to_bytes(4, 'little') for w in program.text))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
