"""Write the report only from completed, independently verified evidence."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
v = json.loads((HERE / 'evidence/validation.json').read_text())
g = json.loads((HERE / 'evidence/gpu.json').read_text())
h = json.loads((HERE / 'evidence/hf.json').read_text())
c = json.loads((HERE / 'evidence/checkpoint.json').read_text())
assert g['complete'] and v['mismatches'] == h['mismatches'] == 0
assert c['checkpoint_offset'] == v['tokens_checked']
rows = []
for p in v['positions']:
    low,high = p['measured_compute_minutes_bracket']
    rows.append(f"| {p['position']} | {p['phase']} | {p['instructions']:,} | {p['trace_tokens']:,} | {low:.2f} to {high:.2f} | {p['phase0_projected_minutes']:.2f} |")
text = f'''# First tiny-LLM predictions on the compiled Qwen CPU

The RTX 5070 Ti generated all **{v['tokens_checked']:,} trace tokens with zero
mismatches**, from ELF boot through HALT, using the verified stacked ROM and
feedback executor. It emitted `{v['generated_text']}`. This is the fp32 compact
executor for the compiled Qwen3.5-layer RISC-V circuit. The separate stock-HF
result below establishes a bounded prefix through actual Qwen3_5ForCausalLM
layers. No pretrained language-model weights are loaded.

## Neural result

The prompt is `Once`, encoded as `{v['prompt_tokens']}`. The workload executes
five inference positions, one prefill position and four generated predictions.
Prediction IDs are `{v['prediction_ids']}`. The generated continuation IDs are
`{v['generated_prediction_ids']}`. The guest stops after its fixed five-position
budget. This comma is not an EOS or a completed story.

The observer decodes BYTE records from the actual GPU token file. It does not
copy reference text. The guest's diagnostic records also contain position,
input ID, argmax ID, Q12 logits hash, and KV clipping count. The full actual
trace matches the independently generated Mac trace byte for byte. All
65,536 final recurrent RAM bytes match the Mac reference. The independent
comparison is [validation.json](evidence/validation.json).

Generation ran at **{g['tokens_per_second']:,.2f} trace tokens/s** in this one
retained execution. Successful generation chunks total
{g['compute_seconds']/60:.2f} minutes, or {v['mean_minutes_per_inference_position']:.2f}
minutes per inference position including prefill. The elapsed wall interval
from the first successful prefix worker through the completed full report is
{c['elapsed_wall_seconds_including_restart_gaps']/60:.2f} minutes. It includes
setup, checkpointing, validation, observation, IO, and hand-off gaps.
Successful worker process durations sum to
{c['sum_successful_process_wall_seconds']/60:.2f} minutes. These are distinct
from the generation-only timer.

| Position | Phase | RV32 instructions | Trace tokens | Measured generation minutes, bounded | Phase 0 projection at 4,380/s |
| --- | --- | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

The measured bounds use cumulative GPU-generation timers immediately before
and after each marker. A marker inside an 8,192-token chunk is bracketed,
without assuming a uniform within-chunk rate. Position intervals exclude
text decoding and diagnostic output. The original Phase 0 projection assumed
that the earlier Doom executor rate would transfer to this workload. These
results measure this workload instead. There is no five-run benchmark or
causal explanation of the rate difference.

## Circuit, state, and resume

The pinned upstream is `ca13310e2f9f788d1dd3feeea8213c1b0c23913c`.
The CPU retains 64 KiB RAM at `0x01000000`, 4 MiB data ROM at `0x02000000`,
and 256 KiB code ROM at `0x00040000`. The existing frozen memory-map preparation
supports the Phase 0 ELF unchanged. No instruction, circuit, weight, head,
or upstream source changes are made. The standalone FreeRun embedding adapter
omits the merged-chat-only Q family, as in the verified DMC runtime.

The stacked executor uses the accepted kernels from
[dmc-rom-fusion](../dmc-rom-fusion/README.md) and
[dmc-feedback-fusion](../dmc-feedback-fusion/README.md). Their current hashes
match the prior direct-word acceptance report. That acceptance covers two
65,536-token windows and is not a universal state-equivalence claim. This
experiment checks every token in this full workload and all final RAM bytes.
It does not compare all latent state words after every LLM trace token.

Only the ELF initial state and GO are loaded at boot. Every subsequent token
and its next embedding come from the GPU circuit. The reference supplies
comparison data and a stopping budget. It supplies no intermediate registers,
RAM, KV cache, logits, predictions, or recurrent-state snapshots.

Each checkpoint stores actual GPU S tensors, every lag history, the next
embedding, last generated token, trace offset, and cumulative timer. Cleared
scratch buffers are checked before saving. Atomic replacements include bounded
Windows sharing retries. On restore, the retained actual token prefix is
checked and any uncheckpointed suffix is truncated. Source and fixture hashes
must match the checkpoint binding.

A 65,536-token prefix passed from explicit zero histories. Restoring that actual
checkpoint generated 256 more exact tokens. The full worker then resumed at
65,792. It checkpoints every 262,144 tokens and writes a heartbeat every chunk.
The final actual-state checkpoint is at {c['checkpoint_offset']:,}. See
[checkpoint.json](evidence/checkpoint.json) and [gpu.json](evidence/gpu.json).

The first SSH-owned Start-Process launch exited without generating tokens.
The accepted detached launch uses a WMI-created PowerShell parent that waits
for Start-Process with redirected stdout and stderr. It survives SSH disconnect.
The compact worker has a four-hour watchdog and an 8 GiB allocator cap.
Peak PyTorch allocation was {g['peak_allocated_bytes']:,} bytes.

The original PowerShell process wrapper recorded a null HF exit code. The
completed Python report, runtime stderr, and independent token comparison are
the result evidence. The launcher now retains Process.Handle before waiting.
Separate exit-0 and exit-7 controls pass in
[launcher-exit.json](evidence/launcher-exit.json). No historical exit code is
inferred from those controls.

## Stock Hugging Face prefix

The same ELF and memory map generated **{h['generated_tokens']:,} self-fed
prefix tokens with zero mismatches** through stock Qwen3_5ForCausalLM forward
and greedy generation on CUDA. The timed path took
{h['elapsed_seconds_one_run']/60:.2f} minutes at {h['tokens_per_second']:.2f} tokens/s,
including PAD prefill, boot-state injection, online comparison, and progress
writes. Compilation is excluded. Process wall time was
{h['process_wall_seconds']/60:.2f} minutes. The independent Mac prefix comparison
also passes. See [hf.json](evidence/hf.json).

The standalone model has {h['parameters']:,} parameters and uses compiled fixed
BF16 weights. It has 44 circuit layers plus the required zero full-attention
layer. It does not use pretrained weights. Original emitter arithmetic is
unchanged. A generator assigns one layer's BF16 dictionary at a time to avoid
retaining every FP32 layer on the host. First, middle, and last layer words
match the original emitter exactly in [emitter.json](evidence/emitter.json).

The stock process ran after the main circuit run. The user authorized its
separate {h['allocator_cap_bytes']/2**30:.0f} GiB allocator cap. Before loading,
CUDA and nvidia-smi both had to show enough space for the parameters, a
512 MiB working allowance, and at least 1 GiB headroom. Peak allocation was
{h['cuda_peak_allocated_bytes']:,} bytes and peak reserved memory was
{h['cuda_peak_reserved_bytes']:,} bytes. The minimum sampled free VRAM was
{h['minimum_free_vram_bytes']:,} bytes, sampled with CUDA every second and nvidia-smi every five seconds. The
active guard stops generation below 1.25 GiB. Unused allocator cache is released
every eight generated tokens. No CPU/GPU layer offload was used. BF16 reduced-precision reductions and TF32 are disabled. Installed Transformers
uses its stock PyTorch convolution and DeltaNet fallbacks. Optional kernel
packages were not installed.

The first stock attempt checked 14,544 exact tokens, then was stopped after
nvidia-smi showed 927 MiB free. It is not accepted for the requested headroom
requirement. The retained prefix was independently checked in
[hf-attempt1.json](evidence/hf-attempt1.json). The reported result above is a
fresh run from boot with active memory control. No state from the interrupted
run is supplied to it.

The prefix starts at ELF boot and enters the beginning of guest transformer
inference. The first guest position's end marker is trace token
{v['positions'][0]['end_token']:,}. This prefix does not reach it or establish
a whole guest stories260K prediction through stock HF. The complete
token-generation proof above uses the compact executor.

## Reproduce

Use fresh private cache and output directories for another run. Do not reuse a
checkpoint from a different ELF, trace, or worker. From the Mac repository root,
build the original Phase 0 workload and stream its independent trace.

```sh
python3 experiments/tiny-llm-rv32/build.py \\
  --out .cache/tiny-llm/neural-stories --prompt Once --steps 5 --stack-bytes 8192
python3 experiments/tiny-llm-neural/reference.py \\
  --build .cache/tiny-llm/neural-stories
```

The current build scripts reproduce all original loadable code, RAM, and DROM
bytes. ELF metadata differs. [reproduction.json](evidence/reproduction.json)
compares every PT_LOAD byte, address, zero-fill extent, and permission.

Sync the experiment scripts, frozen memory-map prepare helper, accepted fusion
modules, state helper, ELF, and reference trace/report using tar over `ssh gpu`.
The launcher uses the approved PC workspace
`C:/Users/Admin/qwenputer-games-work` and its existing `.venv`.

```sh
python3 experiments/tiny-llm-neural/launch.py --stop-after 65536 --label prefix
```

After completion, retain `report.json` as `prefix-report.json`. Resume with
`--stop-after 65792 --label resume`, then retain that report as
`resume-report.json`. Start `launch.py --label full`. The worker's `report.json`
and `heartbeat.json` are under `artifacts/tiny-llm-neural` on the PC.
`launch.py --hf-after-main --label hf` waits for a successful full report before
starting the separately bounded stock check. Do not start another GPU workload.

After completion, run `export_gpu.py` on the PC. Copy the actual trace, actual
output, actual RAM, chunks.jsonl, reports, and export.json into a private Mac
cache directory. Run `verify.py --actual DIRECTORY` on the Mac. Copy the stock
report privately and run `verify_hf.py --report REPORT`. Preserve export.json as
`evidence/checkpoint.json`, then run `report.py` to write this report.

Full traces, model/ELF images, copied packages, RAM, and tensor checkpoints stay
under `.cache/` or private PC artifacts. Only scripts and textual evidence are
included here. No commits, installations, or upstream edits were made. Custom
model fit, quality, and cost appear in
[the engine report](../tiny-llm-rv32/README.md#custom-model-integration).

The Model the Domain principle kept checkpoints bound to their fixture and
kernel hashes. Prove It Works required independent comparison of the actual
token file, emitted bytes, and recurrent RAM. Work ran inline under AGENTS.md.
No independent agent review is claimed.
'''
(HERE / 'README.md').write_text(text)
