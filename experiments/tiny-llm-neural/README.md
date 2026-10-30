# First tiny-LLM predictions on the compiled Qwen CPU

The RTX 5070 Ti generated all **16,213,774 trace tokens with zero
mismatches**, from ELF boot through HALT, using the verified stacked ROM and
feedback executor. It emitted `Once upon a time,`. This is the fp32 compact
executor for the compiled Qwen3.5-layer RISC-V circuit. The separate stock-HF
result below establishes a bounded prefix through actual Qwen3_5ForCausalLM
layers. No pretrained language-model weights are loaded.

## Neural result

The prompt is `Once`, encoded as `[1, 403]`. The workload executes
five inference positions, one prefill position and four generated predictions.
Prediction IDs are `[403, 407, 261, 378, 432]`. The generated continuation IDs are
`[407, 261, 378, 432]`. The guest stops after its fixed five-position
budget. This comma is not an EOS or a completed story.

The observer decodes BYTE records from the actual GPU token file. It does not
copy reference text. The guest's diagnostic records also contain position,
input ID, argmax ID, Q12 logits hash, and KV clipping count. The full actual
trace matches the independently generated Mac trace byte for byte. All
65,536 final recurrent RAM bytes match the Mac reference. The independent
comparison is [validation.json](evidence/validation.json).

Generation ran at **3,844.25 trace tokens/s** in this one
retained execution. Successful generation chunks total
70.29 minutes, or 14.06
minutes per inference position including prefill. The elapsed wall interval
from the first successful prefix worker through the completed full report is
72.63 minutes. It includes
setup, checkpointing, validation, observation, IO, and hand-off gaps.
Successful worker process durations sum to
70.83 minutes. These are distinct
from the generation-only timer.

| Position | Phase | RV32 instructions | Trace tokens | Measured generation minutes, bounded | Phase 0 projection at 4,380/s |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | prefill | 2,122,366 | 3,193,567 | 13.89 to 13.96 | 12.15 |
| 1 | decode | 2,136,726 | 3,218,248 | 14.01 to 14.09 | 12.25 |
| 2 | decode | 2,150,990 | 3,249,380 | 14.16 to 14.23 | 12.36 |
| 3 | decode | 2,164,866 | 3,262,735 | 14.19 to 14.26 | 12.42 |
| 4 | decode | 2,179,478 | 3,286,637 | 13.86 to 13.90 | 12.51 |

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
The final actual-state checkpoint is at 16,213,774. See
[checkpoint.json](evidence/checkpoint.json) and [gpu.json](evidence/gpu.json).

The first SSH-owned Start-Process launch exited without generating tokens.
The accepted detached launch uses a WMI-created PowerShell parent that waits
for Start-Process with redirected stdout and stderr. It survives SSH disconnect.
The compact worker has a four-hour watchdog and an 8 GiB allocator cap.
Peak PyTorch allocation was 231,865,856 bytes.

The original PowerShell process wrapper recorded a null HF exit code. The
completed Python report, runtime stderr, and independent token comparison are
the result evidence. The launcher now retains Process.Handle before waiting.
Separate exit-0 and exit-7 controls pass in
[launcher-exit.json](evidence/launcher-exit.json). No historical exit code is
inferred from those controls.

## Stock Hugging Face prefix

The same ELF and memory map generated **32,675 self-fed
prefix tokens with zero mismatches** through stock Qwen3_5ForCausalLM forward
and greedy generation on CUDA. The timed path took
25.00 minutes at 21.78 tokens/s,
including PAD prefill, boot-state injection, online comparison, and progress
writes. Compilation is excluded. Process wall time was
25.66 minutes. The independent Mac prefix comparison
also passes. See [hf.json](evidence/hf.json).

The standalone model has 6,212,045,824 parameters and uses compiled fixed
BF16 weights. It has 44 circuit layers plus the required zero full-attention
layer. It does not use pretrained weights. Original emitter arithmetic is
unchanged. A generator assigns one layer's BF16 dictionary at a time to avoid
retaining every FP32 layer on the host. First, middle, and last layer words
match the original emitter exactly in [emitter.json](evidence/emitter.json).

The stock process ran after the main circuit run. The user authorized its
separate 14 GiB allocator cap. Before loading,
CUDA and nvidia-smi both had to show enough space for the parameters, a
512 MiB working allowance, and at least 1 GiB headroom. Peak allocation was
12,962,223,104 bytes and peak reserved memory was
13,404,995,584 bytes. The minimum sampled free VRAM was
1,815,085,056 bytes, sampled with CUDA every second and nvidia-smi every five seconds. The
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
3,193,725. This prefix does not reach it or establish
a whole guest stories260K prediction through stock HF. The complete
token-generation proof above uses the compact executor.

## Reproduce

Use fresh private cache and output directories for another run. Do not reuse a
checkpoint from a different ELF, trace, or worker. From the Mac repository root,
build the original Phase 0 workload and stream its independent trace.

```sh
python3 experiments/tiny-llm-rv32/build.py \
  --out .cache/tiny-llm/neural-stories --prompt Once --steps 5 --stack-bytes 8192
python3 experiments/tiny-llm-neural/reference.py \
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


<!-- AUDIT_TIMING_CORRECTION_BEGIN -->
## 2026-10-31 timing correction after the audit

The historical approximately 3.4× ratio mixes an original-unfused projection with a shorter measured run. The approximately 5.3× ratio compares projections. The 1.86× historical ratio changes both network and executor. These are not measured same-network accelerator gains. Earlier projection tables retain their stated scope and are not measurements of complete scalar execution.

One complete run per binary, from boot on the same unified W=4 compiled Qwen3.5-architecture network, with identical model data, prompt `[1, 403]`, a five-position budget and compact fused executor. Every token matches each binary's own independent reference, as do final RAM and all register copies. Both prediction-ID sequences are `[403, 407, 261, 378, 432]`. The shared stock-emitted BF16 weight identity is `0e515bbb…`; compact evaluation is fp32, bound to the same constructor. Stock `Qwen3_5ForCausalLM.generate` coverage remains bounded prefixes only, separate from these full runs. The scalar worker initially stopped after generation because its older reference JSON omitted final registers. A fresh local replay reproduced its entire reference trace and RAM literally; validation then restored the terminal actual GPU checkpoint without new generation. The error, repair and validation-only restart are retained in the evidence.

| Build | Generation seconds | CPU trace tokens | Trace tokens/s | Min/inference position |
| --- | ---: | ---: | ---: | ---: |
| Scalar | 4,361.42 | 16,213,774 | 3,717.54 | 14.5381 |
| NDOT4 | 1,446.63 | 5,423,298 | 3,748.93 | 4.8221 |

Scalar / NDOT4 generation time is **3.0149×**, one complete execution each. This is an accelerator comparison with the same fused executor on both sides. It does not apply the shorter fixture's ratio to the 38-position story. Generation timing excludes setup, comparisons, IO, checkpoint/restart gaps. The audit report is unchanged.

Evidence: [controlled measurement](../checkpoint-integrity/evidence/controlled-measurement.json), independently checked by [verify.py](../checkpoint-integrity/verify.py).
<!-- AUDIT_TIMING_CORRECTION_END -->
