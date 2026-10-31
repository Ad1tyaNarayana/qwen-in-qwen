# The tiny Qwen3.5 inside the Qwen3.5

This directory holds the language model that runs *inside* the compiled network, and the program that runs it.

- **The model:** an 8-layer, 1,510,800-parameter Qwen3.5 (`Qwen3_5ForCausalLM`), trained from scratch on TinyStories. It uses Gated DeltaNet layers, with full attention at layers 3 and 7, the same hybrid layout as the 65-layer network it runs in. The weights, tokenizer and training log are on Hugging Face as [`Ayt1da/qwen-in-qwen-tiny`](https://huggingface.co/Ayt1da/qwen-in-qwen-tiny).
- **The program:** a model-independent integer LLM runtime for RV32IM plus the `ndot4` accelerator. Model geometry, layer types, int8 weights, lookup tables, tokenizer bytes, cache offsets, prompt tokens and the position budget all live in a data image (DROM). `runtime.c` has no model-specific generated header. One compiled object and one native executable serve every model image, llama2.c and Qwen3.5 alike. Linking an ELF embeds a different DROM image; executable bytes stay identical.
- **The result:** given the prompt `Once`, the compiled Qwen3.5-architecture CPU network generated all 35,836,005 trace tokens of this program with zero mismatches. The guest wrote ` upon a time, there`. All 65,536 RAM bytes and all three register copies match the independent reference.

| | Outer network | This model |
| --- | --- | --- |
| Layers | 65: 64 DeltaNet + 1 full attention | 8: DeltaNet ×6, full attention at layers 3 and 7 |
| Weights | compiled, never trained | trained, 60,000 steps |
| Weights live in | the network | the CPU's RAM, as a 1,674,224-byte DROM image |

The sections below are generated from the evidence files and record each completed gate. A random model is never accepted as the trained model, and a failed gate stops the pipeline.

<!-- MEASURED_RESULTS_BEGIN -->
## Measured results

Parts A through D pass. One native executable and one RV32 object serve both model families.
All 579 native legacy positions match the unchanged old engine in IDs, logit hashes, clipping counts and literal output bytes.
The independent RV32 reference passes all 64 stories260K and all 129 custom-model positions.
Executable bytes match the trained Qwen image and the second, untied Qwen geometry.

The eight-layer model has 1,510,800 parameters and a 1,674,224-byte DROM image.
Training completes 60,000 steps in 43.49 minutes in the isolated training venv.
The final floating validation estimate is 1.53958 nats per token over 20,480 targets. It is not an integer held-out-loss result.

The selected packing uses midpoint nonlinear LUTs and covariance-guided int8 weight rounding.
The three predefined tuning prompts match float HF greedy generation in 167/167 generated predictions.
Across all six inspected prompts, agreement is 291/327 predictions, or 89.0 percent.
Five continuations match completely. The rabbit prompt first differs at position 28, with a float top-two logit margin of 0.00874.
The six prompts have been used during quantization investigation and are not an unseen quality benchmark.
All cache and numeric clipping counters stay zero.

The six-position guest retires 25,121,038 instructions and emits 35,836,005 trace tokens.
Its 2,257,920 NDOT4 instructions account for every aligned matrix MAC.
Observed stack use is 368 bytes against the 1,024-byte reservation.
The supplied 3,735 trace tokens/s rate projects to 26.65 minutes per LLM position.
Including its one prefill position, the whole five-token continuation projects to 159.91 minutes, or 31.98 minutes per new token.
This projection includes the complete guest workload and is not a new GPU timing.
Table rows measure position begin/end markers; whole-program totals also include startup and diagnostic output. Costs describe these six positions.

| Position | Phase | RV32 instructions | Trace tokens | Projected minutes |
| --- | --- | ---: | ---: | ---: |
| 0 | prefill | 4,152,984 | 5,898,082 | 26.32 |
| 1 | decode | 4,170,062 | 5,952,285 | 26.56 |
| 2 | decode | 4,181,210 | 5,971,258 | 26.65 |
| 3 | decode | 4,192,102 | 5,987,282 | 26.72 |
| 4 | decode | 4,205,803 | 6,006,144 | 26.80 |
| 5 | decode | 4,215,968 | 6,016,945 | 26.85 |

The neural CPU generates all 35,836,005 trace tokens with zero mismatches.
Actual GPU BYTE records produce `Once upon a time, there`.
All 65,536 RAM bytes and all three register copies match the Mac reference exactly.
Retained generation calls measure 27.82 minutes per inference position.
Including prefill, the five new tokens take 166.90 retained generation minutes, or 33.38 minutes per new token.
The timer excludes setup, comparisons, checkpoint IO, shared-lock waits, shutdown time and replayed work after the last pre-shutdown checkpoint.
The full-map stock HF prefix contains 16,384 tokens with zero mismatches.
Minimum sampled free VRAM is 976 MiB under the 15 GiB cap and 512 MiB guard.
The stock prefix is a bounded low-level CPU trace and does not complete a guest language-model prediction.
The PC shut down after 16,359,680 generated trace tokens.
The run resumed from its hash-checked actual GPU checkpoint at 16,318,720, replaying 40,960 tokens.
The checkpoint archive, binding and pre-shutdown actual trace pass independent Mac checks.

Detailed bindings and checks are in [mac-gate.json](evidence/mac-gate.json),
[trained-quality.json](evidence/trained-quality.json), [covariance-quality.json](evidence/covariance-quality.json),
and [summary.json](evidence/summary.json).
Final acceptance and artifact bindings are in [final-checks.json](evidence/final-checks.json) and [neural-validation.json](evidence/neural-validation.json).
<!-- MEASURED_RESULTS_END -->

## Runtime and graph

`format.py` defines a little-endian, versioned model header and a list of layer
records. The C layouts have static size assertions. Layers carry their own
attention type, head geometry, MLP width, norm locations, matrices and scales,
RoPE table and cache offsets. The residual width and vocabulary belong to the
model. `pack.py` supports llama2.c checkpoints and saved HF
`Qwen3_5ForCausalLM` directories. Unsupported dimensions or numeric formats fail
before a model image is emitted.

The supported Qwen path has gated DeltaNet, short depthwise causal convolution,
L2-normalized q/k, repeated linear key heads, decayed recurrent delta updates,
full GQA with q/k RMSNorm, partial half-split RoPE, query-projection output gates,
SwiGLU, zero-centered RMSNorm, and tied or untied embeddings. Plain text uses
identical positions on the three mRoPE axes. Attention bias, vision inputs and
non-default RoPE scaling are outside this packer's profile.

The software uses the existing `ndot.h` intrinsic for aligned rows and scalar
RV32IM for unaligned rows or tails. It uses no new ISA instruction, memory-map
change, floating-point operation, host-computed inference value or changed
CPU/accelerator weight. Host input is the initial prompt's token IDs. The guest
computes every subsequent token and emits bytes through the existing ecall.

## Numeric formats and RAM

[NUMERICS.md](NUMERICS.md) describes scales, LUT grids, saturation and bounds.
The legacy path preserves the original integer-engine arithmetic. Qwen uses
Q12 activations and convolution history, Q15 normalized q/k, and Q20 DeltaNet
values and states. Qwen decay uses a finer exponential LUT than the legacy path.
All clipping is counted. Float agreement is an empirical quality measurement,
not a neural correctness tolerance.

`train_config.json` specifies an eight-layer, hidden-128, MLP-344 Qwen3.5 model
with the standard three-linear/one-full pattern. Full attention has four query
heads, one KV head and head width 32. Linear attention has two key heads and
four value heads, with key and value widths 16 and convolution width four.
RoPE rotates eight dimensions per full-attention head. Embeddings are tied.
The model has 1,510,800 parameters, including 1,505,280 matrix parameters.
The tokenizer is the existing custom TinyStories vocabulary of 1024 tokens.
Training uses 128-token sequences. The runtime context is 64.

At context 64, the model uses 24,576 bytes for DeltaNet states, 9,216 for
convolution history, 16,384 for full-attention KV, and 11,408 for scratch.
The total arena use is 61,584 bytes. The fixed runtime reserves 63,136 static
bytes, 256 bottom bytes and a 1,024-byte stack, leaving a linked margin of
1,120 bytes. The fixed arena also accommodates the existing custom llama
model at context 129. The ELF linker checks RAM and ROM limits independently.
A context-71 Qwen image fails the arena gate. No larger memory map is requested.

## Verification gates

`verify.py` compares the unchanged old engine with the new native executable
across three stories260K prompts at 64 positions and three custom-model prompts
at 129 positions. All 579 positions match in IDs, logit hashes, clipping counts
and literal output bytes. It also checks identical executable bytes across
model images and runs ASan and UBSan. Both complete legacy software-reference runs pass after the numeric bound
corrections. The trained Qwen guest also matches its native build exactly.
The evidence directory records current results, not promises of later gates.

An additional random Qwen fixture uses hidden width 96, four layers, untied
embeddings, six full-attention query heads, two KV heads, linear head widths
8 and 12, three linear value heads, kernel width three, epsilon `1e-5`, and a
half-width rotary prefix. All six software-reference positions match the native
runtime. Its executable bytes match both llama models and the tied Qwen fixture.
This fixture checks runtime generality. It is not a language-quality result.

`operators.py` calls the actual C attention functions and compares their values
with the installed Transformers 5.21 implementation. The rejected Q12 state
probe is retained. It exposed large errors for valid small DeltaNet states and
led to the Q20 state format. Operator errors and float greedy agreement remain
separate from exact software and neural validation.

`quality.py` uses ordinary unmodified HF greedy `generate` with full prompt
prefill and a fixed position budget, with EOS stopping disabled to match the
guest. It compares three free-running prompts. `gate.py` requires a real,
hash-bound completed training export, native legacy parity, complete legacy
reference runs, at least 90 percent measured float greedy agreement, zero
clipping, identical executable bytes, linked RAM fit and exact trained-model
RV32/native output before it permits GPU execution.

`launch_gpu.py` checks the accepted fusion source hashes and uses a private
source mirror. It detaches a prefix, checkpoint restore, full neural execution,
and 16,384-token stock prefix sequentially through WMI. Each stage acquires the
shared atomic lock file on the GPU host, writes its owner and expected duration,
and checks for other Python jobs. A `finally` block releases that stage's lock
on success or failure. The full neural worker has a 235-minute watchdog.
Lock waits do not hold the GPU. The stock worker retains the 15 GiB cap, 512 MiB CUDA and
physical-free-VRAM guard, no offload, and unmodified HF generation.
`verify_gpu.py` checks the exported actual trace, all 65,536 RAM bytes, all three
register copies, actual BYTE output, checkpoint provenance, restore boundaries,
chunk hashes, and stock trace against the independent Mac reference.
Only the ELF boot state enters the neural machine. Reference values supply
comparisons and the stopping budget, never post-boot state.

The GPU PC shut down during the full run after 16,359,680 trace tokens. After
the orchestrator restored access, tasklist showed no Python jobs and the
original lock still belonged to this task. The complete on-disk trace matched
the independent reference. The atomic checkpoint loaded on the Mac with a
matching ELF/source binding and SHA-256
`cae4b92b719168d48a9f5c7b6957462505e51b929bbb5c1e273780b0d8a44773`.
`resume_gpu.py` preserved the original checkpoint and trace, then restored the
actual GPU state at token 16,318,720 and replayed 40,960 tokens. It regenerated
scratch code and trimmed only the chunk journal past that saved boundary.
The first restart rejected the existing scratch package before checkpoint
restoration; its error is retained. That ended hold was released and a new
hold was acquired atomically before the corrected restart. Neither attempt
used reference state. The shutdown and startup error are recorded separately
from neural mismatches in [recovery-checkpoint.json](evidence/recovery-checkpoint.json)
and [recovery-startup-failure.json](evidence/recovery-startup-failure.json).
Final timing excludes the outage and the discarded post-checkpoint generation
calls, in addition to the usual setup, I/O and lock waits.

## Reproduction

### Use the model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("Ayt1da/qwen-in-qwen-tiny")
model = AutoModelForCausalLM.from_pretrained("Ayt1da/qwen-in-qwen-tiny")
out = model.generate(tok("Once", return_tensors="pt").input_ids, max_new_tokens=5, do_sample=False)
print(out[0].tolist())  # [1, 402, 407, 261, 378, 941, 383]: Once upon a time, there
```

The float checkpoint gives the same greedy tokens as the in-CPU integer run for this prompt. The Hugging Face repository also has `in-cpu/drom-image.bin`, the exact 1,674,224-byte image the guest read (SHA-256 `95932990…d0c1365`). Its `model.safetensors` is the checkpoint bound by [calibration.json](evidence/calibration.json) (SHA-256 `b7929b5d…f87e626c`).

### Check the retained run

The [evidence bundle](https://github.com/Ad1tyaNarayana/qwen-in-qwen/releases/tag/evidence-v1) contains this run's complete actual and reference traces, final RAM and registers, output bytes, chunk journals, the program ELF and the pre-shutdown CPU checkpoint. From the repository root:

```sh
python tools/verify_all.py --tier 1 \
  --bundle https://github.com/Ad1tyaNarayana/qwen-in-qwen/releases/download/evidence-v1/qwenputer-evidence-v1.tar.gz
```

### Rebuild from retained artifacts

These commands need the retained working cache in `.cache/tiny-qwen35/`, which is not distributed. Large images, binaries,
traces, checkpoints and copied packages stay there.
The selected HF checkpoint is `training/hf/`; the selected DROM image, ELF and
independent reference are in `trained-program/`. The fixed native executable
and RV32 object are in `final-safe/runtime/`. Collected GPU artifacts are in
`actual-covariance-20261030/`, with the archived actual pre-shutdown checkpoint
under `pre-shutdown/`. These paths are relative to that cache directory.

```sh
.venv/bin/python experiments/tiny-qwen35/verify.py --out .cache/tiny-qwen35/final-safe
.venv/bin/python experiments/tiny-qwen35/reference.py --build .cache/tiny-qwen35/final-safe/stories-0
.venv/bin/python experiments/tiny-qwen35/reference.py --build .cache/tiny-qwen35/final-safe/custom-0
```

Training used `train.py` and `train_config.json` on the TinyStories shards and
1,024-token tokenizer prepared by [tiny-llm-train](../tiny-llm-train/README.md).
The loop has a 3,500-second cutoff and a 60,000-step cap. Shard zero is
validation; shards one through 49 supply training batches. The original run
used a separate Windows GPU host with its own isolated virtual environment,
driven over SSH by `launch_train.py`. That script refuses to overwrite its
result directory or to run while another Python process is present.

The selected weight rounding uses CPU calibration on the first 64 retained
held-out TinyStories excerpts, truncated to 64 tokens each. Only the resulting
int8 weights enter DROM. [calibration.json](evidence/calibration.json) binds the
checkpoint and excerpt file. To rebuild this calibration from the retained
checkpoint and tokenizer export:

```sh
.venv/bin/python experiments/tiny-qwen35/calibrate.py \
  --checkpoint .cache/tiny-qwen35/training/hf \
  --tokenizer .cache/tiny-qwen35/trained-program/tokenizer.json \
  --stories .cache/tiny-llm/custom/heldout_stories.json \
  --out .cache/tiny-qwen35/calibration.pt
```

The full pipeline runs the Mac gates, the neural GPU run, and independent artifact validation. Run only one copy. Reuse of an existing GPU artifact label is rejected.

```sh
.venv/bin/python experiments/tiny-qwen35/finish.py --label FRESH --quantization midpoint+covariance \
  --calibration .cache/tiny-qwen35/calibration.pt
```

Use `--mac-only` to stop after A through C. Inspect the retained progress and
logs before resuming a partial run. An existing failed numerical result is not
silently retried. Per-position instruction and trace counts come from actual
reference execution. Dividing trace tokens by 3,735 tokens/s gives a projection,
not a measured neural duration for this guest.

The append-only decision trail is [decisions.tsv](decisions.tsv).
