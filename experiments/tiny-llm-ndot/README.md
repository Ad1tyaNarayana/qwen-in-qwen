# Stories260K with NDOT4

## 2026-10-30 Gate 2 stops at exact final state

The shared full-map W=4 circuit passes
[Gate 1](../neural-accelerator/evidence-m5/gate1.json). It uses the same generated
graph and emitted BF16 weights as the passing stock M3/M4 build. The adapted
unfused Executor2 passes both 8,192-token stock-prefix comparisons and both
262,144-token legacy regressions. ROM and feedback fusion remain disabled.

[build.py](build.py) derives the guest source from the original integer engine.
It replaces only the int8-by-int16 matrix-dot loop, aligns the int16 scratch vector
to eight bytes, and uses the existing [ndot.h](../neural-accelerator/ndot.h)
NDOT4 intrinsic. The loop processes four lanes at a time. Unaligned rows and
remaining lanes use scalar cleanup. Attention, row rescaling, quantization,
weights, tokenization, and the physical memory map are unchanged.

[native64.json](evidence/native64.json) passes all 64 native positions for the
original Phase 0 prompt `Once upon a time`. Predicted IDs, every-position logit
hashes, decoded bytes, and KV clip counts agree exactly. Both packed model files
are byte-identical. KV clipping is zero. The linker retains 46,272 static bytes,
an 8,192-byte stack reservation, and 10,816 margin bytes.

[verify.py](verify.py) runs the original scalar software Machine and the extended
ndot Machine independently. It compares their output bytes with both native
builds, checks the first 1,024 dots against the raw-byte oracle, measures retired
instructions and serialized trace tokens, and compares final registers, PC, and
all RAM words literally. Emulator wall time is not a neural timing result.

[software64.json](evidence/software64.json) records both completed software runs.
Both halt cleanly and exactly match the complete native output. All 64 predicted
IDs, every-position logit hashes, decoded bytes, and KV clip counts agree.
The first 1,024 dots agree with the independent raw-byte oracle.

[gate2.json](evidence/gate2.json) fails the literal final architectural-state
comparison. No state coordinates are masked, normalized, or compared with a
tolerance. The common final PC is `0x40018`; both exit zero without a fault.
Registers x28 through x31 and exactly three RAM words differ.

| Register | Original | NDOT4 |
| --- | ---: | ---: |
| x28 | 4294912529 | 4294967286 |
| x29 | 16820820 | 33587232 |
| x30 | 4294912000 | 4294912529 |
| x31 | 105648 | 16820820 |

| RAM address | Original word | NDOT4 word |
| --- | --- | --- |
| `0x0100ff18` | `0xffffffff` | `0x0000003f` |
| `0x0100ff1c` | `0x0004138c` | `0x0100ab5c` |
| `0x0100ff2c` | `0x00040fb0` | `0x00041010` |

Both ELFs have `__scratch_end=0x0100b5c0` and `__stack_top=0x01010000`.
All three differing RAM words are within the reserved stack region. Every RAM
word below that region agrees. Compiler register allocation and changed code
addresses are plausible explanations for these differences. Their location does
not waive the requested architectural equality. The failed assertion and all
exact values remain in the evidence. No cleanup or canonicalization was added.

The deterministic workload counts below include actual refresh and variable-width
records. Total counts include diagnostic output. Position counts start at the
begin marker and exclude decoding and diagnostic output.

| Scope | Original instructions | NDOT4 instructions | Original trace tokens | NDOT4 trace tokens |
| --- | ---: | ---: | ---: | ---: |
| All 64 positions, whole program | 162,208,013 | 72,062,221 | 242,609,345 | 104,621,475 |
| Mean inference position | 2,534,019.125 | 1,125,491.125 | 3,790,108.359375 | 1,634,047.890625 |
| Position 4, first generated prediction | 2,179,478 | 770,950 | 3,286,637 | 1,126,656 |
| Position 63 | 2,926,052 | 1,517,524 | 4,341,698 | 2,189,576 |

The whole program retires 90,145,792 fewer instructions, a 55.57% reduction,
and emits 137,987,870 fewer trace tokens, a 56.88% reduction. These are exact
software workload counts from one run, not GPU latency measurements. The full
per-position counts are in the software evidence.

## Neural run status

Work stopped at the failed Gate 2, as requested. The five-position `Once` M5 GPU
run was not started. No guest text from that neural run, checked-token count,
tokens/s, or minutes per prediction exists. No speed comparison with Phase 1's
14.06 minutes per prediction is claimed. The 16,384-token autonomous stock check
of the ndot stories program was also not started.

The open issue is the literal final register and stack-state difference between
the rebuilt program and the original ELF. Acceptance remains failed until that
requirement is met or explicitly revised. Output equivalence alone does not
pass it.

Private build directories are `.cache/tiny-llm/ndot-original64` and
`.cache/tiny-llm/ndot64`. The software report is
`.cache/tiny-llm/ndot-verify64/report.json`. Generated source, binaries, model blobs,
and RAM dumps stay private. All source changes remain uncommitted.


## Reproduce the failed software gate

Run from the repository root with fresh output paths.

```sh
.venv/bin/python experiments/tiny-llm-rv32/build.py \
  --out .cache/tiny-llm/ndot-original64-FRESH \
  --prompt 'Once upon a time' --steps 64 --stack-bytes 8192
.venv/bin/python experiments/tiny-llm-ndot/build.py \
  --out .cache/tiny-llm/ndot64-FRESH --prompt 'Once upon a time' --steps 64
.venv/bin/python experiments/tiny-llm-ndot/verify.py \
  --original .cache/tiny-llm/ndot-original64-FRESH \
  --ndot .cache/tiny-llm/ndot64-FRESH \
  --out .cache/tiny-llm/ndot-verify64-FRESH
```

The final command exits one at the final-state assertion. No later GPU stage
should start from that result. No commits, installations, upstream edits, or
changes to `experiments/doom-showcase/` or `docs/articles/` were made.

## 2026-10-30 Gate 2 correction and continuation

The orchestrator corrected the gate. Literal final-state equality between two
differently compiled binaries is ill-posed. Compilation can change caller-saved
temporaries and dead stack contents without changing program behavior. The old
failed literal comparison above remains historical evidence.

The revised software-equivalence gate requires all 64 predicted IDs,
every-position logit hashes, decoded bytes, and KV clip counts to agree exactly.
All RAM outside the reserved stack must agree byte for byte. Every remaining
stack difference must be below both final stack pointers. A differing word at or
above final SP fails the gate. Registers x28 through x31 are the RISC-V psABI
caller-saved t3 through t6. Their final values must be outside the exit/output ABI,
and every store or output reached by those final values must be accounted for.

The orchestrator further corrected the original phrase "no read after last
write". A store can legitimately use a temporary after its final write. The
required proof follows its final uses, records store sites and effective
addresses, and checks that the targets are covered by exact non-stack RAM,
compared output, or popped stack differences. HALT executes no later guest
instruction. ECALL uses a7 for the operation and a0 for its argument. Neither
x28, x29, x30, nor x31 is an exit argument. The standalone observer extracts
BYTE records and recurrent RAM; it does not treat t3 through t6 as output.

[gate2_liveness.py](gate2_liveness.py) replays both retained 64-position ELFs
without changing their architectural state. It records the final temporary
writes, register data/address dependencies, stores, effective addresses, and
ECALL dependencies in the final position. The replay must reproduce the
retained complete output, registers, RAM, and instruction counts exactly.
This is a proof for the executed workload, not universal compiler equivalence.

Neural faithfulness remains a separate, stricter gate. The five-position ndot
run must match its own independently generated trace at every token, and all
final RAM bytes and all three recurrent register copies must equal that
binary's Mac reference exactly. There is no dead-state exception in this check.

The fresh five-position fixture uses prompt `Once`, encoded as `[1, 403]`.
Its independent trace contains 5,423,298 tokens. The expected native output is
`Once upon a time,`. This fixture result is not a GPU observation.

The GPU sequence uses the unified W=4 full-map unfused Executor2 and FreeRun.
It checks a 65,536-token boot prefix, restores that actual GPU checkpoint for
256 tokens, then resumes to HALT. Reference files only provide comparisons and
the stopping budget. Checkpoints contain generated GPU recurrent tensors,
lag histories, the next embedding, last token, offset, and cumulative timer.

Phase 1's 14.06 minutes per position used the fused original graph. It is a
historical result from a different backend. The new comparison alternates five
65,536-token runs each of the original circuit with the Phase 1 program, the
unified circuit with the same Phase 1 program, and the unified circuit with the
ndot program. Setup and validation are excluded from generation timers.
Any original-unfused minutes-per-position estimate from the bounded rate must
be labeled a projection, separate from the full ndot measurement.

The stock target is 32,768 tokens from boot. The first NDOT4 instruction begins
at reference trace offset 18,616, so this target includes accelerated matvec
code. Stock uses ordinary `Qwen3_5ForCausalLM.generate`, a 15 GiB allocator cap,
a 512 MiB free-VRAM guard, initial snapshot injection only, and no offload.
GPU jobs run sequentially through the existing `.venv` and `ssh gpu`.

The corrected gate passes in [gate2-liveness.json](evidence/gate2-liveness.json).
Both final stack pointers are `0x01010000`. The differing stack words at
`0x0100ff18`, `0x0100ff1c`, and `0x0100ff2c` are entirely below that pointer.
Both replayed final states reproduce the retained reports exactly.

The original binary's last x28 through x31 writes execute at `0x40174`,
`0x40108`, `0x40170`, and `0x40158`. Their final data/address flows reach the
store at `0x40110`, whose effective address is `0x0100aa54`. The ndot binary's
last writes execute at `0x40108`, `0x40124`, `0x401c0`, and `0x40118`.
Its x28 is a shift selector last read by the branch at `0x401ac`; its final
x29 pointer has no later use. Its final x30 and x31 reach the store at
`0x40120`, also at `0x0100aa54`. Both stores write `0xffff2a11`, exactly the
retained word in both final non-stack RAM images. No tracked final-register
flow reaches an ECALL argument or a live stack slot. Equal stored values and
compared RAM/output cover downstream consumers of that logit. The disassembly
and executed step numbers are retained in the evidence.


## 2026-10-30 five-position neural gate passes

[neural-validation.json](evidence/neural-validation.json) independently verifies
5,423,298 actual self-fed GPU tokens on the Mac with zero mismatches.
The observed text is `Once upon a time,`. All 65,536 final RAM bytes and every
register in all three recurrent copies equal this binary's own independent
reference exactly. No dead-state exception is used. The full generation timer
is 2,269.2393337 seconds, or 2,389.92 trace tokens/s and 7.5641 minutes per
inference position including prefill. This is one full execution.
The original-unfused comparison and stock prefix remain separate gates.


## 2026-10-30 unfused windows pass; stock startup repair

[baseline.json](evidence/baseline.json) retains five alternating 65,536-token
windows for each side. Every exported actual prefix matches its Mac reference.
The original graph's median is 2,086.64 trace tokens/s, with range 2,086.07 to
2,087.05. The unified graph running the same original stories binary measures
2,392.30 trace tokens/s, with range 2,392.28 to 2,392.32. The unified graph with
the ndot binary measures 2,392.34, with range 2,392.23 to 2,392.37.
The original workload projects to 25.90 minutes per position on the original
unfused graph and 22.59 on the unified graph. These are bounded-rate projections.
The full ndot result is a measured 7.56 minutes per position. Phase 1's historical
14.06 minutes used the fused original graph and is a different backend.

The first stock process exited two before generating tokens because the new
experiment helper `profile.py` shadowed Python's standard-library module during
library imports. [stock-startup-failure.json](evidence/stock-startup-failure.json)
retains the exact error. The helper is now [profile_gpu.py](profile_gpu.py).
The experiment's stale module and its own bytecode were removed from the PC.
A real `torch`, `cProfile`, and standard-library `profile` import control passes.
Only stock restarts, from ELF boot in a fresh private artifact directory.
No failed numerical comparison is retried or waived. The completed neural run
and all 15 completed unfused prefixes remain unchanged.

## 2026-10-30 physical-memory guard rejects stock attempt

The first fresh stock attempt after the helper rename generated 10,205 actual
tokens with zero mismatches on the Mac, but physical free VRAM reached
531,628,032 bytes, or 507 MiB. This is below the 512 MiB floor. The attempt is
rejected and is not the required stock result. The last periodic progress report
was at 9,216 tokens. [stock-memory-failure.json](evidence/stock-memory-failure.json)
retains both the exact checked count and the failed report. CUDA-only free
memory was higher, which is why the separate physical-memory sampler matters.

A separate 16-token allocation probe uses the native allocator with
`PYTORCH_ALLOC_CONF=expandable_segments:True`. It retains the same full-map model,
weights, stock generation path, 15 GiB cap, and 512 MiB guard. Its actual token
file is exact on the Mac. Reserved memory is 14,417,920,000 bytes versus
14,403,160,064 allocated bytes, with zero inactive split bytes. Minimum sampled
physical free VRAM is 953,155,584 bytes, or 909 MiB.
See [allocator-probe.json](evidence/allocator-probe.json).
This short probe does not pass the required stock prefix. A fresh 32,768-token
run starts from ELF boot with that allocator setting. No model state is copied
from either failed attempt or the probe. The launcher now records this allocator
choice for future stock runs. No cap, guard, model weight, or forward is changed.


## 2026-10-30 complete Gate 2 flow accounting

[gate2-corrected.json](evidence/gate2-corrected.json) combines the executed
liveness audit, control-use audit, and fresh native 64-position recheck.
[verify_gate2.py](verify_gate2.py) checks their source bindings and exact retained
RAM/output bytes. This is the current Gate 2 acceptance report.

The direct data/address flows above end at the final logit word. In addition,
the final NDOT x28 shift selector controls all 512 rescaling decisions at
`0x401ac`. [gate2_control.py](gate2_control.py) independently replays the retained
64-position ndot binary and records every selected store at `0x40120`.
Its effective addresses cover `0x0100a258` through `0x0100aa54`, exactly the
512-word logits array. Every value stored in those final executed occurrences
matches both final non-stack RAM images byte for byte. The branch is not taken
in all 512 occurrences. The final registers, RAM, output and SP reproduce the
retained state exactly. See [gate2-control.json](evidence/gate2-control.json).
These equal logit values and the exact 64-position ID/hash/byte/clip observations
cover downstream argmax, hash and output consumers. No differing value escapes
through an exit ABI argument or live stack slot.


## 2026-10-30 M5 measured GPU result

The unified W=4 full-map unfused Executor2 and FreeRun generated all
5,423,298 trace tokens from boot through HALT with zero mismatches against
this ndot binary's independent Mac reference. Actual generated BYTE records
produce `Once upon a time,`. The five prediction IDs are `[403, 407, 261, 378, 432]`.
All 65,536 final RAM bytes and every register in all three recurrent register
copies compare exactly on the Mac. The earlier cross-binary dead-state
exception is not used for neural validation.

[validation.json](evidence/validation.json) checks the exported raw token file,
output bytes, RAM dump, register dumps, script hashes, checkpoint restore
boundaries, and every timed chunk's actual token hash. The prefix checkpoint is
at 65,536 tokens, the independently checked restore reaches 65,792, and the
full run resumes from that actual GPU checkpoint. The final checkpoint is at
5,423,298 tokens. No reference state is supplied after initial boot.

The one retained full execution measures 2389.92 trace
tokens/s over 37.82 generation minutes, or
7.56 minutes per inference position including
prefill. This is one execution, not a repeated full-workload benchmark.
The timer includes graph replay, self-feedback, synchronization, token copies,
and CPU transfer. It excludes model setup, comparison, IO, observations,
checkpoints, and restart gaps. Peak PyTorch allocation is
266,149,376 bytes under an 8 GiB cap.

### Unfused comparison

The bounded comparison alternates five runs per side. Each run autonomously
starts from its own ELF boot snapshot and checks 65,536 generated tokens.
The original and unified graphs both execute the identical retained Phase 1
stories binary in the first two rows. The third row uses the ndot binary.
All 15 prefix token files match their independent reference exactly.
These are generation timers with the same chunk size and backend family.

| Circuit and guest | Median trace tokens/s | Five-run range | Microseconds per transition |
| --- | ---: | ---: | ---: |
| Original graph, Phase 1 program | 2086.64 | 2086.07 to 2087.05 | 479.24 |
| Unified graph, Phase 1 program | 2392.30 | 2392.28 to 2392.32 | 418.01 |
| Unified graph, ndot program | 2392.34 | 2392.23 to 2392.37 | 418.00 |

At the measured bounded original-unfused rate, Phase 1's 16,213,774-token
workload projects to 25.90
minutes per inference position. This is a projection from a boot window, not a
new full original-unfused execution. At the unified graph's bounded rate, the
same original workload projects to
22.59 minutes per position.
The full ndot measurement above uses its actual 5,423,298-token workload.
Phase 1's historical 14.06 minutes per position used the original graph with
ROM and feedback fusions. It is a different backend and is not a controlled
speedup baseline. [baseline.json](evidence/baseline.json) retains all runs.
No causal claim about a particular kernel is inferred from these rates.

### Stock prefix

Ordinary stock `Qwen3_5ForCausalLM.generate` autonomously generated
32,768 tokens on the unified full-map model with zero
mismatches against this ndot ELF's reference. Independent Mac comparison of
the exported raw token file passes. The prefix executes 1,136 NDOT4 instructions
and reaches matvec code; it does not finish a guest LLM position.
See [stock.json](evidence/stock.json) and
[stock-prefix-scope.json](evidence/stock-prefix-scope.json).

The native allocator uses `expandable_segments:True`. The allocator cap is
15 GiB and the free-VRAM guard is 512 MiB. The minimum
free VRAM across CUDA guard checks and physical-memory samples is
552,599,552 bytes, or 527 MiB.
Peak allocated memory is 14,740,581,888 bytes and peak reserved
memory is 14,818,476,032 bytes. There is no layer offload,
forward replacement, custom generation loop, teacher forcing, or intermediate
cache injection. Only the initial ELF boot snapshot enters the model.
The reference only checks generated tokens. GPU jobs run sequentially.

ROM and feedback fusion adaptation was not attempted. It is optional follow-up;
M5's required unfused and stock checks pass. Full tensors, traces, model blobs,
and binary builds remain private under `.cache/` and the PC artifact directory.
The earlier helper-name startup failure and the rejected physical-memory
guard attempt remain in [stock-startup-failure.json](evidence/stock-startup-failure.json)
and [stock-memory-failure.json](evidence/stock-memory-failure.json). The separate
16-token allocator probe is not counted toward this prefix.
No commits, installations, upstream edits, or excluded-directory edits were made.
The append-only decision trail is [decisions.tsv](decisions.tsv).
Prove It Works required the exported raw files and exact final state.
Model the Domain kept actual checkpoints separate from comparison fixtures.
Work ran inline without an independent agent review.

### Separate profiler samples

[profile-original.json](evidence/profile-original.json) and
[profile-unified.json](evidence/profile-unified.json) each check a 2,048-token
self-fed warmup and 64 profiled tokens against the reference. Both workers exit
zero. Their durations are excluded from all reported benchmark timers.
The head kernel is the largest GPU-time component, at 68.26% of self CUDA time
for the original graph and 49.08% for unified W=4. In the unified sample, FFN
kernels account for 22.89% and residual/history application for 20.77%.
These samples identify the hot kernels. They do not isolate which constructor
change causes the difference or establish a workload-wide causal profile.
The final tasklist contains no Python jobs.

### Retained inputs and reproduction

The five-position ELF and its independent trace are in
`.cache/tiny-llm/ndot5-20261030`. The original Phase 1 fixture is unchanged at
`.cache/tiny-llm/neural-stories`. Actual exported files and collection provenance
are in `.cache/tiny-llm/ndot-m5-20261030-actual`.
The completed main and all 15 baseline windows are retained on the PC under
`C:/Users/Admin/qwenputer-games-work/artifacts/tiny-llm-ndot-m5-20261030`.
The accepted fresh stock run is under
`C:/Users/Admin/qwenputer-games-work/artifacts/tiny-llm-ndot-stock-expandable-m5-20261030`.
The earlier stock startup and memory failures remain in their separate original
directories. The collection keeps their exit codes and raw evidence separate
from the 19 successful worker exits. No failed prefix contributes tokens or
state to the accepted stock run.

Verification from the retained private inputs uses these commands.

```sh
.venv/bin/python experiments/tiny-llm-ndot/verify_gate2.py
.venv/bin/python experiments/tiny-llm-ndot/verify_gpu.py \
  --actual .cache/tiny-llm/ndot-m5-20261030-actual \
  --evidence experiments/tiny-llm-ndot/evidence
```

A new GPU execution from those same fixtures requires a fresh label.
The launcher now selects native expandable segments for its stock stage.

```sh
.venv/bin/python experiments/tiny-llm-ndot/launch.py \
  --label FRESH \
  --gate experiments/tiny-llm-ndot/evidence/gate2-corrected.json
```

The Mac verifier originally compared a partial 32,768-token stock chunk against
a full 65,536-token reference slice. That array-length error is corrected by
bounding both slices at the requested prefix length. The unchanged actual stock
file passes. A one-bit corruption of the last token in a temporary copy fails
with exactly one mismatch, recorded in
[verifier-tail-control.json](evidence/verifier-tail-control.json).
No generation rerun or numerical exception was used for this verifier repair.

Required M5 checks pass. The stock result is a bounded prefix and the original
unfused minutes per position are projections. Optional ROM/feedback fusion
adaptation remains open. There are no unresolved mismatches in the accepted
runs. Fix Root Causes required renaming the shadowing helper and changing the
allocator rather than lowering the physical-memory guard.

## 2026-10-30 38-position autonomous story passes

The unified W=4 circuit with accepted ROM and feedback fusion generates all
53,296,563 trace tokens from ELF boot through HALT with zero mismatches
against the independent Mac trace. The prompt is `Once`. All 38 inference
positions complete, including prefill. The observed text is:

> Once upon a time, there was a little girl named Lily. She loved to play outside in the park. One day, she saw a big

This text comes from actual self-fed GPU BYTE records. The Mac verifier
independently extracts those bytes from the exported actual token file and
compares the complete guest output with the native and software references.
All 65,536 final RAM bytes and all 32 registers in each of the three recurrent
register copies equal this ELF's own Mac reference literally. No state coordinate
is masked or normalized, and no tolerance is used.

The one complete execution measures 14,225.901527 generation seconds,
or 237.098 minutes and 3.952 hours. Its rate is 3,746.445
trace tokens/s and 6.2394 minutes per inference position.
These are measured full-run values. The timer includes graph replay, self-feedback,
synchronization, trace copies and CPU transfer. It excludes setup, validation,
observations, checkpoint IO and restart gaps. This is one full execution, separate
from the five-repeat bounded timing experiment. The longer context increases the
mean guest workload per position, so the earlier five-position projections use a
different denominator.

[Independent Mac acceptance](../tiny-llm-ndot-fused/evidence/story-validation.json)
checks every actual token, final RAM bytes, all three register copies, observed
output, timed chunk hashes and actual checkpoint restore boundaries. The run
restores actual generated GPU state at offsets 65,536 and 65,792 before continuing.
All recurrent S tensors, lag histories, next embedding and token come from the
GPU checkpoint. Ordinary and private outputs are asserted zero before saving.
The reference supplies comparisons and the stopping budget only. No reference
state enters after boot, and no checkpoint is synthesized by the host.

The 38-position fixture has 53,296,563 trace tokens. At the slower fused window
median it projected to 14,308.65 generation seconds, about 3.97 hours. This selected
38 positions within the requested approximate four-hour budget. The next position's
end boundary projects to 14,779.08 seconds before final diagnostic output, above
the conservative four-hour budget. [Budget counts](../tiny-llm-ndot-fused/evidence/story-budget.json)
remain reference-only projections. The guest model
file is byte-identical to M5. The generated unified circuit hash remains
`5e156bdb8fa6c543d96f0760a1a4d18dfea4cba3e558673a909bed1be0fde7da`.
See [fixture bindings](../tiny-llm-ndot-fused/evidence/story-fixture.json),
[the worker report](../tiny-llm-ndot-fused/evidence/story-run.json) and
[checkpoint provenance](../tiny-llm-ndot-fused/evidence/story-checkpoint.json).

The detached accepted directory is
`C:/Users/Admin/qwenputer-games-work/artifacts/tiny-llm-ndot-story-38-20261030b`.
The independent Mac fixture is `.cache/tiny-llm/ndot-story38-20261030`, and the actual
export is `.cache/tiny-llm/ndot-story38-20261030-actual`. Raw traces and checkpoints
stay private. GPU access uses `ssh gpu`, cmd and encoded PowerShell with the
verified existing venv. Jobs run sequentially and detach through
`Invoke-CimMethod Win32_Process Create`.

The first story directory `tiny-llm-ndot-story-38-20261030a` generated an exact
65,536-token prefix. Its restore setup refused the retained scratch package before
new generation. The launcher now removes its own scratch package between processes.
[The startup failure](../tiny-llm-ndot-fused/evidence/story-restore-startup-failure.json)
remains separate. Accepted run b starts fresh from boot. No failed numerical
comparison was retried or waived.

The fusion follow-up passes both full isolated 65,536-token windows and standalone
race stress. Five alternating timings measure 419.083 to 268.472 microseconds per
token at boot and 418.543 to 268.253 in the reference-seeded matvec window.
Those windows project the five-position M5 workload to 7.576 versus 4.853 and
7.566 versus 4.849 minutes per position. They are bounded-rate projections.
The complete 38-position result above is measured independently.
See [fusion acceptance and timing](../tiny-llm-ndot-fused/README.md).

There are no unresolved mismatches in the accepted runs. The exact fusion gate
covers this graph and the two stated windows; the story covers this complete
38-position execution. It is not a proof for arbitrary graphs or workloads.
All changes remain uncommitted. Upstream, circuit weights and the excluded
directories are unchanged. Work ran inline without independent agent review.
