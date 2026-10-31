# Unified W=4 ROM and feedback fusion

This experiment preserves the unified graph, all circuit heads, weights, memory
encoding, arithmetic and feedback rules. It changes the Executor2 launch schedule.
The full isolated raw-bit gate and five alternating timing repeats pass.
The separate 38-position story also passes independent Mac final-state validation.

## Graph-derived schedule

[fusion.py](fusion.py) scans head sublayers for data-ROM heads. A candidate must
have no S writes, decay, integer-counter heads or lagged residual reads. Its sparse
query and gate reads must not intersect any earlier candidate's output dimensions.
A dependency ends the current batch. Sublayer indices, value-head counts, key-head
offsets, projection widths, group mapping and output shapes come from the graph.
The short unified smoke run selects one batch of 28 consecutive sublayers.
It does not assume the original graph's 31 groups or 32 value heads per group.

Each group retains a private output slice. The batched head kernel uses the
unchanged non-counter arithmetic. The ordered apply kernel adds group outputs
in original order, shifts each group's three histories, and clears outputs.
An explicit block barrier finishes all cross-warp history and output reads before
stores. Multi-block head execution finishes before apply through the launch
boundary. Feedback reuses the verified single-block decode/embed kernel, including
its explicit read-before-write barrier. No following multi-block circuit work is
folded across a global dependency.

The alternative of keeping fixed group counts was rejected because the unified
allocator reuses residual dimensions. Splitting decode and embedding into two
kernels would retain a safe launch boundary, but the verified single-block
feedback already handles this batch-one geometry without an intermediate tensor.

## Exact verification

[isolation.py](isolation.py) runs one executor in one process. Separate baseline
and fused processes each record 65,536 tokens at boot and 65,536 tokens beginning
at reference offset 32,768. The second window is reference-seeded for equivalence
and performance only. [window-scope.json](evidence/window-scope.json) records
3,127 NDOT4 instructions in the covering records of that window.

Every token has a SHA256 root over raw 32-bit state. The recorder scans every S
word against the preceding state, transfers changed index/value pairs, reconstructs
all S bytes on the CPU, and updates SHA256 leaf hashes. It hashes complete raw
bytes of every lag history, pre-feedback residual, next embedding, both token
words and ordinary output buffer. Private fused output is recorded and must be
zero. One-bit controls exercise the recorder. Final reconstructed S must equal
the actual GPU tensors. A delta capacity overflow fails rather than dropping data.
[compare.py](compare.py) checks equal token counts and roots in both windows.
No float tolerance or reduced numerical checksum is used.

[apply_race.py](apply_race.py) checks residual, histories and private output under
8,192 queued graph replays and retains an unsafe control. [feedback_race.py](feedback_race.py)
compares all embedding bits and the int64 token against original PyTorch feedback
under 8,192 queued replays. Cases include every encoded family, random payloads,
signed ties-to-even rounding and family and table-row clamping.

## Timing contract

[bench.py](bench.py) requires both full isolated windows, the recorder control,
zero reference mismatches, standalone stress passes and matching fusion sources.
It alternates unfused and fused normal `FreeRun.run` calls five times per window.
Each call generates and checks 65,536 actual tokens. The timer includes graph
replays, self-feedback, trace copies, synchronization and CPU conversion.
Construction, capture, reset, reference preparation, comparison and evidence IO
are excluded. Minutes per position are projections using the mean trace workload
of the retained five-position `Once` fixture, explicitly separate from a full story.

## Accepted result, 2026-10-30

[checks.json](evidence/checks.json) binds both isolated windows, source hashes,
standalone stress and all 1,310,720 timed tokens. Each isolated side checks 131,072
tokens. All 86 common tensors match after every token, including every S word,
three lag histories per head sublayer, pre-feedback residual, next embedding,
both token words and the ordinary output buffer. The fused private output is
also zero after every token. There are zero reference-token mismatches and zero
full-state digest mismatches. Peak fused recorder allocation is 5,713,201,152
bytes, below the 8 GiB cap. Final S reconstruction and one-bit controls pass.
See [the comparison](evidence/comparison.json), [baseline](evidence/baseline-isolation.json)
and [fused recorder](evidence/fused-isolation.json).

The accepted graph-derived batch is sublayers 27 through 54, containing 896
value heads across 28 groups. Projection widths are KQ=16, KK=2, KV=2 and KZ=2,
with QHOT=2. The planner derives these values instead of asserting an old layout.

[apply-race.json](evidence/apply-race.json) records 8,192 queued replays with
nonzero initial lag histories. Residual, all history words and cleared private
output pass. The unsafe control reports 16 bad coordinate checks. The protected
kernel reports zero. The earlier zero-history test is retained separately in
[initial-apply-race.json](evidence/initial-apply-race.json). Its unsafe diagnostic
hard-coded the original 31-group expected-value bits although this graph has 28
groups. That diagnostic field is corrected in the accepted nonzero-history test.
[feedback-race.json](evidence/feedback-race.json) also passes 8,192 queued replays
with zero embedding-bit or token differences.

The table reports median wall time and full five-run ranges in microseconds per
token. See [bench.json](evidence/bench.json) and [timing-summary.json](evidence/timing-summary.json).
Projected minutes use the retained five-position workload mean, 1,084,659.6 trace
tokens per inference position including prefill. They are bounded-rate projections.

| Window | Backend | Median us/token | Five-run range | Projected min/position |
| --- | --- | ---: | ---: | ---: |
| Boot | Unfused | 419.083 | 418.358 to 419.968 | 7.576 |
| Boot | Fused | 268.472 | 268.250 to 268.681 | 4.853 |
| Matvec, reference-seeded | Unfused | 418.543 | 418.483 to 418.706 | 7.566 |
| Matvec, reference-seeded | Fused | 268.253 | 268.246 to 268.261 | 4.849 |

Fusion is 1.5610 times faster at boot and 1.5603 times faster in matvec. The
improvement exceeds the observed ranges. All timed tokens match. A separate
[checked profiler sample](evidence/profile.json) excludes warmup and tracing from
these timers. FFNs now take 35.99% of self CUDA time, remaining heads 29.77%,
ordinary apply 18.89%, ordered ROM apply 6.66%, batched ROM heads 6.22% and feedback
2.16%. FFNs and remaining head work limit further gains. These profiler shares
are not additive components of the wall-time table. Host snapshots show 12 logical
CPUs, 3% to 6% sampled CPU load, 100% GPU utilization and 2,880 to 2,887 MHz SM
clocks during timing. Only this experiment's venv Python process and its shim run.

The first benchmark process failed before generating tokens because fusion was
imported before runtime package preparation. The repaired import order uses a fresh
benchmark directory. [benchmark-startup-failure.json](evidence/benchmark-startup-failure.json)
retains the exception. No numerical mismatch was waived or retried.

[input-bindings.json](evidence/input-bindings.json) checks matching SHA256 hashes
on the Mac and PC for both retained ELF/reference/model fixtures and the immutable
fusion dependencies. All 14 files match. The check reads files without changing
inputs or GPU state.

## Reproduction and private artifacts

Run from the repository root. The existing `.venv` and clean pinned upstream
sources are required. No installation or upstream edit is part of this workflow.

```sh
.venv/bin/python experiments/tiny-llm-ndot-fused/launch.py --label FRESH
```

The launcher tar-syncs only named experiment dependencies and the retained private
five-position fixture. It verifies the remote venv, checks tasklist before each
worker, and detaches a sequential PowerShell script with `Invoke-CimMethod
Win32_Process Create`. All PowerShell goes through `-EncodedCommand`; GPU access
uses `ssh gpu`. Each executor has a watchdog and an 8 GiB allocation cap.
New remote outputs use `artifacts/tiny-llm-ndot-fused-*` directories.

Large raw states, per-token JSONL files, graph artifacts and traces stay private
on the PC. Small report JSON files are exported to `evidence/`. The longer story
uses a separate `artifacts/tiny-llm-ndot-story-*` directory. Story checkpoints
contain actual generated GPU tensors only. No reference-seeded checkpoint enters
that run. Its actual token trace, literal final RAM, all three register copies,
observed BYTE output and checkpoint restore boundaries are checked on the Mac.

All work stays uncommitted. The excluded directories and frozen upstream sources
are untouched. Model the Domain kept graph-derived batches and actual story
checkpoints explicit. Prove It Works keeps timing behind raw-state acceptance.
Benchmark Checklist requires five alternating repeats and checked generated tokens.
The append-only [decision trail](decisions.tsv) records choices and gates.
The user requires inline execution, so no independent agent review is performed.

## Complete autonomous story

The [dated story result](../tiny-llm-ndot/README.md#2026-10-30-38-position-autonomous-story-passes)
reports 53,296,563 actual self-fed GPU trace tokens, zero mismatches, 3,746.445
trace tokens/s and 6.2394 measured minutes per inference position.
The observed text is `Once upon a time, there was a little girl named Lily. She loved to play outside in the park. One day, she saw a big`. All final RAM bytes and all three register
copies match independently on the Mac. Actual GPU checkpoints restore at 65,536
and 65,792 tokens. [story-validation.json](evidence/story-validation.json) retains
all acceptance checks. No reference-seeded state enters the story.


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

## 2026-10-31 story-verifier correction

The story verifier now requires a nonempty, contiguous journal with positive
chunk sizes, full trace coverage, every chunk hash, monotonic cumulative timing,
and a terminal time equal to the report. Historical evidence is unchanged.
Commit `9815ee1` holds the verifier before this correction. Older reports retain
their original source hashes. The retained checkpoint binds workers from `e7bd077`; select that
revision explicitly for script-hash checks, following
[Historical source bindings](../checkpoint-integrity/README.md#historical-source-bindings).
All data checks still use the retained exports. The fresh result is
[evidence/story-validation-20261031.json](evidence/story-validation-20261031.json).

```sh
.venv/bin/python -B experiments/tiny-llm-ndot-fused/verify_story.py --inputs .cache/tiny-llm/ndot-story38-20261030 --actual .cache/tiny-llm/ndot-story38-20261030-actual --source-revision e7bd077 --out experiments/tiny-llm-ndot-fused/evidence/story-validation-20261031.json
```
