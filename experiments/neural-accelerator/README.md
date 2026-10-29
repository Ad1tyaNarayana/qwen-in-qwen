# Neural dot-product accelerator, stopped at M2

This experiment implements an independent ISA oracle, guest intrinsics, an
isolated software reference extension, and a new full-map neural constructor.
M1 passes the retained tests. M2 fails its token-family corruption control.
W=8 and M3 were not run after that failure. There is no stock autonomous
generation result or accelerator performance claim.

## Scope and method

The instruction contract is in [SPEC.md](SPEC.md). The project-private custom-0
instructions accumulate four or eight signed-byte by signed-halfword products
into one architectural register modulo 2^32. The physical memory map, ordinary
value records, register commit path, and original scalar reference remain.
Attention, hidden accumulators, pointer updates, and multiple destinations are
excluded. Width 16 is illegal.

[oracle.py](oracle.py) uses raw byte arrays and Python integers. It imports no
compiler, reference, serializer, or circuit helper. [reference.py](reference.py)
subclasses the frozen scalar Machine and implements dots separately. Unextended
instructions use the inherited record method. [ndot.h](ndot.h) uses `.insn` with
unsigned accumulation and an early-clobber output constraint.

[constructor.py](constructor.py) creates a separate constructor module from the
private expanded-map core source, with checked insertion sites. It neither
edits upstream nor modifies the prepared package files. Its manifest records
the base source, generated constructor, and insertion sites. [prepare.py](prepare.py)
retains the existing revision and clean-upstream checks and records the extra
reference and IR source hashes. Code predecode reads instruction words only.

The constructor selects DROM columns with the original two-hot row queries.
RAM words remain striped across the original banks. Output routing retains
separate lanes without a RAM copy. [arithmetic.py](arithmetic.py) constructs
magnitudes, unsigned nibble products, normalized columns, signed lane bits,
a balanced reduction tree, and the final old-rd addition. It never assembles a
32-bit result into one arithmetic residual scalar.

## Retained results

[evidence.json](evidence.json) binds the scripts, versions, source pin, and
individual results. The environment is Mac CPU, Python 3.13, torch 2.15.0,
Transformers 5.21.0, NumPy 2.5.4, and Zig 0.17.0. No packages were installed.
The GPU machine was not accessed. No commits were made.

M0 was reproduced from the design's script. Its nibble and digit graphs pass
256 and 512 exhaustive cases with IR exactness and kernel checks. Supplied-input
stock CPU decode and prefill match at 11 product and nine digit-adder positions.
The Python full-dot checks remain mathematical evidence, not neural execution.

M1 passes 25,174 directed and seeded instruction cases, 42 prior-store cases,
128 chained dots across two refreshes, and a nonzero fault-PC check. Separate
vectorized checks compare independent byte interpretations against the reference
signed interpretation for all 16,777,216 s8/s16 operand pairs. That count is an
arithmetic interpretation check, not 16 million neural or full-instruction runs.
Tests cover reserved and disabled encodings, aliases, x0, alignment residues,
first/last spans, every DROM bank, RAM rows/banks/slots, signed extremes,
cancellation, wraparound, and nibble and token-chunk boundaries.
The guest object contains exactly `40c5850b` and `40c5950b` for the two intrinsics.

The unchanged Doom first-frame fixture retires 3,885,544 instructions and
6,342,933 tokens. Old and extended reference traces have SHA-256
`23d6d7138e2612dfd971f3d86ac3f711c98a0cd2ac90e77410bcc38240edde38`.
Final registers, PC, RAM hash, output hash, fault, and exit status also agree.
The fixture is bound to the design's `counts-legacy.json`. An initial test-script
precondition used an older report with a different ELF metadata hash; that
precondition was corrected to the design's matching fixture before replay.

The original five-position stories260K Phase 1 fixture retires 10,756,734
instructions and 16,213,774 tokens. Both reference machines reproduce the
historical trace SHA-256
`68c7738430dbc024b3d78711be5fee39f8e5c8b6c7cda5de45c8c8f76836efa5`.
Historical final-RAM and output hashes also agree. This regression covers the
five-position fixture, not a new 64-position replay.

M2 W=4 passes the first and last legal spans, including the last full-map DROM
bank. It also passes 380 instructions and 940 supplied-input tokens with chained
dots, varied memory boundaries, SW/SB/SH, and immediate scalar loads. Every
tested dot position compares operand bits, arithmetic result bits, and predicted
record tokens. `evaluate(check=True, kernel=True)` passes those positive checks.
This checks the BF16 materialization contract and the evaluator's recurrent and
chunk read-kernel models. It is not stock full-model generation.

| Geometry | Original | W=4 | Added |
| --- | ---: | ---: | ---: |
| Chip layers | 44 | 65 | 21 |
| Residual dimensions | 2,679 | 4,794 | 2,115 |
| Selected hidden size | 2,816 | 5,120 | 2,304 |
| FFN units | 42,014 | 66,492 | 24,478 |
| Key heads | 627 | 627 | 0 |
| Logical value heads | 1,247 | 1,247 | 0 |

W=4 exceeds both the design's 8–14-layer W=4 estimate and its 10–16-layer
W=8 estimate. The graph's largest FFN sublayer has 7,995 units, within 9,216.
No dense parameter count or peak stock memory result was measured.

## Failed gate and remaining work

The token-family corruption assertion fails at [test_ir.py](test_ir.py), line
221. The test flips the final tied-output `c.code[0]` residual, then calls
`core.predictions`. In standalone mode that helper reads the earlier
`c.out_fam` and `c.out_pay` bits instead. Static inspection explains why this
control does not change the observed token. This is a verification defect;
the retained failure does not demonstrate a dot arithmetic error. The gate
still fails and was not bypassed or repaired after failure.

The raw traceback in `ir-w4.json` contains a misleading source-line display.
Additional tests had been edited on disk while the process was running, so
Python displayed the newer text for the old compiled line number. Those
unexecuted additions were removed after stopping. The restored script's line
221 is the failed token-family assertion. The evidence retains the raw failure
and separately records this diagnosis. No post-failure neural test was run.

The earlier control assertions were reached without an exception, but their
individual values were not flushed before the final control failed. They are
not published as separate completed evidence. Complete memory enumeration,
all-byte final neural RAM comparison, exhaustive fault coverage in the neural
constructor, cached/chunked stock-state comparison, checkpoint/reset coverage,
and a successful corruption suite remain open. W=8 code exists but is untested.
There is no M3 implementation or stock generate run.

A continuation must first repair the observer/control mismatch, retain each
control result immediately, and finish W=4 coverage. W=8 follows only after that
gate passes. M3 then needs the tiny generic map, full chip vocabulary and output
head, stock CPU greedy generate on two programs and changed DROM with identical
weights, at least 1,024 dependent instructions per run, and exact PC/register/RAM
comparisons at commits. It must also compare cached decode with chunk execution
on the generated sequence and retain complete reset and resume histories.

M4 must measure the actual full-map stock geometry, parameters, memory, and
bounded full-map stock execution, with compact/stock equivalence. M5 must rebuild
the aligned stories guest and compare all 64 predictions, every-position logit
hashes, decoded bytes, clipping counts, and final architectural state against
the original integer engine. Neither milestone is established here.

## Reproduce

Run from the repository root with the existing virtual environment.

```sh
.venv/bin/python experiments/neural-accelerator/m0.py
.venv/bin/python experiments/neural-accelerator/test_isa.py
.venv/bin/python experiments/neural-accelerator/test_intrinsic.py
.venv/bin/python experiments/neural-accelerator/test_legacy.py doom
.venv/bin/python experiments/neural-accelerator/test_legacy.py stories
.venv/bin/python experiments/neural-accelerator/test_ir.py --width 4
```

The final command is expected to fail at the retained corruption control.
Fresh private packages and object files go under `.cache/neural-accelerator/`.
Result JSON files go beside the scripts. The failed gate intentionally prevents
a W=8 or M3 success claim.

Model the Domain shaped the immutable oracle outcome and separate instruction,
reference, and circuit modules. Prove It Works required actual legacy replays
and retained the negative gate. Architecture alternatives were considered
inline under AGENTS.md. No independent agent review was performed.

## 2026-10-29 continuation: W=4 M2 passes before and after reduction

This section supersedes the initial M2 failure and remaining-work statements
above. Earlier findings remain historical. The raw initial failure is retained
in [ir-w4-initial-failure.json](ir-w4-initial-failure.json).

The repaired token-family control flips `out_fam[0]`, which standalone
`core.predictions` actually reads. Every control now records its expected and
observed values, flushes and fsyncs the result, then asserts detection. Memory,
old-accumulator, PC, write-gate and token-family controls all pass, both before
and after reduction. Their boundaries are respectively the first DROM operand
bit, old-rd bit, predecode PC bit, first RAM operand after SW, and first predicted
result token. A NumPy-scalar JSON serialization defect in the new persistence
code was corrected before publishing these results.

The full-map W=4 suite passes `evaluate(check=True, kernel=True)` on all 80,549
sequence tokens and 34,256 instructions, with 34,044 per-token dot comparisons.
It covers all aligned spans within one 8 KiB DROM bank, first and last spans
of all 512 DROM banks, and every 8-byte-aligned span in the 64 KiB RAM. This
enumerates the DROM column, row and value-slot selectors and RAM row, slot and
bank selectors. Prior SB, SH and SW cases exercise all eight byte lanes, four
halfword lanes and two word lanes. The earlier dependent partial-store and
scalar-load sequence remains included.

All 31 invalid alignment-residue pairs, 14 region and wrapped-address cases,
seven disabled/reserved widths, and nine read-only store cases pass. Each of
these 61 faults checks the exact serialized fault and nonzero PC, every register
in all three recurrent register copies, and all 65,536 unchanged RAM bytes.
First and last legal spans pass separately. Final neural RAM is decoded from
the evaluator's recurrent state, without consulting the reference during
extraction. Every byte matches. Both variants have SHA-256
`6f424c340eec2feca22ac1a7ea2ba1a622ea9f8770b0ddbf2ff0a7dd714c9516`.

| Geometry | Original | W=4 before reduction | W=4 after reduction |
| --- | ---: | ---: | ---: |
| Chip layers | 44 | 65 | 64 |
| Residual dimensions | 2,679 | 4,794 | 2,113 |
| Selected hidden size | 2,816 | 5,120 | 2,304 |
| Graph FFN units | 42,014 | 66,492 | 68,917 |
| Key heads | 627 | 627 | 627 |
| Logical value heads | 1,247 | 1,247 | 1,247 |

[REDUCTION.md](REDUCTION.md) explains the clear and history proof and the two
allocation schedules considered inline. The selected allocator reuses 2,553
dimensions, pins all 264 convolution-lag sources and three tiny counter outputs,
and retains every probe. Its exact separate clear adds FFN units but no layers.
Removing redundant enables on already-zero inactive lane products saves one
layer. [liveness-w4.json](liveness-w4.json) records every lifetime and handover.
The largest graph MLP has 8,260 units against 9,216 available. Output cleanup
also fits. No head shape, recurrent table, memory map or feedback rule changed.

The unreduced result is [ir-w4-before-reduction.json](ir-w4-before-reduction.json).
It incorporates the hash-bound [control and fault run](ir-w4-controls-and-faults.json)
after an intentional interruption to increase memory-test chunk size from 64
to 2,048. The reduced [ir-w4.json](ir-w4.json) reruns everything without that
resume option. An earlier stricter reuse candidate was intentionally interrupted
before completion, so it is not an accepted M2 result.

## 2026-10-29 stock resources and conditional compact cost

[resources.py](resources.py) instantiates complete stock HF configs on the meta
device. It counts the complete chip vocabulary, tied embeddings, 16 key and
32 value heads per linear layer, intermediate size 9,216, and the required zero
full-attention layer. It allocates no dense model storage on the Mac or GPU.

| Variant | Exact parameters | BF16 parameter bytes | Peak allocated GiB | Peak reserved GiB | Fits 16 GiB with 1 GiB free |
| --- | ---: | ---: | ---: | ---: | --- |
| Original | 6,212,045,824 | 12,424,091,648 | 12.072 | 12.484 | Yes, prior bounded stock evidence |
| W=4 before | 16,035,550,912 | 32,071,101,824 | 30.750 | 31.163 | No |
| W=4 after | 7,115,541,760 | 14,231,083,520 | 13.794 | 14.206 | Estimated yes |

Only the original peak is measured. New peaks are estimates calibrated to the
original 12,962,223,104 allocated bytes and 13,404,995,584 reserved bytes in
`../tiny-llm-neural/evidence/hf.json`. The model adds exact fp32 recurrent-state
bytes per layer, retains the calibrated remaining runtime overhead, scales that
remaining overhead upward for wider hidden sizes, and retains the measured
reserved-minus-allocated gap. It assumes the same bounded-prefix runtime scope.
Reduced W=4's reserved estimate exceeds the approved 14 GiB allocator cap.
No new full-map stock allocation was attempted. Fit on 16 GiB remains an
estimate until a bounded stock run satisfies the cap and free-memory guard.

The compact cost model uses the measured Phase 1 stories rate of
3,844.248005854528 trace tokens/s, or 260.1289 microseconds per transition.
It normalizes the feedback-fusion category costs by their 239.208-microsecond
profiler span. FFN work scales with graph units, residual/history apply with
layers times physical dimensions, feedback with dimensions, and unmodeled
overhead with layers. Head and DROM-apply work stay fixed by assumption because
head shapes, counts and projection nonzero counts are unchanged. This assumes
the same GPU scheduling cost despite additional nonzero striped RAM queries.
The changed geometry and fusion kernels have not been timed or validated
together. These are conditional estimates, not measured accelerator latency.

| Variant | Projected microseconds per transition | Position 4 minutes/token | Position 63 minutes/token |
| --- | ---: | ---: | ---: |
| W=4 before | 349.989 | 5.74 to 6.12 | 11.94 to 12.32 |
| W=4 after | 308.214 | 5.06 to 5.39 | 10.52 to 10.85 |

These projections replace only measured matrix-dot loop work with 7 to 8
tokens per aligned group and retain measured nondot work. Packing, register
allocation, refresh recount and the actual aligned guest require M5 evidence.
Attention remains scalar. [resources.json](resources.json) retains configs,
exact counts, calibration and all assumptions. No new speed measurement is
claimed, and M3 has not started.

```sh
.venv/bin/python experiments/neural-accelerator/test_ir.py --width 4
.venv/bin/python experiments/neural-accelerator/test_ir.py --width 4 --unreduced
.venv/bin/python experiments/neural-accelerator/resources.py
```

The old reproduction command's expected-failure statement is superseded.
The final W=4 acceptance precedes changes confined to the W=8 test branch.
No constructor or W=4 execution-path change followed that acceptance.

The later read-only GPU inventory reports 16,303 MiB total and 14,826 MiB free.
[gpu-inventory.json](gpu-inventory.json) retains the raw sample. This free-memory
budget is smaller than reduced W=4's estimated peak plus the required 1 GiB
headroom. Therefore the nominal 16 GiB fit estimate above does not establish a
safe fit on the sampled desktop configuration. Both that availability check
and the 14 GiB allocator cap currently block a full-map stock allocation.
No GPU allocation was attempted.

## 2026-10-30 W=8 M2 passes

[ir-w8.json](ir-w8.json) passes the complete M2 gate on the reduced wider
circuit. Five corruption controls and 201 faults pass. These include 127
NDOT8 alignment cases, 31 retained NDOT4 alignment cases, 28 region cases,
six disabled or reserved widths and nine read-only stores. Each fault checks
the nonzero fault PC, every register copy, and every unchanged RAM byte.

The legal sequence checks 53,210 instructions and 125,050 supplied-input tokens,
with 52,656 per-token dot comparisons. Both NDOT4 and NDOT8 enumerate every
aligned span of one DROM bank, both edges of all 512 DROM banks, and every
aligned RAM span. Both widths also test prior SB, SH and SW at every lane.
The narrower upper-stripe case passes. All arithmetic, operand and output bits
pass `evaluate(check=True, kernel=True)`. All final RAM bytes match, with SHA-256
`2db8f068dde50fde839f972dc1d0dfb361e30773636687a47ef1246cfbc8c93d`.

The control and fault results were retained in
[ir-w8-controls-and-faults.json](ir-w8-controls-and-faults.json) before an
intentional interruption to expand the legal NDOT4 sweep. The completed run
hash-binds that unchanged circuit's retained results. The original unrun W=8
test sequence also needed an alignment correction before its first execution.
Neither interruption followed a failed numerical acceptance check.

| W=8 reduced geometry | Value |
| --- | ---: |
| Chip layers | 66 |
| Residual dimensions | 2,237 |
| Selected hidden size | 2,304 |
| Graph FFN units | 94,073 |
| Key / logical value heads | 627 / 1,247 |
| Reused dimensions / clear units | 4,153 |
| Largest graph MLP units | 8,260 |

[liveness-w8.json](liveness-w8.json) retains every lifetime and clear.
The same 264 lag sources and three tiny outputs remain pinned. Including
output and ballast cleanup, the final MLP uses 3,024 of 9,216 units.

The W=8 stock config has exactly **7,318,811,264 parameters** and
**14,637,622,528 BF16 parameter bytes**. Its estimated allocated peak is
**14.176 GiB** and reserved peak **14.589 GiB**. It fits a nominal empty
16 GiB device with 1 GiB headroom by this model, but exceeds both the approved
14 GiB allocator cap and the sampled free-memory budget with headroom.
This is not a stock allocation or execution result.

The conditional compact estimate is **348.220 microseconds per transition**,
1.3386 times the measured Phase 1 baseline. Matrix-only projected minutes per
stories260K token are **4.40 to 4.59** at position 4 and **10.57 to 10.76** at
position 63. The W=4 estimate is lower per transition and similar at the late
position, where nondot work matters more. No timing was measured for either
new geometry. [resources.json](resources.json) and
[resources.html](resources.html) contain the final four-variant accounting.

```sh
.venv/bin/python experiments/neural-accelerator/test_ir.py --width 8
.venv/bin/python experiments/neural-accelerator/resources.py --include-w8
```

Unreduced reproductions now write `ir-w4-before-reduction.json` or
`ir-w8-before-reduction.json`, preserving accepted reduced output files.
After acceptance, only output-file isolation and serialization-before-open
were changed in the test harness. Numerical construction and test paths are
unchanged. JSON reports retain the exact test-source hash used for their runs.

M2 is complete for both widths. M3 was not started. The next recommendation is
bounded generic-map W=4 stock greedy generation on two programs with changed
data and identical weights, at least 1,024 dependent instructions per run,
and exact commit-state and cache comparisons. Full-map M4 remains blocked by
the current memory limits and unproven stock and compact-executor equivalence.
Additional width or depth reduction is needed before a safe full-map stock
allocation on the sampled GPU configuration. Attention remains deferred.
No upstream edits, commits, installations, or changes to the other thread's
directories were made.

## 2026-10-30 reduced-circuit autonomous legacy regression

Gate A passes for both reduced widths. Each run starts from the unchanged ELF
boot snapshot and GO, then feeds back only its own generated tokens. The backend
is the unfused upstream Executor2 and FreeRun on the new constructor's IR.
The Phase 1 ROM/feedback fusion extensions were not installed for these runs.
The retained standalone vocabulary adapter removes the merged-chat-only Q family,
as in Phase 1; it changes neither the graph nor its arithmetic.

| Width | Unmodified guest | Tokens checked from boot | Mismatches |
| --- | --- | ---: | ---: |
| 4 | Doom first frame | 262,144 | 0 |
| 4 | stories260K Phase 1 | 262,144 | 0 |
| 8 | Doom first frame | 262,144 | 0 |
| 8 | stories260K Phase 1 | 262,144 | 0 |

[legacy_neural.py](legacy_neural.py) recreates each complete independent original
scalar trace and requires its historical ELF and complete trace hashes before
generation. Doom reproduces 6,342,933 reference tokens and stories reproduces
16,213,774. Those complete reference replays are fixture validation; the new
neural runs cover only the bounded prefixes above. Reference tokens never enter
the recurrent state or feedback path. The worker resets all histories and scratch
after CUDA graph capture and reloads the boot snapshot before generation.

[legacy-neural-validation.json](legacy-neural-validation.json) independently
compares all four exported actual token files on the Mac. The per-run reports
are [W4 Doom](legacy-neural-w4-doom.json),
[W4 stories](legacy-neural-w4-stories.json),
[W8 Doom](legacy-neural-w8-doom.json), and
[W8 stories](legacy-neural-w8-stories.json). Each detached worker exited zero.
Private GPU outputs are under
`C:/Users/Admin/qwenputer-games-work/artifacts/neural-accelerator-legacy-20261030`.

```sh
.venv/bin/python experiments/neural-accelerator/legacy_neural.py \
  --prepare-fixtures --out .cache/neural-accelerator/legacy-inputs-FRESH
.venv/bin/python experiments/neural-accelerator/launch_legacy.py \
  --inputs .cache/neural-accelerator/legacy-inputs-FRESH --include-w8 --label FRESH
.venv/bin/python experiments/neural-accelerator/verify_legacy.py \
  --actual EXPORTED_DIRECTORY --inputs .cache/neural-accelerator/legacy-inputs-FRESH \
  --out LOCAL_VALIDATION.json
```

The launch scripts sync through tar over `ssh gpu` and detach a PowerShell
parent with `Invoke-CimMethod Win32_Process Create`. They use the existing GPU
virtual environment. A command-length limit in the initial private launcher
required syncing a script before launching its short encoded invocation. A
stray character in the new stock launcher prevented its first launch before
Python started and was corrected. Neither launch defect is numerical evidence.

## 2026-10-30 M3 stops at exact cached/chunked state comparison

[stock-m3.json](stock-m3.json) retains the failed gate and source hashes.
[stock-m3-exit.json](stock-m3-exit.json) records exit code 1. No further neural
execution followed this failure. M4 was not attempted, so there is no new
full-map stock allocated/reserved peak, OOM result, or bounded stock prefix.
The newly authorized 15 GiB allocator cap and 512 MiB free-memory guard remain
available for M4 after M3 passes; the historical 14 GiB restriction is superseded.

The generic small map has 16 KiB RAM, 8 KiB DROM and the original 256 KiB code
window, at the original base addresses. The W=4 constructor has 30 chip layers,
2,152 physical residual dimensions, hidden size 2,304, 68 key heads in its graph,
and 68,686 graph FFN units. Its stock config retains 16 key and 32 value heads
per linear layer, width 9,216 MLPs, the complete chip vocabulary with tied output
head, and the appended zero full-attention layer. This is a bounded-map result.

[stock.py](stock.py) streams the existing emitter's layer dictionaries into the
ordinary stock model. It changes materialization only. A zero PAD forward and
the constructor's initial snapshot injection precede unmodified
`Qwen3_5ForCausalLM.generate(do_sample=False)` from GO. Read-only forward hooks
inspect cache commits and a streamer compares generated tokens. Neither changes
forward outputs, weights, model inputs, or cache state. No host guest result is
injected after boot. The run uses the installed stock PyTorch convolution and
DeltaNet fallbacks without installing optional kernels.

The first guest's independent trace has 1,416 instructions, 128 NDOT4 operations,
and 1,804 tokens through HALT. Each dot is independently checked with the raw-byte
oracle before the extended reference supplies comparison records. The generated
stock trace matches all 1,804 tokens. At all 1,415 consumed non-HALT record
commits, all three register copies, the exact PC anchor/count, and every one of
the 16,384 RAM bytes match. The model stops when it generates HALT; no extra
forward drains that terminal token into the cache. The guest exercises dots
followed by dependent addresses, stores, immediate loads, and branches.

A fresh boot cache then replays that actual generated sequence in chunks of
16 input tokens. All 1,804 next-token predictions match. Exact comparison of
every linear layer's recurrent and convolution tensors stops at tensor index 0,
the first layer's recurrent state, with exactly one unequal element. The retained
assertion is `('chunk cache mismatch', 0, 1)`. The coordinate and two values were
not saved, and the cause is unknown. No tolerance was applied. The nested run's
`status` remains `running` because it is marked passed only after cache equality;
the report's overall `status: fail` and exit code 1 are authoritative.

The second program, changed-DROM variants, and all five stock corruption controls
were not reached. Their code in [stock_m3.py](stock_m3.py) is unaccepted. The
separate [test_stock_observer.py](test_stock_observer.py) passes extraction from
synthetic snapshots and detects deliberate register, PC and RAM changes. That
unit test is observer evidence only, not autonomous execution or stock controls.
Identical network weights across multiple workloads are not yet established.

```sh
.venv/bin/python experiments/neural-accelerator/test_stock_observer.py
.venv/bin/python experiments/neural-accelerator/launch_stock.py --label FRESH
```

Private stock outputs are under
`C:/Users/Admin/qwenputer-games-work/artifacts/neural-accelerator-m3-20261030`.
The GPU has no Python jobs after the failed worker exits. All work remains
uncommitted and within this experiment directory. No upstream changes,
installations, memory replication, hidden accumulator, new feedback path, or
changes to `experiments/doom-showcase/` or `docs/articles/` were made.

The next gate is a bounded diagnosis of the exact cache difference, followed by
the remaining M3 cases and controls. M4 then needs an actual full-map stock
allocation under the authorized limits and both 8,192-token bounded prefixes.
M5 still needs an aligned stories260K guest rebuilt with NDOT4, a bounded
per-dot comparison, then all 64 integer-engine predictions, every-position logit
hashes, decoded bytes, KV clipping counts, and final architectural state checked
bit for bit. Attention remains scalar.

Model the Domain shaped the fixture-bound reports and explicit commit observers.
Prove It Works required independent actual-token comparisons and retained the
negative cache gate. The decision trail is [decisions.tsv](decisions.tsv).
Work ran inline. No independent agent review was performed.

## 2026-10-30 exact counter fix, M3 and bounded M4 stock checks pass

The original mismatch is a readable counter value. It is neither signed zero
nor unreachable state, so the documented exception cannot pass it. The new
opt-in stock counter construction fixes the mismatch exactly. No tolerance or
exception was used. [stock-cache-diagnosis.json](stock-cache-diagnosis.json)
summarizes the retained negative and fixed evidence.

[stock-cache-accelerator.json](stock-cache-accelerator.json) reproduces the
original final 16-token-replay difference at zero-based layer 0, value head 2,
key head 1, key row 0, value column 2, table `count`. Cached is 4.0,
`0x40800000`; chunked is 4.000028610229492, `0x4080003c`. This is the only
unequal final element. Across all boundaries, the differing cells are row 0,
column 2 of `count`, `j` and `vcnt`, at value heads 2, 4 and 6. Their queries
read both rows 0 and 1 on every non-reset token and project column 2. Reference
ballast occupies columns 0 and 1. No padding cell, unused key row, convolution
cell or signed-zero difference explains the failure.

[cache_diagnosis.py](cache_diagnosis.py) records every boundary's coordinates,
values and raw bits while replaying only inputs actually consumed by ordinary
stock greedy `generate()`. Chunk sizes 1, 2, 4, 8, 16 and 64 are checked. The
first unequal fixed boundaries are respectively none, 2, 4, 8, 32 and 64 inputs.
GO is input 1. After GO and V(0), at input 2, `j` and `vcnt` are cached 64.0,
`0x42800000`, versus chunked 64.00009155273438, `0x4280000c`. With fixed
16-token chunks, boundary 16 is exact; refinement of the second block finds
input 17 exact and input 18 unequal. Prefixes 19, 21 and 24 become exact again.
The saved bisection probes and enumeration of prefixes 17 through 32 expose
this non-monotonic equality. Bisection alone would miss transient differences.

The original circuit control in [stock-cache-original.json](stock-cache-original.json)
replaces every dot with four scalar LB/LH/MUL/ADD groups. It generates all 4,620
tokens and checks all 3,335 non-HALT commits. All six chunk sizes are checked;
the same readable counter differences occur. The CPU control in
[stock-cache-scalar-reference.json](stock-cache-scalar-reference.json) establishes
equivalent final RAM, DROM and every register except the scalar implementation's
t3/t4 scratch registers. The preliminary add-only 64-token control is retained
as [stock-cache-original-probe.json](stock-cache-original-probe.json), but is not
used as evidence of scalar guest equivalence.

The installed stock-kernel isolation in
[counter_kernel_diagnosis.py](counter_kernel_diagnosis.py),
[counter-kernel-cpu.json](counter-kernel-cpu.json) and
[counter-kernel-gpu.json](counter-kernel-gpu.json) reproduces the mechanism without
an accelerator. CPU and GPU fallback source hashes agree. Cached recurrence
applies each decay separately. `exp(-softplus(0))` rounds to exactly 0.5 and
`exp(-128)` underflows to zero. Chunk mode uses fp32 cumulative log decays,
subtraction, exponentiation, triangular solves and matrix reductions. After
-128 and -0.6931471824645996, the cumulative value is -128.69314575195312.
Its pairwise exponential is 0.5000007152557373, `0x3f00000c`. This explains the
two-input difference exactly. Cumulative log-half rounding also differs without
the large reset. One-hot key normalization and saturated beta are exact here;
this is not an inexact dot-product value or a write gate that failed to saturate.
Query normalization affects the read, not this state update. No unchanged-token
argument is used to declare these readable cells harmless.

Removing reset decay was rejected because cumulative halving still differs.
Moving halving to beta=0.5 was rejected because chunk summation cancels the live
value to zero at lengths 32 and 64. These isolated synthetic-channel checks are
primitive evidence, separate from autonomous generation.

[integer_counters.py](integer_counters.py) supplies the accepted stock-only
variant. Row 0, column 2 stores `181*n`; both reference columns store
`181*2^17`. Reset writes zero with beta=1, idle uses beta=0, and a step uses
beta=`2^-44` with value `181*2^44` and no log decay. The bf16 step logit -30.5
produces exactly that beta on both installed CPU and GPU kernels. Independent
reset/step embedding features keep every emitted projection coefficient bf16
representable. The host only loads the initial counter state.

For the tested count range, fp32 subtraction of the small state from the huge
step value rounds back to that value, so each actual stock update adds exactly
181. Reset cancellation and the chunk reductions remain exact. The reference
scale makes the bf16 attention read round to power-of-two reference values and
`16*n` data values; the stock gated norm and existing output projection then
return exactly n. [test_integer_counter_stock.py](test_integer_counter_stock.py)
checks 38 sequences with initial counts 0 and 63, resets, idle steps and seeded
patterns, against independent integer counts, at all six chunk sizes. Both
[CPU](integer-counter-cpu-v2.json) and [GPU](integer-counter-gpu-v2.json) pass.
The stock normalized read is separately exact for every count 0 through 127,
covering the valid PC and record-counter ranges.

`constructor.build(..., stock_integer_counters=True)` selects the variant;
stock M3 and M4 select it explicitly. The legacy constructor default remains
available. The frozen IR evaluator assumes saturated beta gates and cannot
evaluate this new primitive. No IR checker was weakened and no new compact
backend equivalence is claimed. Supporting the new encoding in IR and compact
execution is an open follow-up. Prior M2 IR evidence applies to the legacy
counter variant. The new stock result is established by its independent
integer primitive checks and full autonomous stock checks.

[stock-cache-fixed.json](stock-cache-fixed.json) reruns the full original trace
with the fix. All 1,804 tokens and 1,415 cached commits match. Every recurrent
and convolution word matches bit for bit after every boundary at all six chunk
sizes. Prefix refinements also match. This is the root-cause-fixed acceptance
case; no signed-zero or unreachable-state exception, and no paired continuation
to justify such an exception, is used.

[stock-m3-fixed.json](stock-m3-fixed.json) and its
[exit report](stock-m3-fixed-exit.json) complete M3 with one unchanged model.
Each workload retires 1,416 dependent instructions and includes 128 dots,
dependent addresses, stores, immediate loads and branches. Program 0 uses SW/LW;
program 1 uses SH/LH. Initial DROM bytes change from 7 to -13 with identical
weights. Every commit compares all register copies, PC anchor/count and every
one of the 16,384 RAM bytes. Generated HALT is not drained into the cache.

| Program | First DROM byte | Generated tokens | Checked commits | Exact cache replay |
| --- | --- | --- | --- | --- |
| 0 | 7 | 1,804 | 1,415 | pass |
| 0 | -13 | 1,804 | 1,415 | pass |
| 1 | 7 | 2,332 | 1,415 | pass |
| 1 | -13 | 2,316 | 1,415 | pass |

All five stock controls fail at the independently predicted first token
boundary, indexed from zero. Memory is detected at 10, accumulator at 9, PC at
0, disabled write gate at 16 and token-family bit at 0. Positive runs retain
zero raw-bit cache differences. There are 8,256 checked generated tokens,
5,660 commit checks and 512 independent raw-byte dot checks across the four
runs. These stock controls are distinct from synthetic observer tests.

The requested M4 stock checks pass in [stock-m4.json](stock-m4.json), with
[exit code zero](stock-m4-exit.json). [stock_m4.py](stock_m4.py) verifies the
passed M3 report and its implementation hashes before allocating. The generic
map is 64 KiB RAM and 4 MiB DROM at the unchanged addresses, with the original
256 KiB code window. The graph has 64 chip layers, 2,113 physical dimensions,
627 key heads and 68,923 FFN units. The stock model has hidden width 2,304,
16 key / 32 value heads per linear layer, 128-dimensional heads, 9,216 MLP
units and the appended zero full-attention layer. Its 7,115,541,760 parameters
require 14,231,083,520 bf16 parameter bytes. Every parameter is materialized on
CUDA; there is no offload.

These are actual PyTorch CUDA allocator measurements from construction and
both bounded guests in one completed run. They exclude other applications and
CUDA driver allocations; global free memory is separately guarded and recorded.

| Scope | Allocated bytes | Reserved bytes |
| --- | --- | --- |
| After model materialization | 14,231,158,272 | 14,547,943,424 |
| Completed-run peaks | 14,488,735,232 | 15,061,745,664 |

The peaks are 13.494 / 14.027 GiB, below the 15 GiB allocator cap. Global free
VRAM immediately after materialization is 1,060,372,480 bytes. The worker checks
a 512 MiB free-memory threshold. Tasklist and NVIDIA process inventory were
retained; only desktop graphics applications were present alongside this job.
There was no concurrent compute job. No Python worker remains after exit.

| Full-map guest | Autonomous tokens | Checked commits | Independent dot checks | Result |
| --- | --- | --- | --- | --- |
| Small ndot loop | 8,192 | 6,433 | 585 | exact |
| Original stories260K ELF prefix | 8,192 | 5,632 | 0 | exact |

Both guests use the same materialized weights and native greedy `generate()`.
The small guest increases the ordinary guest loop count to 2,048; it does not
specialize model weights. Every observable commit checks all register copies,
PC and all 65,536 RAM bytes. The final generated token is not secretly drained.
[verify_stock_m4.py](verify_stock_m4.py) independently regenerates both bounded
prefixes on the CPU and compares the exported stock token files byte for byte.
[stock-m4-validation.json](stock-m4-validation.json) passes. The stories prefix
also matches the first 8,192 tokens of the retained original legacy trace,
whose complete historical hash is verified. The prefix token hashes are
`fee6f01606bb8eaa7d68e70583ecc6d939e11787ac80e824670b0608a1c77bdb`
for ndot and
`cc73b4f82c12660cf964528dcfb251a5edc0af06101d597a129d542d9dbf2421`
for stories. These are CPU-model protocol tokens, not 8,192 LLM predictions.
No full integer-LLM quality or speed claim follows.

The first full-map attempt in [stock-m4-guard-stop.json](stock-m4-guard-stop.json)
stopped after 6,655 exact tokens and 5,226 checked commits. Global free VRAM
fell to 533,987,328 bytes, below 536,870,912. It was a guard failure, not OOM or
a numerical mismatch; [its exit code](stock-m4-guard-stop-exit.json) is one.
Its nested run still says `running`; the overall `fail` and exit code are
authoritative. At that point 644,463,104 bytes were reserved but unallocated.
The successful retry starts from boot and releases only unused CUDA allocator
blocks when the guard needs headroom, then checks the same threshold again.
One reclamation releases 314,572,800 bytes and restores free VRAM from
533,987,328 to 848,560,128. Active weights, recurrent states and KV tensors stay
on GPU. No model-cache clearing, injection or offload occurs. Peak counters are
not reset by reclamation. This resource-management change is recorded in the
report and does not replace any model or generation operation.

The failed interrupted diagnosis in
[stock-cache-interrupted.json](stock-cache-interrupted.json) and its
[exit report](stock-cache-interrupted-exit.json) is also retained. Copying a live
Windows report briefly locked the target during atomic replacement.
`stock_m3.save()` now retries replacement for at most two seconds; its synthetic
reader-lock check passes. Subsequent progress collection uses stdout and waits
for exit before exporting final reports. The first GPU source export required
explicit UTF-8 and was rerun. Two integer-counter construction probes stopped
before generation, on a Python coefficient type and a non-bf16 combined weight;
independent Boolean embedding features fix the latter without rounding weights.
The initial increment-128 normalization candidate has 19 unequal reads in
[integer-counter-normalization-cpu.json](integer-counter-normalization-cpu.json);
the accepted scale 181 has none. Negative evidence was preserved.

[test_stock_observer.py](test_stock_observer.py) checks synthetic extraction,
register/PC/RAM corruption, exact mismatch coordinate and hex-bit retention,
signed zero, and an integer counter at n=17 with a one-unit corruption. These
checks and Python compilation pass. They do not substitute for the five actual
stock corruption controls. No upstream file, excluded directory, installed
package or native opcode executor was changed. No commit was made. Work ran
inline and no independent agent review was performed.

M5 remains open. First model the stock-only counter primitive in IR and the
compact backend with its actual fp32 update and bf16 read contract, then prove
that representation agrees with stock. Rebuild an aligned stories260K ndot
guest, compare each dot in a bounded prefix, and compare all 64 integer-engine
predicted IDs, every-position logit hashes, decoded bytes, KV clipping counts
and final architectural state bit for bit. Attention remains scalar. The two
bounded M4 prefixes do not replace that regression or establish new compact
backend triangulation for these counter weights.

Reproduction uses fresh labels and the existing environments.

```sh
.venv/bin/python experiments/neural-accelerator/launch_stock.py \
  --label FRESH-diagnosis --diagnosis accelerator --integer-counters --tokens 1804
# Wait for diagnostic exit before starting M3.
.venv/bin/python experiments/neural-accelerator/launch_stock.py --label FRESH-m3
# Only after that report passes, use its remote path for M4.
.venv/bin/python experiments/neural-accelerator/launch_stock.py \
  --label FRESH-m4 --m4-report REMOTE_PASSED_M3_REPORT \
  --stories .cache/tiny-llm/neural-stories/engine.elf
.venv/bin/python experiments/neural-accelerator/verify_stock_m4.py \
  --actual EXPORTED_M4_RUN_DIRECTORY \
  --legacy-inputs .cache/neural-accelerator/legacy-neural-inputs-20261030 \
  --out FRESH_VALIDATION_DIRECTORY
.venv/bin/python experiments/neural-accelerator/test_stock_observer.py
```

The launcher syncs with tar over `ssh gpu` and detaches encoded PowerShell with
`Invoke-CimMethod Win32_Process Create`. Private outputs are below
`C:/Users/Admin/qwenputer-games-work/artifacts`, in
`neural-accelerator-m3-diagnosis-full-retry-20261030`,
`neural-accelerator-m3-diagnosis-original-scalar-20261030`,
`neural-accelerator-m3-integer-counter-full-20261030`,
`neural-accelerator-m3-integer-counters-m3-20261030`,
`neural-accelerator-m4-integer-counters-full-map-20261030` and
`neural-accelerator-m4-integer-counters-reclaim-20261030`.
First-difference `.pt` cache pairs remain in the diagnosis directories. Source
snapshots and the completed M4 raw token files are retained privately. Local
raw token exports are under `.cache/neural-accelerator/m4-actual`.

Model the Domain shaped explicit state coordinates and the counter encoding.
Fix Root Causes required changing the readable counter instead of waiving it.
Prove It Works required actual GPU generation, raw-bit cache comparisons and
independent token-file checks. Explain the Number kept memory claims tied to
measured allocator peaks in one completed run. The append-only trail is
[decisions.tsv](decisions.tsv).
