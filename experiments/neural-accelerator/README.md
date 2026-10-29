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
