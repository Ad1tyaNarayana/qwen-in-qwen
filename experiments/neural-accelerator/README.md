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
