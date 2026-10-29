# Residual reuse, 2026-10-29

The constructor retains ordinary Qwen3.5 MLP and DeltaNet sublayers, token-only
feedback, and the original recurrent tables and memory map. Only current-token
scratch dimensions share physical residual columns. `reuse.py` records each
logical lifetime and handover in `liveness-w4.json` or `liveness-w8.json`.

## Alternatives considered inline

The existing emitter allocator combines an old-value subtraction with a new
MLP output. An exact subtraction by itself does not prove that `new-old` is
BF16-exact. For example, 512 and 1 are BF16 values, but -511 is not. We do not
use that allocator here.

A separate clear sublayer avoids that mixed-value output. An initial candidate
required the last reader strictly before the clear sublayer. It produced 2,332
dimensions and hidden size 2,560 for W=4. All five controls and the fault cases
reached passed before an intentional interruption. It is not a completed gate.

The selected schedule allows the last MLP readers in the clear sublayer itself.
Every MLP unit reads the same sublayer input, and the clear occurs in its down
projection and residual update. The new writer is in a later sublayer. In the
IR, original readers precede the appended clears at the same level. No reader
observes a partially cleared residual. This gives 2,113 dimensions and hidden
size 2,304 in the constructed W=4 graph. Numerical acceptance is recorded
separately by the full M2 run.

## Clear and history proof

For old residual value x, the clear unit is `-x * relu(ONE)`. Its emitted gate
is 128, its up input is x/128, and its output is -x. Multiplication and division
by these powers of two preserve the supported BF16 values. The old residual
plus -x is exactly zero before a later writer adds its own output. The full
suite checks this unit and its residual update with `check=True`. Tiny counter
outputs are pinned because their below-threshold values have a separate
rounding allowance and must not enter this argument.

The allocator requires the last write strictly before the clear. It counts all
reads in q, k, v, decay, output gates, beta, and MLP expressions. It pins every
source of a nonzero convolution lag, all output code dimensions, and every
retained probe. Consequently, no reused dimension supplies a lag read at all.
Recurrent state, convolution channels, tap sequences and histories are unchanged.
Every new token initializes its residual from its own embedding as before.
The allocation changes no cross-token state representation.

The physical graph runs in sublayer order. Every write adds to the residual,
including writes to initially zero slots, so IR evaluation exercises actual
clear and reuse semantics rather than overwriting stale values in the host.
Per-level clear units count against the unchanged intermediate-size budget.

## Depth

Both routed operands of a disabled lane are zero. A zero lane product therefore
needs no second enable mask. Removing those masks saves one chip layer and
128 W=4 or 256 W=8 FFN units before allocation. The balanced reduction and final
carry-lookahead additions retain their dependency order. Independent nodes
remain packed by the inherited sublayer scheduler.

The throughput checkpoint remains after M3. These changes establish no stock
generation or timing result. Model the Domain shaped the explicit lifetime and
handover records. Prove It Works requires the physical graph to pass the full
numerical suite and all corruption observers.

## 2026-10-30 depth audit

[depth-schedule.json](depth-schedule.json) records the actual logical schedules.
Activation magnitudes finish by sublayer 30 while weight magnitudes finish at
96. All lane nibble products share sublayer 98 in both widths. Independent
activation work therefore already overlaps the DROM reads, and independent
lane products already share one MLP sublayer.

Normalization finishes at 106 and signed lane results at 112. W=4 reduction
levels finish at 116 and 120, accumulation at 124, and serialization at 128.
W=8 adds a third reduction level at 124, then accumulation at 128 and
serialization at 132. These are dependent two-stage carry-lookahead adds.
Further signed-column or carry-save restructuring needs another bounded
arithmetic proof. This task retains the tested carry gadgets. The resulting
64 and 66 layers still exceed the original provisional depth estimates.
