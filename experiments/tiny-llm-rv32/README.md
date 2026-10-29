# Tiny LLM in the software RV32 machine

This Phase 0 experiment runs an integer transformer in the existing software
reference CPU on the Mac. It does not execute a neural circuit, use a GPU,
implement an accelerator, or establish the stock Qwen layer return path.
The CPU, instruction semantics, and memory map are unchanged.
Phase 1 neural execution is reported separately in [tiny-llm-neural](../tiny-llm-neural/README.md).
The custom-model integration below extends this engine without changing the CPU.

## Results

The complete 64-position reference run matches native output byte for byte.
It executes 162,208,013 instructions and emits 242,609,345 trace tokens,
including output and diagnostics. It halts with exit status zero and no fault.
Observed stack use is 272 bytes. The ELF contains 5,920 code bytes,
46,272 writable static bytes, and a 299,835-byte model blob.

Across three native prompts, free-running greedy agreement with float is
170/173 generated predictions (98.27%). Common-history agreement is 172/173
(99.42%). The first prompt differs at position 61. The other two match for
all generated predictions. KV clipping remains zero in all three runs.
The native ASan and UBSan run completes without diagnostics and matches the
optimized native output. The float probe's decoded text also matches the
unmodified original CLI at temperature zero for each prompt.

Per-position counts below cover inference. Times are projections at the
supplied 4,380 trace tokens/s, not measured neural timings.

| Position | Phase | RV32 instructions | Trace tokens | Dot loop instruction share | Minutes |
|---|---|---:|---:|---:|---:|
| 0 | Prefill | 2,122,366 | 3,193,567 | 85.53% | 12.15 |
| 4 | First generated prediction | 2,179,478 | 3,286,637 | 83.29% | 12.51 |
| 32 | Decode | 2,555,224 | 3,829,821 | 71.04% | 14.57 |
| 63 | Decode | 2,926,052 | 4,341,698 | 62.04% | 16.52 |

The four prefill positions average 2,143,737 instructions and 3,230,982.5 trace
tokens. The 60 decode positions average 2,560,038 instructions and 3,827,383
trace tokens, or 14.56 projected minutes per prediction. Whole-run totals
average 3,790,771 trace tokens per position across prefill and decode.
There are 259,328 scalar matrix MACs at every position. The profiler checks
that the compiled seven-instruction inner loop retires exactly that many MACs.
Its fixed cost becomes a smaller share as attention grows.

| Hypothetical memory dot width | Minutes at position 4 | Minutes at position 63 |
|---|---:|---:|
| 4 | 3.74–3.99 | 7.79–8.04 |
| 8 | 2.89–3.01 | 6.93–7.06 |
| 16 | 2.45–2.51 | 6.50–6.56 |

The assumptions for these estimates appear below. Per-position opcode counts,
instruction-mix categories, matvec shares, hashes, and projection arithmetic
are in the evidence JSON. [The chart](evidence/performance.html) plots every
position, including fluctuations caused by value-dependent instruction paths.

For example, the first integer sample begins "Once upon a time, there was a
little girl named Lily. She loved to play outside in the park." The second
begins "One day, a little girl named Lily went to the park with her mom."
The third begins "There was a big dog named Max." Complete integer and float
samples and their token records are retained in the evidence JSON.

## Method

`pack.py` reads the legacy llama2.c checkpoint as little-endian data. It does
not load the PyTorch pickle or run downloaded Python. It checks geometry,
file lengths, finite weights, and memory budgets. The ROM image contains
int8 matrices, a signed Q24 scale per output row, Q12 RMS weights, Q15 RoPE
sin/cos pairs, lookup tables, and tokenizer byte strings. `pack.json` records
input and output SHA-256 hashes. The classifier shares the embedding matrix
when the checkpoint header requests it. Separate classifiers also work.

Per-row scales require one rescale per output row. Group scales would require
more rescaling inside every dot product. The measured quality comparison
supports per-row scales for stories260K. This choice needs another quality
check for a different trained model.

`engine.c` is the same integer engine in both builds. Activations use signed
Q12 int32. Each matvec dynamically shifts its input into an int16 scratch
vector, with an accumulation bound derived from its width. Products accumulate
in int32, then an int64 multiplication applies the row scale. RMSNorm uses an
integer square root and integer division, with epsilon rounded to Q24.
RoPE multiplies Q12 values by ROM Q15 sin/cos pairs. Attention uses int16 Q10
keys and values, integer dot products, a Q20 exp table over [-16, 0] with
1/128 spacing, and Q15 normalized probabilities. SiLU uses a sigmoid table
over [-16, 16] with 1/128 spacing. Values beyond the table interval use the
nearest endpoint. These are approximations to float inference.

The guest selects the lowest token ID among tied maximal logits. It decodes
the packed tokenizer vocabulary, including raw byte pieces and BOS leading
space removal, and emits bytes with `ecall a7=1`. Prompt tokenization is a
host input preparation step. No activations, KV state, logits, or predicted
tokens are supplied by the host. Generation is fixed length for measurement.
There is no sliding window or stochastic sampler.

The selected context cap is **64 positions, including the prompt**. KV storage
uses 40,960 bytes. Scratch arrays use 5,304 bytes before a few scalar globals
and alignment. The original Phase 0 build reserved the bottom 256 RAM bytes and the top
8,192 bytes for the stack. Current builds reserve 1,024 stack bytes and require
a further 1,024-byte margin through both packer and linker checks. A larger context requires a rebuild and must pass
both the packer and linker gates. The checkpoint's advertised 512 positions
do not fit this cache in 64 KiB.

The compiler uses RV32IM with the ilp32 ABI and no compressed, floating point,
or Zba instructions. The existing reference and circuit also support Zba,
but this experiment does not enable it. The reference package preparation
reuses `experiments/memory-map/prepare.py` at 64 KiB RAM and 4 MiB data ROM,
exactly as the Doom experiment does. It copies into `.cache`, checks the pinned
upstream revision and clean status, and leaves upstream untouched.

## Reproduce

Run these commands from the repository root. An existing local Zig installation
and Clang are required. No system installation is performed. The default Zig
path is the existing `.venv/lib/python*/site-packages/ziglang/zig` binary.
Pass `--zig` to `build.py` to use another existing binary.

```sh
python3 experiments/tiny-llm-rv32/build.py
python3 experiments/tiny-llm-rv32/verify.py
.venv/bin/python experiments/tiny-llm-rv32/run_reference.py
python3 experiments/tiny-llm-rv32/report.py
```

The first command packs the model and builds native and guest executables.
The second copies and compiles the downloaded llama2.c `run.c`, runs its
original CLI at temperature zero, and checks that the float probe emits the
same text. It compares three 64-position native runs, both free running and
with float evaluated on the integer engine's history. Prompt predictions are
excluded from quality agreement. The original C source remains unchanged.

The reference command runs all 64 positions from ELF initial state through
HALT. It requires exact byte-for-byte equality with native output, including
token IDs, per-position logits hashes, KV clipping counts, and decoded text.
Each hash folds every signed Q12 logit in vocabulary order with unsigned
FNV-style recurrence `(hash ^ word) * 16777619`, starting at 2166136261.
It also records stack low water and faults. A budget exhaustion is a failure.

Generated binaries, copied C source, ROM images, machine scratch packages,
native logs, and raw reference output remain under `.cache/tiny-llm/`.
The checked-in-size textual result is [evidence/evidence.json](evidence/evidence.json).
The working decision trail is [decisions.tsv](decisions.tsv). No commits are made.

## Interpret the counts

Per-position inference counts start at the guest `position_begin` function
entry and stop at `position_end`. They include the begin marker and exclude
text decoding and diagnostic output. Whole-run totals include those costs.
Positions before the last prompt input are labeled prefill. The last prompt
input produces the first generated prediction and is labeled decode.

The profiler sums actual `Machine.record()` token lengths, including refresh
records and variable-width value and division records. It excludes neural
prefill and the GO token. It identifies the compiled dot loop from its entry
label through the backward branch, including loads, multiply, accumulation,
pointer increments, and the branch. It separately measures all of `matvec`.
Instruction counts and mix are deterministic workload counts. Emulator wall
time is not a neural performance measurement.

`report.py` divides measured trace tokens by 4,380 tokens/s and by 60 to
estimate minutes per LLM prediction. This assumes that the supplied fused
executor rate transfers to this workload and excludes setup. It is not a
stock Hugging Face timing estimate.

The width 4, 8, and 16 projections replace only the measured dot loop. Each
group hypothetically uses one memory-reading int8-by-int16 accumulate
instruction, two pointer increments, and one branch. The dot result costs
two to three trace tokens, each pointer result costs two, and the branch one.
Other work retains its measured cost. This assumes the accelerator reads both
operand vectors without extra guest load instructions. No such instruction,
latency, memory interface, or autonomous stock-layer return path is proven.
The existing return-path limitations in `experiments/one-instr-design/DESIGN.md`
still apply to later phases.

## Swap a model

`build.py --model PATH --tokenizer PATH --context N --steps N` accepts another
legacy llama2.c checkpoint. Geometry and offsets are generated into headers,
so swapping geometry requires recompilation, not edits to engine loops.
The vocabulary must use the llama2.c tokenizer format with BOS ID 1 and byte
fallback starting at ID 3. Wider activation ranges need fresh quality and
integer-range verification. Generality of the loops is not a guarantee that
every checkpoint's ranges are safe or its text quality survives quantization.

For a shared-classifier model with dim 128, hidden 344, eight layers, eight
query heads, four KV heads, and vocabulary 1024, the projected parameter count
is 1,583,232. Weights, norm values, and scales need about 1,632,256 ROM bytes,
before tables and vocabulary. This leaves substantial space within 4 MiB.
At context 16, KV storage is 32,768 bytes and scratch is 10,160 bytes. Including
the 8 KiB stack reservation and 256 reserved bytes gives a 51,376-byte budget.
Context 22 projects to 63,688 bytes, while context 23 exceeds 64 KiB.
These are budgets before a few scalar globals and linker alignment.
With two KV heads instead, context 32 has a 51,440-byte budget and context 45
has a 64,804-byte budget. The custom model tested in Phase 1 has different geometry. Its results follow below.

## Scope of the Phase 0 evidence

The evidence establishes bounded native quality and software RV32 feasibility
for stories260K. It does not establish neural execution speed, accelerator
correctness, stock Qwen compatibility, or quality of the model being trained.
Only one prompt has full reference/native verification. Three prompts have
native float comparisons. This implementation is an experimental numeric
format, not a general untrusted-model runtime or a production inference engine.

The workflow applies Model the Domain through generated geometry and offsets,
and Prove It Works through actual guest/native output comparison. Independent
review is not claimed. This implementation task runs inline under the supplied
AGENTS.md instructions.

## Custom model integration

The local custom checkpoint has dim 192, hidden 512, six layers, 12 query heads,
one KV head, vocabulary 1024, and a trained sequence limit of 256. Its integer
ROM is about 2.53 MB. The CPU and 64 KiB RAM / 4 MiB data-ROM map are unchanged.

The selected cache is int16 Q10 at **129 positions including the prompt**.
Writable static storage uses 63,120 bytes, including 49,536 KV bytes. The bottom
256 reserved bytes and 1,024-byte stack reservation bring the budget to 64,400
bytes, leaving 1,136 bytes beyond the reserved stack. The packer and linker
require at least 1,024 margin bytes. Context 130 fails the packer budget.

An int8 cache fits 255 positions with the same margin rule. It uses 63,040
static bytes and leaves 1,216 bytes after the stack reservation. Context 256
passes the payload estimate but fails the linker because globals and alignment
consume the remaining margin. The actual linked ELF is the final memory gate.

Three 129-position prompts contain 362 generated predictions after prefill.
Int16 Q10 matches float greedy predictions in all 362 cases, both free-running
and on common history, with no KV clipping. Int8 Q4 and Q3 each match 335/362
on common history. Free-running agreement is 74/362 and 64/362 respectively.
Q4 clips 938 values across the three runs. Q3 has no clips. Q2 drops to 274/362
on common history and 32/362 free-running. These bounded greedy comparisons
support retaining int16. They do not measure held-out integer log loss or
longer-context quality. See [custom-quality.json](evidence/custom-quality.json).

All 12 prompt/format combinations pass AddressSanitizer and UBSan, and their
output bytes match the optimized native builds. The unchanged original float
CLI matches the float probe for all three prompts. The int8 Q3 five-position
RV32 run also reaches clean HALT with byte-exact native output and measured
320-byte stack use. See [custom-checks.json](evidence/custom-checks.json) and
[custom-int8-prefix.json](evidence/custom-int8-prefix.json). The default
stories260K native output remains byte-exact with the original Phase 0 sample.

<!-- custom-results:start -->
The full 129-position RV32 run reaches clean HALT and matches native output
byte for byte, including all argmax IDs, logits hashes, decoded text, and clipping
counts. It retires 2,647,019,627 instructions and emits 3,935,538,680
trace tokens. Observed stack use is 320 bytes. The 1,024-byte stack
reservation is 3.2 times that observation. Lowest SP remains
1,840 bytes above static storage. The final margin-gated build
has identical loadable bytes to this full-reference ELF in
[custom-reproduction.json](evidence/custom-reproduction.json).

There are 2,445,312 scalar matrix MACs per position. The profiler verifies the
actual seven-instruction dot loop retires exactly that many MACs at every
position. Decode averages 20,594,052 instructions
and 30,611,156 trace tokens per prediction.

| Position | Instructions | Trace tokens | Projected minutes at 4,380/s | Projected minutes at measured stories260K rate |
| --- | ---: | ---: | ---: | ---: |
| 0 | 18,117,393 | 27,072,823 | 103.02 | 117.37 |
| 4 | 18,270,245 | 27,447,249 | 104.44 | 119.00 |
| 32 | 19,333,153 | 28,875,858 | 109.88 | 125.19 |
| 64 | 20,552,363 | 30,574,844 | 116.34 | 132.56 |
| 128 | 22,840,725 | 33,613,185 | 127.90 | 145.73 |

The instruction and trace counts are measured. Both GPU-minute columns are
projections for the custom model. No custom-model neural run is claimed.
See [custom.json](evidence/custom.json) for every position and the timing scope.
<!-- custom-results:end -->

```sh
python3 experiments/tiny-llm-rv32/custom.py
python3 experiments/tiny-llm-rv32/custom_checks.py
python3 experiments/tiny-llm-rv32/build.py \
  --model .cache/tiny-llm/custom/model.bin \
  --tokenizer .cache/tiny-llm/custom/tok1024.bin \
  --context 129 --steps 129 --stack-bytes 1024 \
  --out .cache/tiny-llm/custom-selected
python3 experiments/tiny-llm-rv32/run_reference.py \
  --out .cache/tiny-llm/custom-selected --max-instructions 4000000000
python3 experiments/tiny-llm-rv32/custom_report.py
```

`--kv-bits 8 --kv-frac 3 --context 255` selects the unclipped int8 alternative.
The default KV format remains int16 Q10. Changing KV representation changes
numeric approximation only. It adds no instruction, accelerator, or new CPU
state. Artifacts and copied original sources remain under `.cache/tiny-llm/`.

The Phase 1 decision trail is [tiny-llm-neural/decisions.tsv](../tiny-llm-neural/decisions.tsv).
