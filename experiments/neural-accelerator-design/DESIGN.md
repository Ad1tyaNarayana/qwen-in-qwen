# Generic neural dot-product extension

Design investigation, 2026-10-29. Branch `tiny-llm`. Compiler pin
`ca13310e2f9f788d1dd3feeea8213c1b0c23913c`. No upstream changes or commits.

## Recommendation

Build a bounded **`ndot8.s8s16.acc`** prototype first. It reads eight signed
bytes from DROM and eight signed halfwords from RAM, adds their dot product to
`rd`, and returns the low 32 bits through the existing value-record protocol.
Also implement width four in the same datapath for tails. Consider width 16
only after measuring the width-eight graph. Keep pointers in ordinary guest
registers and increment them with ADDI. Preserve the existing address map.

This is a generic ISA extension inside the compiled Qwen3.5 network. Model
weights remain DROM state, not compiled coefficients. A fixed accelerator
network handles changed programs, model weights, shapes and input data. The
host loads initial images and observes output; it never calculates a dot,
activation, accumulator, branch or return token.

Use bit/nibble arithmetic rather than a scalar floating-point dot. Retain
ordinary generated F0/F1/F2 records. Do not depend on STEP, tensor feedback,
host-selected lanes, a custom DeltaNet kernel, or an unproved hidden accumulator.
This avoids the unresolved late-to-early feedback problem in the one-instruction
design. It does not make an instruction take one transition.

For attention, compare a second, more expensive build with
**`ndotN.s32s16.lo` / `.hi`**, two RAM inputs and a 64-bit sum returned one
32-bit half at a time. Include `.s16s16` and `.u16s16` modes in that shared
arithmetic unit only if their emitted cost is small. QK uses int32 Q, not int16
Q in the current guest. Leave the strided weighted-value dot scalar initially.
A generic strided instruction is possible with serial read rounds, but a
one-transition arbitrary gather requires many replicated RAM ports. Neither
attention option is free. The sensitivity estimates favor the matrix-only
width-eight build as the first implementation.

Expected first-build budget is **zero added logical memory heads, 10–16 added
chip layers, 10–24k FFN units and 1.5–3k residual dimensions** for width eight.
These are engineering envelopes, not compiled measurements. A dual-RAM,
full-width attention build budgets **64 extra key / 128 value heads and 18–28
added layers in total**, including its four RAM-copy layers. Stock-model
geometry and numerical gates must pass before either is called implemented.

## Evidence and current execution

Sources read for this design are [one-instruction design](../one-instr-design/DESIGN.md),
[integer engine report](../tiny-llm-rv32/README.md), its
[counts](../tiny-llm-rv32/evidence/evidence.json),
[training report](../tiny-llm-train/README.md),
[memory map](../memory-map/README.md), and the
[handoff boundaries](../../docs/doom-24fps-handoff.md#boundaries-for-proposed-optimizations).
The boundaries allow generic vector ISA extensions only when the neural circuit
implements the arithmetic and memory semantics.

The pinned [spec.py](../../.upstream/qwenputer/experimental/riscv/chipformer/rv/spec.py)
defines records and result transport.
[core.py](../../.upstream/qwenputer/experimental/riscv/chipformer/rv/core.py)
defines decode, three register read copies, memory and five-stage MUL.
[q35/ir.py](../../.upstream/qwenputer/experimental/riscv/chipformer/q35/ir.py)
defines exact bounded SwiGLU and DeltaNet operations.
[rv/hf.py](../../.upstream/qwenputer/experimental/riscv/chipformer/rv/hf.py)
emits stock `Qwen3_5ForCausalLM`, injects the initial snapshot and compares stock
cached decode or prefill with the IR. Its helper checks supplied input tokens;
it is not by itself an autonomous `generate()` test.

The expanded CPU has 44 chip layers, 2,679 residual dimensions, hidden size
2,816, 42,014 FFN units, 627 key / 1,247 logical value heads and 81,723,392
logical state bytes. DROM uses 512 key / 1,024 value heads, RAM uses 64/128,
and code uses 32 key heads. Each chip layer has room for 16 key / 32 value
heads and 9,216 FFN units. Most depth comes from the 32 DROM sublayers.
Stock heads have 128-dimensional keys/values and a four-tap convolution.

At an instruction boundary the current input token commits the previously
computed result. The network then computes its next token. Register memory
comes before RAM and DROM reads. Its write values arrive in token payloads;
its destination comes from the previous decoded instruction through a conv tap.
The existing `regs.rd` read copy already supplies the old destination, so an
accumulating R-type operation does not need a fourth register read port.

`generate()` can feed a late result back through the emitted token and its
embedding. It cannot feed an arbitrary late residual tensor to an earlier
layer. Adding layers or a tap to an earlier projection does not change this.
Use zero-prefix initialization and initial neural cache injection as already
specified by the stock contract; do not inject intermediate answers.

## Instruction alternatives

| Design | Architectural operation | Main benefit | Limitation | Decision |
|---|---|---|---|---|
| A. Memory dot and architectural accumulation | `rd += dot(DROM[rs1], RAM[rs2]) mod 2^32` | Removes guest loads, multiplies and most loop control | Needs wider read routing and value records | Recommended first |
| B. Packed registers | e.g. four s8 lanes in rs1 times four s8 in rs2, accumulate in rd; or two s16 lanes | No new memory interface; small bounded arithmetic | Packing/loading dominates; s8 activation quantization changes the engine | Useful arithmetic control, weak workload accelerator |
| C. Hidden accumulation stream | begin, append partial dots, final reduction | Avoids returning partial sums to early registers | Direct recurrent accumulator is unconstructed; bounded partial-dot log is plausible but needs new state/control/exactness proofs | Research only, excluded from estimates |

An s8-by-s16 packed form fits only two products in two RV32 source registers
without extra register operands. Four products require an implicit register
pair or more load/pack instructions. A P-extension-like encoding is inspiration,
not a claim of standard P or RVV compliance. Even a four-s8 form must execute
the original loads and often shifts/masks. Compare it against A with actual
guest traces before expanding the ISA.

### Proposed A semantics

For each compiled maximum width W in {4,8,16}, encode active N in {4,8,16}
with N <= W. Do not enable wider encodings in a smaller circuit.

```
ndotN.s8s16.acc rd, rs1, rs2
  A = old_x[rs1]; B = old_x[rs2]; C = old_x[rd]
  require A % N == 0 and B % (2*N) == 0
  require [A,A+N) wholly within DROM
  require [B,B+2*N) wholly within RAM
  s = sum(sign8(mem[A+i]) * sign16(le16(mem[B+2*i])))
  x[rd] = (C + s) mod 2^32, unless rd == x0
  pc += 4
```

Read every source from the old architectural state, including aliases
`rd==rs1`, `rd==rs2`, and `rs1==rs2`. The latter normally faults for A because
the windows differ. x0 suppresses only the register write, not validation.
There are no memory writes, saturation, scale multiplication or floating-point
rounding. The little-endian load interpretation matches the guest. Check the
entire span with widened address arithmetic before reading; RV32 wraparound
cannot make an invalid span valid. Report misaligned load before load-out-of-range,
using the existing fault protocol; fault PC is the dot instruction and all
architectural writes are suppressed. Reserve undecoded encodings as illegal.

Strong alignment deliberately avoids cross-row DROM reads and RAM lane wraps.
Use scalar cleanup for arbitrary lengths. Do not pad by reading beyond an
object, a row or the region. Width-16 hardware also supports width eight for
the baseline's hidden width 172 tail and eight-wide attention heads. Width-four
cleanup handles the remaining matrix tail. Requiring these alignments can
require aligning guest scratch and individual matrix rows; it changes guest
layout, not the physical memory map. Packer offsets and row padding must be
accounted for in the ROM budget. Existing evidence uses padded-group estimates,
not an already aligned replacement ELF.

Auto-increment choices are (1) ordinary ADDI, recommended; (2) implicit pointer
state initialized from rs1/rs2 with neural advance per dot; or (3) writes to
rs1, rs2 and rd on final commit. Option 3 needs a new multi-destination commit
contract and write transport. Incremented pointers are available early enough
to use local conv taps in a redesigned register path, but this must be built
and proved, including aliases and suppression on faults. It is not three free
writes under the existing protocol. Option 2 adds hidden architectural state,
initialization/reset rules, dependencies and fault handling. The tables retain
two ADDIs and one branch per group, costing five tokens in the present map.

### Attention operations and memory shape

The current engine has two attention reductions.

* QK reads contiguous **int32 Q** from RAM and contiguous **int16 K** from RAM.
  Its sum is int64 before a shift and scale. A generic s16 dot cannot replace it
  without a range proof or an arithmetic change. Use `.s32s16.lo/.hi`, where
  each returns the selected half of the same signed 64-bit dot. Both instructions
  reread unchanged memory. Guest code adds the halves to a 64-bit sum with carry.
* Weighted V reads **int32 probabilities** in [0,32768] and int16 V at
  `((layer*context+t)*KVDIM+kh+j)`. V is strided by `2*KVDIM` bytes. Probability
  32768 is not a signed int16 value. `.u16s16` can apply only after an exact
  conversion to a new halfword scratch array, with its guest instructions and
  RAM budget included. Otherwise use `.s32s16` with a generic stride mode.

A contiguous `.s16s16.lo/.hi` mode is useful to other programs and future
validated Q formats; it is not an automatic optimization of the current QK.
A two-register result could avoid recomputation, but requires a new two-result
record and commit mechanism. Returning two halves in two existing instructions
is the conservative initial attention interface. No hidden state links them.
Intervening stores naturally change the second operation's result, as specified.

The two attention sums each contain L*D*(position+1) scalar MACs. They are
1,600 / 20,480 each at positions 4 / 63 for stories260K and 5,760 / 73,728
for the custom model. Softmax, normalization, lookup tables, division, RoPE and
row rescaling remain guest work.

For the contiguous attention forms, require A alignment to `4*N` for s32
(or `2*N` for s16/u16), B alignment to `2*N`, and both complete spans in RAM.
The result is the low or high 32 bits of the signed 64-bit mathematical sum,
without adding old rd. The u16 mode zero-extends its first operand. The same
alias, x0, fault priority and wraparound checks as A apply. A future stride mode
would need a separately specified stride operand, validation of every lane and
an explicit latency/continuation protocol; no opcode is allocated to it here.

## Read bandwidth and port costs

One transition means one model token position, not a full instruction record.
The current network exposes **one selected 32-bit memory word**, from the
scalar effective address, to the load datapath. The number of physical heads
does not mean it exposes 512 independent arbitrary DROM reads today.

For DROM let `wi=(address-DROM_BASE)>>2`. The key head is `wi>>11`, the value
head is bit 10, the row pair is `p=(wi>>4)&63` and `p+64`, and the two halves
and eight word columns select within a 64-byte block. Both query rows are
active to preserve byte read exactness. `byte_rows()` initializes complementary
data columns in those rows and ballast in both. Each value head exposes 64
distinct byte columns for that row pair; a selected x/row can therefore supply
**up to 64 contiguous aligned bytes** by changing output gates and routing,
without copying DROM or adding key heads. Both value heads can expose 128 bytes
at their separate 4 KiB-spaced locations, not 128 contiguous bytes. An N-byte
aligned dot with N<=16 stays inside one 64-byte block. Retain the existing
two-hot query and per-column norm/ballast contract. New routing must pass the
stock kernel check; do not change q to a single row as a shortcut.

RAM word index `wi=(address-RAM_BASE)>>2` maps to key head
`(wi&15)|((wi>>12)<<4)`, value slot `(wi>>4)&1`, row `(wi>>5)&127`.
Each selected word is bit columns 2..33 plus the merged-store columns 34..65.
Their sum is the current word. Adjacent words stripe across 16 different key
heads, so an aligned 64-byte block can read **16 independent words** from one
row/slot per selected bank in one transition. N halfwords need N/2 heads and
N words need N heads. The normal query and output projection must be changed
to retain and rotate those words into distinct lanes instead of adding all
bank outputs into the same `mem` dimensions. This is extra gates/projection/FFN
work, even when it needs no extra heads. Previous stores must update all words
before those reads, including every partial-store representation.

Thus A needs N DROM bytes and 2N RAM bytes for N=4,8,16, with **no memory
replication**. Its capacity comes from distinct columns and banks, not from
multi-hot queries to unrelated RAM rows. A multi-hot query adds rows together;
it cannot preserve two independent values from the same columns.

Two unrelated RAM operands may select the same key head at different rows.
The two value heads share q/k and are not two independent ports. The conservative
attention design duplicates all RAM, not DROM, adding **64 key / 128 value heads,
8,388,608 logical fp32 state bytes and four 16-key-head layers**. Every store
updates both copies, preserving read-after-write ordering and fault semantics.
The existing register file uses this replication technique. No second DROM
read port is required for A. A genuinely arbitrary second DROM query would add
512/1,024 heads, 67,108,864 logical bytes and 32 attention layers before routing.

For weighted V, baseline KVDIM=32 gives a stride of 16 words. All lanes can
hit the same low-four-bit RAM bank, at different rows. Custom KVDIM=16 strides
eight words, alternating two low banks. Existing two-slot value heads cannot
turn these into independent row reads. Worst-case arbitrary-stride N lanes
need N copies of the second RAM port. Including the first port requires N+1
total copies, adding **64N key / 128N value heads, 8N MiB and 4N layers**.
N=8 would add 512/1,024 heads, 64 MiB and at least 32 layers, before MACs.
Special-casing known model strides in circuit weights would violate generality.

The alternative is generic conflict scheduling in neural control and serial
read rounds, each with a generated continuation record, a lane index and
bitwise operand holding state. Worst case is N rounds plus result records.
That changes the instruction's latency/protocol, not the host executor. Price
it from a compiled schedule. Do not count it as a one-transition gather.
Guest transposition is another option, but costs guest loads/stores, storage
and trace tokens. No transposed cache is assumed here.

## Exact arithmetic and layer placement

Fp32 represents every integer through 2^24 inclusive. An s8*s16 product has
absolute bound 128*32768 = 2^22. Four products have bound 2^24; from N=5
onward arbitrary sums are no longer guaranteed exactly representable. Eight
products have bound 2^25, sixteen 2^26. The worst power-of-two endpoint itself
can remain exact even when nearby sums do not. Adding an arbitrary old rd also
rules out a scalar-fp32 accumulator. BF16 is stricter, with only eight significant
bits. A scalar product being fp32-exact does not make it stock-BF16-exact.

An s16*s16 product has bound 2^30. N products need up to 32+ceil(log2(N))
signed bits, including the positive `(-32768)*(-32768)` case. An s32*s16
product has bound 2^46, and N<=16 gives sum magnitude <=2^50. Signed int64
is sufficient for both attention modes. The current guest uses int64, so
silently truncating an attention sum to 32 bits before shifting is incorrect.

Use the following exact construction, keeping all materialized arithmetic
outputs as bits, nibbles or small carries.

1. Interpret memory bits as signed two's-complement. Form unsigned magnitudes
   with bounded bit adders and a sign bit. INT8_MIN, INT16_MIN and INT32_MIN
   become 128, 32768 and 2147483648 in their unsigned bit vectors, never a
   signed overflowing scalar. Extract base-16 digits in [0,15].
2. Compute each unsigned nibble product `a_i*b_j` in [0,225] with the existing
   MUL pattern, `sum(v * AND(a_i==v) * b_j)` for v=1..15. A SwiGLU unit emits
   `silu(K*b)*(a/K)` with K=128. Predicate gates are positive 256 or sufficiently
   negative to make SiLU exactly ReLU in the pinned arithmetic. Do not use
   ordinary smooth SiLU on unconstrained signed activations as a multiplier.
3. Split each product into lo in [0,15] and hi in [0,14] using bounded threshold
   gadgets, as in `RvCore.split()`. In the widest 8-nibble by 4-nibble product,
   each column has at most four lo and four hi terms, hence <=116. With carry
   <=7, column total <=123; the next carry is again <=7. The narrower cases
   obey this bound too. All integers and threshold inputs fit BF16's exact
   small-integer region. Normalize into digits before combining lanes.
4. Negate the lane bit vector when signs differ, modulo 2^32 for A or 2^64 for
   attention. Signed overflow semantics are explicit bit arithmetic. Reduce
   pairs of lane results with a balanced tree of radix-16 adders. Two digits
   plus incoming carry are <=31; carry is 0 or 1. Use the existing two-stage
   carry-lookahead style, with Boolean output bits. Every tree level therefore
   has the same bound, independent of N. For A, include old rd as one more
   tree leaf, or add it afterward with the existing 32-bit adder. Modulo
   reduction commutes with all sums.
5. Select output bits and serialize them with the ordinary value protocol.
   Never materialize the assembled 32- or 64-bit value as one residual scalar.

Nibble convolution avoids a tempting but invalid design that directly sums N
lane columns. Such columns can exceed 256 and lose BF16 units before a later
split repairs anything. A limb accumulator with large signed low/high scalars
has the same problem unless each materialized limb is independently bounded.
Int8-by-int8 would reduce range, but converting current activations to int8
would change the guest's numerical results and quality evidence.

The existing MUL has 64 nibble products <=225, bounded column splits <=255,
signed high-word corrections and a carry-lookahead add. Its five FFN stages
span original L5–L9 and total 9,212 named units in the expanded graph. Reuse
that pattern, not a full 32x32 multiplier per lane. Dot inputs arrive after
RAM and the last of 32 DROM sublayers. Place sign preparation, lane products,
normalization, reduction and serialization **after the last possible memory
read**. Some work on earlier-read operands can overlap later DROM layers, but
the last bank is the depth constraint. IR `_place()` must account for the
9,216-unit limit and real data dependencies. This has not been scheduled yet.

| Maximum width | A added layers | A added FFN units | A added residual dims | Dual-RAM attention build added layers, total |
|---:|---:|---:|---:|---:|
| 4 | 8–14 | 5–12k | 0.8–1.8k | 16–24 |
| 8 | 10–16 | 10–24k | 1.5–3k | 18–28 |
| 16 | 12–20 | 20–48k | 3–6k | 22–36 |

The attention budget includes four extra memory layers and larger 32x16 lane
products/64-bit reduction. Its additional FFN/dimension budget is not yet
compiled; allow roughly another 1–3 times A's arithmetic units and dimensions
as a resource sensitivity, not an allocation promise. The maximum-width
network runs on every token, even scalar opcodes. Opcode predicates do not
justify skipping its FFNs or memory kernels in a stock model.

Hidden size must be selected again with `pick_layout`, not assumed to remain
2,816. Without slot reuse, A width eight alone can push it toward 4.5–6k.
The 44-layer dense stock estimate is already 6.21B parameters / 12.42 GB BF16
weights before runtime. Wider projections and more layers can multiply that
size. The sparse executor's memory footprint does not prove that a full-map
stock model fits a particular GPU or this Mac. Export actual graph statistics,
slot allocation, parameter bytes and peak RAM before a full-map stock build.

## Return path and hidden accumulation

`value_record()` emits F0 for signed [-2^14,2^14-1], V,F1 for
[-2^29,2^29-1] otherwise, and V,V,F2 for other 32-bit results. Thus A costs
**one to three generated tokens per dot**, usually two to three when accumulating
matrix rows. rd=x0 uses NT after checks. PC refresh and faults retain existing
extra records. Repeated computation while emitting V does not commit rd early;
test that only the final token updates the architectural register.

Two attention half instructions cost two to six result tokens, plus normal
guest carry/add/control instructions. Each half is interpreted as a 32-bit
bit pattern, so a small high half may use F0. No new 64-bit token vocabulary
is necessary. It is two instructions, not one magical 64-bit register write.

A direct late accumulator head can receive current products, but a standard beta=1
DeltaNet update overwrites its selected row. It does not add v to the row.
The update is `S += beta*k*(v-k^T*S)` with keep decay. The previous accumulator
read emerges after the head projections have already been evaluated; a later
FFN sum cannot feed that same head's v on the next step through a residual
connection to an earlier layer. Cascading two heads only moves the feedback
problem. Storing a late partial sum in a later head does not make it available
to the preceding MAC stage on the next token.

This does not prove that all possible stock DeltaNet recurrence encodings are
impossible. It does rule out claiming that the current overwrite gadget plus
a convolution tap supplies an accumulator. A different analog/rotating-state
construction would need bounded exactness, arbitrary-length carry, reset,
fault priority and final token selection proofs. An accumulator encoded in
ordinary generated value tokens works, but saves no return tokens. Begin/step/end
controls can advance early pointers locally, yet do not transport late sum bits.

There is a more concrete **bounded partial-dot log** worth testing. It avoids
the feedback cycle by postponing addition instead of holding a running sum.
For a fixed power-of-two block size K<=128, a late head after the MAC stores
each 32-bit partial dot in a different row. One key / one logical value head
has enough columns and rows for this log; the configured stock pairing still
reserves two value slots. Early control maintains a slot counter from generated
control tokens, so it needs no late arithmetic bits. Each append can emit NT
and commit only after normal validation. A final instruction queries the K rows,
obtains per-bit population counts, carries them into a 32-bit sum and emits an
ordinary value record. It never sends a late accumulator to an earlier layer.

All K rows must retain ballast, and every selected slot must be freshly written
in the block. Requiring exactly K appends, including explicit zero partials for
padding, is the simplest initial contract; stale rows cannot be included.
With K a power of two, the fixed multi-hot read's normalization produces a
scaled average, so an output projection scaled by K can in principle recover
bit counts in [0,K]. This is a new numerical mapping, not a proved reuse of the
one-hot RAM primitive. Sum those bit counts with bitwise carry. At each bit,
`count + carry <= 2*K-1 <=255` and the next carry <=K-1, so bounded exact
digit logic is possible. Thirty-two serial bit stages are a depth-heavy fallback;
an efficient carry-lookahead construction and the stock multi-hot normalization
must be proved before budgeting it. Signed partials sum correctly modulo 2^32.
Attention can log 64-bit partials with the analogous, larger final reduction.

This would change hidden architectural state, reset/checkpoint/interrupt scope,
append/end control and fault semantics. Commit the previous append using lagged
MAC inputs at the late table, not a speculative write before its emitted control
is accepted. Faulted appends must not advance the early slot counter. A final
result or status must not control an earlier layer except through its token.
An unbounded stream still needs block reductions and token returns; a finite
log is not an arbitrary-length hidden running accumulator. Per append it might
save one or two result tokens, while begin/end, padding, slot controls, storage
and final reduction consume some of that saving. No such saving appears in the
performance tables. This is a legitimate stock-only candidate for a later PoC,
not a reason to bypass M3 or claim that hidden accumulation is impossible.

## Cost and speed estimates

The imported baseline is **228.155 microseconds per old token**, measured in a
reference-seeded rendering window of the fused sparse executor, not stock HF
and not this workload. See [feedback fusion](../dmc-feedback-fusion/README.md).
Stories260K scalar counts are 3,286,637 / 4,341,698 tokens at positions 4 / 63,
or 12.50 / 16.51 minutes at that cost. The existing README's 12.51 / 16.52 uses
the rounded rate 4,380 tokens/s. Both are projections, not measured LLM latency.

The stacked profiler spans 239.208 us/token. FFNs cost 55.779, DROM heads
53.190, DROM apply 20.464, RAM 41.333, other heads 15.201, code ROM 10.013,
other residual/history apply 23.750 and feedback 5.782 us in the instrumented
captures. Do not add these to 228.155. Normalize changed-category work by
239.208, then scale the wall baseline. For example, adding 50% FFN work alone
suggests factor `1+0.5*55.779/239.208 = 1.117`; a whole RAM replica adds about
`41.333/239.208 = 0.173` before wider routing and new arithmetic. Dense stock
cost may scale quite differently with width and depth.

Planning cost factors below include added arithmetic, wider output routing,
layers and launch overhead. They deliberately exceed simple FFN-unit scaling.
They are sensitivity inputs, not measured uncertainty intervals or proven bounds.
If actual cost exceeds them, recompute all estimates rather than retaining the
claimed speedup.

| Width | A transition factor | A assumed us/token | Matrix + QK factor | Matrix + QK assumed us/token |
|---:|---:|---:|---:|---:|
| 4 | 1.15–1.45 | 262–331 | 1.45–1.90 | 331–434 |
| 8 | 1.25–1.70 | 285–388 | 1.60–2.30 | 365–525 |
| 16 | 1.50–2.20 | 342–502 | 1.90–3.10 | 434–707 |

[estimates.py](estimates.py) regenerates [estimates.json](estimates.json) and
[the comparison chart](estimates.html). Its equations are as follows.

* For stories A, replace only measured matrix dot-loop tokens with 7–8 tokens
  per group. Those are 2–3 result tokens, two two-token pointer increments and
  one branch. Use actual padded group counts 64,832 / 32,576 / 16,288 for
  N=4/8/16. Keep measured nondot work unchanged. Actual aligned packing,
  refresh differences, cleanup, setup and register allocation need new traces.
* For attention sensitivity, assume **16 scalar trace tokens per QK MAC and
  16 per weighted-V MAC**. This is not separately profiled evidence. It fits
  inside measured nonmatvec tokens at both positions. Replace QK only by
  17–21 tokens/group for two half-result dots, 64-bit guest accumulation,
  pointer updates and branch. Keep every weighted-V MAC and other attention
  operation scalar. Wider W=16 uses N=8 for the baseline's eight-wide heads.
* Custom matrix MACs are 2,445,312 versus 259,328, ratio 9.429. Matrix setup
  is scaled by output rows, 10,816/3,512 = 3.080. After subtracting the assumed
  attention MAC cost from baseline nonmatvec work, scale its remaining cost
  by 1.8–3.6, spanning layer-head and layer-dimension ratios. Custom attention
  MACs scale by 3.6. This is a geometry sensitivity, not a custom integer trace
  or quality proof. Row quantization/scaling and actual division paths can
  change the numbers substantially.

Minutes per predicted LLM token, **matrix-only A**, both attention reductions scalar.

| Width | stories p4 | stories p63 | custom p4 | custom p63 |
|---:|---:|---:|---:|---:|
| 4 | 4.30–5.78 | 8.95–11.65 | 24.26–36.99 | 37.40–58.10 |
| 8 | 3.60–5.11 | 8.66–11.99 | 16.20–27.56 | 30.48–52.31 |
| 16 | 3.68–5.53 | 9.74–14.42 | 13.34–25.44 | 30.47–57.46 |

Minutes with the **dual-RAM matrix + QK build**, weighted V still scalar.

| Width | stories p4 | stories p63 | custom p4 | custom p63 |
|---:|---:|---:|---:|---:|
| 4 | 5.32–7.45 | 9.96–13.67 | 30.22–48.03 | 42.38–70.40 |
| 8 | 4.48–6.73 | 9.36–13.82 | 20.26–36.62 | 32.79–62.14 |
| 16 | 4.49–7.54 | 10.29–17.09 | 16.28–34.85 | 30.64–68.21 |

Width-eight A projects about 2.45–3.48x versus scalar at p4 and 1.38–1.91x
at p63. At width 16 the larger graph can lose to width eight despite fewer
guest instructions. The attention build can lose overall because all ordinary
tokens traverse its extra circuit. The larger custom matrix workload can
justify width 16 better; it does not establish a universal optimum.

Weighted-V coverage is deliberately explicit. It is an unaccelerated long-context
floor of 20,480*16 = 327,680 assumed tokens at baseline p63 and 73,728*16 =
1,179,648 at custom p63, before other attention work. Even an ideal zero-cost
replacement saves at most `those_tokens * new_us / 60e6` minutes in this model.
A stride accelerator must beat that saving after paying its extra graph cost
on every remaining token. At N=8, worst-case port replication alone suggests
8*0.173 = 1.38 additional transition factor before arithmetic. It is not
recommended without a substantially different, proved memory organization.
Serial stride rounds or guest transposition require new counts; no numerical
speedup is asserted for them. Varying the assumed 16 tokens/MAC to 12 or 20
changes QK savings by `4*L*D*(p+1)` tokens in either direction and correspondingly
changes the custom extrapolation. Profile both loops before selecting hardware.

For the custom model, training used context 256, but the current integer engine
would store int16 KV. At context 64 that is 24,576 bytes, with roughly 13,312
bytes of scratch plus 8,448 bytes reserved/stack, about 46,336 bytes total
before globals/alignment. It plausibly fits current RAM, subject to the linker.
At context 128 the same estimate is 71,168 bytes and does not fit 64 KiB.
The training report's int8 KV budget cannot be reused as an int16 engine budget.
The requested positions 4 and 63 can be explored without a memory-map expansion.

**There is no defensible stock-HF minutes/token estimate yet.** The compliant
network must first run under stock greedy generation. These tables answer the
conditional sparse-executor throughput question using the supplied measurement.
They are neither a host-computed accelerator proposal nor a claim that sparse
execution alone proves stock mapping. Full-map stock memory and latency are
separate acceptance gates.

## Compiler and reference-machine interface

Use custom-0 opcode 0x0b, R-type, leaving existing digit-file `funct7=0`
encodings unchanged. Proposed allocation, subject to ISA approval, is
`funct7=0x20` for `.s8s16.acc`, `0x21/0x22` for `.s32s16.lo/.hi`, and optional
`0x23/0x24` and `0x25/0x26` for `.s16s16.lo/.hi` and `.u16s16.lo/.hi`.
`funct3=0,1,2` selects N=4,8,16. All other combinations are illegal. These
are project-private instructions, not ratified RISC-V extensions.

Guest usage sketch, width eight.

```c
static inline uint32_t ndot8(uint32_t acc, const int8_t *w,
                              const int16_t *a) {
    __asm__ volatile (".insn r 0x0b, 1, 0x20, %0, %1, %2"
                      : "+r"(acc) : "r"(w), "r"(a) : "memory");
    return acc;
}
/* Guest loop calls ndot8, advances w by 8 and a by 8 halfwords,
   branches, then performs the original integer row rescale. */
```

Use unsigned accumulation at the C boundary for defined wrap semantics; the
current packer keeps actual full-row sums inside int32, so the output is
unchanged. Keep source pointers live and account for register aliases in inline
assembly. Disassemble a minimal compiled test and decode its exact words before
running workloads. No assembler modifications or system installs are needed.

An isolated experiment copy of `rv/ref.py` must decode these raw encodings,
extend `CUSTOM/UOPS`, emit their generic predecoded ROM fields, implement
span/alignment/fault checks and generate ordinary records. Extend the isolated
`rv/core.py` construction with the same generic operation and snapshot handling.
Do not edit pinned upstream. Update map preparation's source-hash manifests
explicitly rather than weakening its clean-pin checks. Host ROM predecode keeps
the existing platform contract; it cannot precompute dot results from data.

Keep a separate specification oracle with raw byte arrays and Python arbitrary
integers. Its proposed interface is `execute(word, old_regs, ram, drom) ->
(new_regs, fault, pc_delta)` plus an independently written record serializer.
It must not call `Machine` arithmetic, `rom_word`, `value_record` or circuit
helpers. The software Machine is a trace/reference implementation, not the
independent correctness oracle. The PoC in this directory is only arithmetic
evidence, not this complete ISA oracle.

## Verification milestones and acceptance

Stop after a failed gate and revise the design. No throughput milestone can
substitute for stock compatibility or bit-exact arithmetic.

| Stage | Work | Acceptance criterion |
|---|---|---|
| M0. Bounded arithmetic | Nibble multiplication, digit carry and sign/limb reference checks | Reproduce this directory's CPU PoC; scope remains primitive only. Completed as described below. |
| M1. Independent ISA contract | Oracle, custom-0 decode, guest intrinsic and isolated software reference | Every encoding/width/alias/fault test agrees with the independent oracle; original reference suites unchanged. No neural speed claim. |
| M2. Memory and full-dot IR | Add aligned wider routing without new map; W=4 then W=8; export actual stats | All DROM/RAM bytes and arithmetic bits agree per instruction; `evaluate(check=True,kernel=True)` passes, including last DROM bank; no BF16-inexact intermediate. One-bit corruption controls fail at the expected earliest boundary. |
| M3. Autonomous stock-HF | Tiny generic map, complete chip vocabulary/output head, cache snapshot, then unmodified `Qwen3_5ForCausalLM.generate(do_sample=False)` | Two different programs and changed DROM contents with identical network weights, >=1,024 dependent instructions, generated tokens only after GO; exact PC/register/RAM at record commits. Include dot followed by dependent address/branch/store and immediate scalar loads after stores. No teacher-forced answers, replaced forward, custom generate loop or intermediate cache injection. |
| M4. Full-map equivalence and resources | Same generic constructor with 64 KiB/4 MiB map; stock resource accounting; compact and stock triangulation | Actual head/layer/width/parameter counts, peak memory and numerical checks published. Small-map proof and full-map proof labeled separately. If full-map stock cannot run, claim only the bounded map and retain the blocker. |
| M5. Integer LLM regression | Rebuild aligned stories guest and compare with original integer engine | All 64 predicted token IDs, every-position logit hash, decoded bytes, KV clips and final architectural state bit-exact; no float-agreement shortcut. Compare at each dot in a bounded prefix, then full run. |
| M6. Custom engine | Convert the delivered dim192 model into the same per-row integer format, link context64, run native baseline first | RAM/ROM pass actual packer/linker budgets; integer range checks and native quality report; accelerator agrees bit-for-bit with that integer baseline, not with the unrelated float/Q8 runner. |
| M7. Cost checkpoint | Compare W=4/8/16 using new geometry and same output workload | Report transitions/prediction and total us/transition, layer/category work, wall latency, reset correctness, setup and memory separately. CPU stock timing first; future GPU work only when authorized/available. Keep or drop attention build based on end-to-end latency. |

Instruction tests include exhaustive nibble truth tables and digit adders,
all 256 s8 values against every s16 value in the independent arithmetic oracle,
signed extreme cross-products, zero, cancellation, alternating signs, old rd
near wrap boundaries and arbitrary old rd bit patterns. For s16*s16 and
s32*s16 use exhaustive subgadgets plus edge/random vectors, not an infeasible
claim of exhausting the full Cartesian product. Include N=4/8/16, positive
2^24, 2^30, 2^31 and 2^50 boundaries, carry chains across every nibble and
15/30-bit token chunk, rd=x0, source/destination aliases, and chained dots.

Memory tests enumerate every legal low alignment residue, invalid byte/halfword
offsets, first/last legal spans, last+one lane, each DROM column/half/value slot,
RAM row/slot/bank boundaries, wrong regions, wrapped addresses, old-map aliases,
read-only faults and prior SB/SH/SW at each lane. Replicated RAM must agree after
every store, checkpoint/resume and reset. Test intervening stores between
attention low/high instructions to confirm their specified independent reads.
Verify the exact fault PC and absence of partial architectural writes.

Stock checks must compare generated records and logical cache state, not only
terminal predictions. Also compare cached decode against chunked execution on
the same generated sequence to expose recurrent/conv differences. Preserve all
history on resume and clear it on a fresh reset. Use corruption controls on a
memory bit, accumulator bit, PC, write gate and token-family bit. No tolerance
may excuse a wrong architectural bit. Primitive supplied-input checks do not
satisfy M3. No independent review has been claimed; work is inline per AGENTS.md.

## PoC evidence and reproduction

Run from the repository root on the Mac with existing packages.

```sh
.venv/bin/python experiments/neural-accelerator-design/arithmetic_poc.py
python3 experiments/neural-accelerator-design/estimates.py
```

[arithmetic_poc.py](arithmetic_poc.py) uses the clean pinned IR and emitter.
[arithmetic-result.json](arithmetic-result.json) records exhaustive 256 nibble
products and 512 two-digit-plus-carry cases passing ideal/BF16 IR checks.
The product uses 15 FFN units in one layer. The small digit/carry construction
uses three units across two layers. Small CPU stock-HF checks match all residual
dimensions at 11 product inputs and nine adder inputs, both decode and prefill.
They use BF16 weights/residuals and standard CPU reference kernels under
torch 2.15.0 / transformers 5.21.0. No package installation was needed.

Separately, Python magnitude/limb reconstruction matches arbitrary-integer
multiplication for 10,025 pairs in each of s8*s16, s16*s16 and s32*s16,
including extrema, and 3,000 random dot vectors per format. These full-dot
checks are mathematical/reference checks, **not** an emitted full-dot circuit.
The IR evaluator uses its own ideal/hardware numeric checks; no new benchmark,
full-dot graph, port routing circuit, guest instruction, stock generation loop
or full LLM neural run was built. Nothing ran on a GPU.

The Model the Domain principle shaped the instruction/memory contracts and
separate evidence categories. Prove It Works led to executable primitive checks
and reproducible estimate arithmetic. Architect alternatives were compared in
one sequential pass under the explicit no-delegation rule.

## Decisions requiring the user

1. **ISA and claim.** Approve project-private custom-0 dot opcodes and a CPU
   claim of the existing ISA plus named generic extensions. Preserve existing
   digit opcodes and predecoded-code-state scope. Do not call this standard P,
   full RVV, a pretrained Qwen weight set or one instruction per transition.
2. **Stock mapping and resource envelope.** Approve increased chip depth/hidden
   width and wider output projections inside stock layers, retaining token-only
   feedback. Decide whether bounded-map stock proofs followed by full-map compact
   checks are an acceptable interim result, or full-map stock execution is required
   before claiming success. No tensor-feedback exception is requested or assumed.
3. **Memory representation.** Approve wider DROM column exposure and striped
   RAM reads under unchanged logical addresses/capacities, with a new stock
   exactness proof. A second RAM copy for attention is a separate optional
   representation decision. No new byte packing, cache transpose or port-copy
   layout is silently accepted.
4. **Memory map.** No change is proposed to RAM `0x01000000`/64 KiB, DROM
   `0x02000000`/4 MiB or code `0x00040000`/256 KiB. Any larger custom context,
   capacity change or region relocation needs explicit approval after real
   integer-engine budgets. Context128 int16 custom KV does not fit the projected
   current budget.
5. **Attention scope.** Choose after M7 whether the QK-only acceleration justifies
   its whole-network cost, and whether a separately proved generic stride engine
   is worth investigating. Changing Q precision to int16, KV to int8 or model
   quantization is a numerical/quality decision, not a bit-exact accelerator fix.
6. **Hidden state or multiwrite.** Any implicit accumulator, pointer state,
   auto-increment multiwrite or two-register result changes the commit/checkpoint
   contract and needs its own approval and stock construction. None is a premise
   of the recommended first build.

The top risks are BF16/normalization exactness of wider reads, arithmetic and
residual-width growth paid on every token, full-map dense stock memory, and
attention's incompatible operand types/strides. The first decisive experiment
is M2 followed by M3, not an optimistic sparse-executor benchmark.
