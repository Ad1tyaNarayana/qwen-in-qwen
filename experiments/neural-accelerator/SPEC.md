# Project-private neural dot extension

The ISA claim is RV32IM plus the named project-specific dot-product extension.
The inherited CPU also supports Zba. This extension is neither P nor RVV.

## Encoding and architectural effects

`ndotN.s8s16.acc rd, rs1, rs2` is R-type custom-0, opcode `0x0b`,
funct7 `0x20`. Funct3 zero selects N=4 and one selects N=8. A W=4
constructor rejects N=8. W is either four or eight. Funct3 2 through 7
and other unallocated funct7 encodings are illegal. The inherited funct7=0
digit-file instructions retain their meanings. Uops 70 and 71 encode NDOT4
and NDOT8 in predecoded immutable code state. Predecode reads no operand data.

All operands come from the old register file, including old rd. Let A=x[rs1],
B=x[rs2], C=x[rd]. Require A divisible by N and B divisible by 2N.
Require the complete half-open spans [A,A+N) in DROM and [B,B+2N) in RAM,
using widened address arithmetic. Misalignment has priority over range faults.
No operand bytes are read before validation. Missing initial image bytes are zero.

Read N signed bytes from A and N signed little-endian halfwords from B.
The result is `(C + sum(w[i]*a[i])) modulo 2^32`. Write only rd, unless rd=0.
Advance PC by four. x0 suppresses the write, never validation. Aliases use the
old state. There is no accumulator state, pointer update, memory write,
saturation, scale operation, or second destination.

An invalid encoding faults as illegal. Misaligned operands fault as misaligned
load. Otherwise invalid spans fault as load out of range. Fault PC is the dot
instruction. Faults suppress register, RAM, and PC writes. Inherited fault
serialization is FLT(cause), eight most-significant-first HEX digits, HALT.

## Map and record transport

RAM is 0x01000000 with 65,536 bytes. DROM is 0x02000000 with 4,194,304 bytes.
Code is immutable at 0x00040000 with 262,144 bytes. No aliases are added.
The ordinary refresh convention remains unchanged.

The result uses F0 for signed [-2^14,2^14), otherwise V,F1 for signed
[-2^29,2^29), otherwise V,V,F2. Chunks contain bits 0..14, 15..29,
and 30..31. Only the final token commits. rd=0 returns NT after validation.
No incomplete result token commits an architectural write.

## Module contract and workflow

`oracle.execute(word, old_regs, ram_bytes, drom_bytes, maximum)` returns an
immutable register tuple, optional cause, and PC delta. Its serializer is
independent of the compiler and software Machine.
`reference.Machine` extends the frozen expanded-map Machine and delegates
unextended instructions to it. `reference.decode` and `reference.rom_word`
extend the predecode without changing the pinned source.
`ndot.h` exposes unsigned accumulators and `.insn` intrinsics.

Two structures were considered inline under the no-delegation instruction.
A generated patched reference would integrate all paths in one copy, but make
unchanged scalar behavior harder to inspect. The selected subclass and separate
decode adapter preserve the scalar method verbatim. The eventual neural
constructor must consume the same predecode codes without using the oracle.

M1 must pass independent arithmetic, record, fault, encoding, intrinsic, and
legacy workload checks before M2. M2 must pass W=4 before W=8, then memory,
arithmetic, kernel exactness, and corruption checks before M3. M3 requires
ordinary stock greedy generate on two programs and changed data with identical
weights, at least 1,024 dependent instructions per run, and state at commits.
The first failed acceptance gate stops this run. Test implementation defects
are recorded separately and never count as circuit evidence.

The throughput checkpoint is after M3. No timing or speed claim is part of
M1 through M3. M4 must establish full-map stock behavior and actual resources;
M5 must establish integer-LLM equivalence.
