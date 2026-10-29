# Standalone compiled CPU with 64 KiB RAM

This throwaway prototype proves that the compiled RISC-V CPU can read and write beyond its original 32 KiB RAM limit. It does not run Doom.

The recorded 64 KiB run used stock `transformers.Qwen3_5ForCausalLM` in bf16 on CPU. Its free-running greedy generation matched all 27 reference tokens. It wrote `A` at `0x8000` and `B` at `0xfffc`, loaded them, and then loaded unchanged alias sentinels `X` at `0x0000` and `Y` at `0x7ffc`. The output was `ABXY`.

This is standalone CPU execution only. The upstream text bridge uses a text buffer at `0x8000`. The widened RAM overlaps that buffer, so this prototype must not be used with the text bridge enabled.

From the repository root, after its normal setup, run:

```sh
.venv/bin/python experiments/ram64/smoke.py --ram-kib 64
```

The script copies the clean pinned upstream package to `.cache/ram64`, changes the RAM word count and the compiled range predicate, emits the model weights in memory, and generates the trace. It writes a report, source hashes, patch manifest, and instruction words under `artifacts/ram64`. No package copy or model weights are part of this experiment's source. Choose a fresh `--scratch` directory on a rerun. The script refuses an existing scratch package rather than overwriting it.

Run the unchanged upstream 32 KiB baseline with the same load/store and alias pattern inside its original address range:

```sh
.venv/bin/python experiments/ram64/smoke.py --ram-kib 32
```

Its first upper-half word is `0x4000`, its last word is `0x7ffc`, and its low-bank alias sentinels are at `0x0000` and `0x3ffc`. Both upstream source hashes must remain unchanged in that run's manifest.

The 64 KiB report contains 64 RAM key heads and 13 chip layers. The original geometry has 32 RAM key heads and 11 chip layers. Both use a hidden size of 2,816. The model parameter counts are 2,361,445,056 and 2,113,019,200 respectively. Logical fp32 recurrent state grows from 12,517,376 bytes to 16,711,680 bytes. The dense stock model allocates recurrent state for inactive heads too, giving 23,068,672 and 27,262,976 bytes respectively. These are capacity counts, not speed claims.

## Next Doom memory map

Use an aligned 8 MiB RAM window at `0x01000000` and an aligned 8 MiB DROM window at `0x02000000`. Keep the existing code ROM at `0x00040000` with its 256 KiB capacity and 16-bit jump anchors. This avoids changing PC, JALR, AUIPC, or jump token geometry in the first memory implementation. The final linked Doom image must still demonstrate that its live code fits that ROM.

Place the original shareware IWAD in DROM after the engine's read-only data. The supplied local file is 4,196,020 bytes, so a 4 MiB DROM window is already too small before engine read-only data. Keep the IWAD local and excluded from the public repository. A memory-backed WAD reader must copy requested bytes using compiled RV instructions. The host supplies the boot snapshot, not individual load results.

The smallest next production function is `region_literals(address_bits, base, size)`. Require a power-of-two size and a base aligned to that size. Return the fixed upper address-bit literals. Use it for `in_ram` and `in_drom`. Derive RAM bank-select width from `RAM_WORDS` and DROM bank-select width from `DROM_WORDS`. In particular, the existing DROM query slice `EA[13:17]` is fixed at four bank bits and must widen for the selected geometry. Update `DATA_BASE`, `SP_INIT`, and linker segments together with these constants. The reference machine, ELF loader, and snapshot writer already read the region constants, but need boundary checks against the new map.

First validate this high-address map with a small number of banks. Cover word, byte, signed and unsigned halfword loads, byte/halfword merged stores, old-address aliases, both region boundaries, out-of-range faults, and the unchanged ROM jump path. Then build the larger graph. The 64 KiB proof only covers SW/LW and two alias pairs.

## Capacity gap at Doom scale

`geometry.py` checks that these windows do not overlap and reproduces `proposed-geometry.json`. Its larger graph counts are analytical extrapolations, not an emitted or executed Doom model.

The existing bit-bank representation stores 1 KiB of actual RAM per key head with two value heads. Eight MiB requires 8,192 RAM key heads. The byte-packed read-only representation stores 8 KiB per key head, so eight MiB requires another 1,024 DROM key heads. At the existing target of 16 key heads per layer, the resulting graph is estimated at 584 chip layers, 18,527 logical value heads, and 1,214,185,472 bytes of logical fp32 recurrent state. Dense stock recurrence allocates approximately 1.22 GB.

Holding the current dense layout and layer dimensions constant gives 73,287,026,944 parameters, or 146,574,053,888 bf16 parameter bytes. This is not a claim that Doom will fit a 16 GB device. The next scale implementation must emit and run the compact circuit representation, or prove a denser writable-memory representation, before trying to allocate a stock dense model of that geometry. A byte-packed writable alternative might reduce bank count, but its update arithmetic and bf16 exactness have not been implemented or tested here.

The provisional 8 MiB RAM budget has a 6 MiB zone, 256,000-byte 320-by-200 RGBA output buffer, 297,309 bytes of unpruned writable object sections, and a reserved 64 KiB stack. It leaves 1,478,307 bytes before libc and additional dynamic allocations. The zone contains the 64,000-byte indexed framebuffer, so that framebuffer is not added twice. These are planning reservations. Only a final link and an actual engine initialization can establish its live requirement.

Once memory is implemented, link a bare-metal RV32IM+Zba Doomgeneric port against a minimal libc and a memory-backed WAD file. Implement the DG callbacks with deterministic ticks and no audio. Let `DG_DrawFrame` emit a completion marker after the real renderer completes its first frame. Stop the generated trace at that marker and decode the framebuffer from the recurrent state, rather than serializing 64,000 individual byte ecalls. That decoder still needs an independent check against emitted bytes on a small test image before it can count as evidence. No host engine rendering may replace those pixels.
