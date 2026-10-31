# Qwenputer games

An original CHIP-8 paddle game and a verified Doom E1M1 frame running as RISC-V firmware through [Qwenputer](https://github.com/divagr18/qwenputer)'s compiled CPU circuit. This repository contains source and execution evidence from an Apple M4 Pro with 24 GB unified memory and an RTX 5070 Ti with 16 GB VRAM. Doom game data and asset-bearing execution files remain local.

![Verified CHIP-8 output, replayed at four frames per second](../artifacts/compact-game-cuda/replay.gif)

**This GIF is accelerated, prerecorded output.** The GPU circuit generated the 12 frames in 8.48 seconds after setup, excluding reference generation, comparison, and rendering. Playback is four frames per second. It is not live gameplay. The earlier Mac compact run produced identical frames in 116.20 seconds, including token comparison.

[Open the shareable replay](https://ad1tyanarayana.github.io/qwenputer-games/). Pause or scrub through the verified frames, then inspect the linked execution evidence.

## What ran

| Experiment | Execution path | Verified result |
| --- | --- | --- |
| Weight Paddle | CHIP-8 ROM → C interpreter compiled to RV32IM → upstream compact tensor circuit executor | 54,570 self-generated tokens, zero reference mismatches, 12 distinct frames |
| Complete Weight Paddle on RTX 5070 Ti | Same firmware → compact Triton circuit and CUDA graphs | All 54,570 tokens and 12 frames match the Mac run; one 8.48-second generation run |
| Doom E1M1 frame on RTX 5070 Ti | DMC strip-rendering port → expanded compact Triton circuit | 6,342,933 generated tokens, zero mismatches, 25 strips covering 320×240 pixels; frame and final RAM match the independent reference |
| RISC-V hello | Stock Hugging Face `Qwen3_5ForCausalLM` with locally compiled CPU weights | Five generated tokens, output `H`, successful halt |
| Game boot prefix | Stock Hugging Face model, same game firmware | First 16 generated tokens match exactly; this prefix does not contain a frame |
| First neural game frame on RTX 5070 Ti | Stock Hugging Face model with bf16 compiled CPU weights on CUDA | 5,415 generated tokens, zero mismatches, one complete frame matching the Mac run |
| First frame with GPU circuit acceleration | Compact fp32 circuit executed with Triton and CUDA graphs | Same 5,415 tokens; five exact runs, median 0.841 seconds after setup |
| Input behavior | Reference execution of two key schedules | Hit schedule gives one hit; miss schedule gives one miss |

The complete game run uses upstream's optimized circuit executor. It uses compiled tensor operations and specialized Gated DeltaNet memory operations. It is distinct from executing stock Hugging Face layers. The separate stock-model checks establish only the bounded results shown above.

The pretrained Qwen3.5-4B language model was not loaded. The stock Hugging Face model generated one complete frame on the RTX 5070 Ti in a single 76.37-second correctness run, with 4.36 GB of peak PyTorch CUDA allocation. The complete 12-frame replays use compact circuit executors on CPU or GPU. The Python reference emulator checks generated tokens and does not supply the executor's next token or game frames. Python renders only bytes emitted by the RISC-V firmware.

## Run the experiment

Verified on macOS arm64 with Python 3.13.4. Setup installs pinned dependencies into `.venv` and pins Qwenputer under `.upstream`. The stock-model smoke test constructs approximately 4.23 GB of bf16 weights and needs additional temporary memory. No pretrained model download is needed.

```sh
python3 tools/setup.py
.venv/bin/python tools/build_game.py
.venv/bin/python tools/run_rv.py
.venv/bin/python tools/render_replay.py
.venv/bin/python tools/check_game.py
```

Open `artifacts/compact-game/index.html` for the replay. Its controls pause playback and select recorded frames. To serve it locally:

```sh
.venv/bin/python -m http.server 8767 --bind 127.0.0.1
```

Then open [the local replay](http://127.0.0.1:8767/artifacts/compact-game/).

Run the stock neural model checks separately:

```sh
.venv/bin/python tools/hf_smoke.py
.venv/bin/python tools/hf_smoke.py --firmware artifacts/compact-game/weight-paddle.elf --max-tokens 16
```

To generate a miss instead of a hit, build with `--controls miss`. The guest interpreter supports the instructions used by this original ROM. It is not a general CHIP-8 compatibility implementation.

For the tested native Windows CUDA path, use [the GPU reproduction instructions](../experiments/cuda/README.md). Its tested PyTorch version is 2.14.1+cu130, distinct from the Mac lockfile. It does not require Ollama or WSL.

## Inspect the evidence

- [Complete compact execution report](../artifacts/compact-game/report.json), [frames](../artifacts/compact-game/frames.json), and [input schedule with ROM bytes](../artifacts/compact-game/game.json).
- [Complete GPU circuit report](../artifacts/compact-game-cuda/report.json), [replay](../artifacts/compact-game-cuda/index.html), and [exact full trace](../artifacts/compact-game-cuda/trace.u32).
- [Stock-model hello result](../artifacts/hf-smoke/report.json) and [game-prefix result](../artifacts/hf-game-prefix/report.json).
- [Stock-model CUDA first frame](../artifacts/hf-cuda-first-frame/frame.png), [annotated execution report](../artifacts/hf-cuda-first-frame/report.annotated.json), and [trace](../artifacts/hf-cuda-first-frame/trace.u32).
- [Saved RISC-V firmware](../artifacts/compact-game/weight-paddle.elf) and [generated token trace](../artifacts/compact-game/trace.u32).
- [Firmware source](../firmware/chip8.c), [original ROM generator](../tools/build_game.py), and [independent artifact checks](../tools/check_game.py).
- [Verification scope and review](../docs/verification.md).

The complete-game and stock-model timings are individual correctness runs, not a controlled hardware comparison. They exclude compilation and reference generation. The stock CUDA first-frame timing also includes online checking and checkpoint writes. Its dense execution path is different from the compact executors, so these runs do not establish a hardware speed ratio.

The subsequent [compact GPU comparison](../experiments/cuda-compact/README.md) alternated five runs of each executor on the same GPU. Triton took a median 0.841 seconds and sparse PyTorch CUDA graphs 3.660 seconds for the identical first-frame trace, excluding setup and warm-up. Those are compact circuit measurements, separate from the stock-model evidence.

## Next milestones

1. **CHIP-8 execution proof.** Complete for the compact circuit path, with an accelerated replay and one complete stock-model frame on the GPU.
2. **First Doom frame.** Complete through the compact Triton circuit. [Source, reproduction and evidence](../experiments/dmc-cuda/README.md) cover one verified E1M1 frame, not stock-model inference or interactive gameplay.
3. **Playable Doom.** Produce continuing frames with input, then measure and improve responsiveness. Not yet complete.

[The Doom feasibility audit](../docs/doom-feasibility.md) identifies the first blocker for stock Doomgeneric. The released CPU has 32,768 writable bytes; that port stores a 64,000-byte indexed framebuffer and a larger allocation zone. The user-supplied DMC100-Doom / Aspire PIXO port offers a different approach: a strip renderer and streamed map records designed for much less RAM. The integrated DMC port now runs on the expanded circuit memory map. No WAD is included.

[A standalone 64 KiB memory prototype](../experiments/ram64/README.md) has now passed actual stock-model generation, including upper-bank reads, writes, and alias sentinels. It is not compatible with the current text bridge. Scaling the existing dense memory representation to the provisional Doom map remains too large for these GPUs; the experiment records that calculation and the next memory design work.

[The expanded compact memory runtime](../experiments/memory-map/README.md) has passed 12 cases at 64 KiB RAM / 4 MiB DROM, including full recurrent-RAM extraction, and a shorter test at 8 MiB / 8 MiB. Its direct sparse constructor removes large temporary matrices. The smaller map is the current target for the strip-rendering Doom port. Neither memory proof is a Doom run.

[The same smaller map passed on CUDA](../experiments/memory-map-cuda/README.md): 142 generated tokens and all 65,536 extracted RAM bytes matched the CPU proof. This verifies GPU state extraction for future strip-frame capture.

Two real Doom ports render E1M1 in the software reference machine: [stock Doomgeneric](../experiments/doom-baremetal/README.md) needs 51,386,454 trace tokens; [the DMC strip-rendering port](../experiments/dmc-rv32/README.md) needs 6,342,933 tokens and only 64 KiB RAM. The latter uses the user-supplied Aspire PIXO source and captures actual rendered strips from guest memory.

The [DMC GPU circuit run](../experiments/dmc-cuda/README.md) subsequently generated that entire trace itself and extracted all 25 strips from actual recurrent GPU RAM. Every token, all frame bytes and all 65,536 final RAM bytes matched the independent Mac reference. Retained-path generation took 3,044.57 seconds (50.74 minutes), excluding setup, comparisons, IO and checkpoint handling. A Windows progress-file sharing conflict near completion required resuming from an actual GPU checkpoint; discarded work and the recovery pause are not included in that retained-path timing. This is one correctness run, not a performance benchmark. The public export contains source and textual evidence; game data, full Doom traces, executable and state remain local.

For the full pretrained model, the author reports an approximately 18.8 GB checkpoint and recommends a 24 GB GPU. The Mac's 24 GB is shared with the OS. A desktop RTX 5070 Ti has 16 GB VRAM and benefits from upstream CUDA support, but cannot hold that unchanged checkpoint entirely in VRAM. We have not validated quantization, offloading, or a Mac GPU port. See [the author's write-up](https://divagr.com/blog/qwenputer) and [NVIDIA's specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5070-family/).

## License and credit

The original Weight Paddle ROM and new code are MIT licensed except where a file carries another license. The DMC platform adaptation retains GPL-2.0-or-later. Qwenputer is by Divyansh Agrawal; DMC100-Doom is by Artur Bieniek, and the Aspire PIXO port is from atc1441's project. Notices are preserved in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) and each experiment's NOTICE. Doom game assets remain separate local inputs. GPT-6.1 Sol agents assisted implementation and investigation, with orchestration and independent verification recorded in this project's working trail.
