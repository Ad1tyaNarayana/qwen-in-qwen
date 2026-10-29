# Qwenputer games

An original CHIP-8 paddle game running as RISC-V firmware through [Qwenputer](https://github.com/divagr18/qwenputer)'s compiled CPU circuit. This repository contains the source, execution trace, inputs, and frames from an Apple M4 Pro with 24 GB unified memory.

![Verified CHIP-8 output, replayed at four frames per second](artifacts/compact-game/replay.gif)

**This GIF is accelerated, prerecorded output.** The saved run produced 12 frames in 116.20 seconds, including token comparison. Playback is four frames per second. It is not live gameplay.

## What ran

| Experiment | Execution path | Verified result |
| --- | --- | --- |
| Weight Paddle | CHIP-8 ROM → C interpreter compiled to RV32IM → upstream compact tensor circuit executor | 54,570 self-generated tokens, zero reference mismatches, 12 distinct frames |
| RISC-V hello | Stock Hugging Face `Qwen3_5ForCausalLM` with locally compiled CPU weights | Five generated tokens, output `H`, successful halt |
| Game boot prefix | Stock Hugging Face model, same game firmware | First 16 generated tokens match exactly; this prefix does not contain a frame |
| Input behavior | Reference execution of two key schedules | Hit schedule gives one hit; miss schedule gives one miss |

The complete game run uses upstream's optimized circuit executor. It uses compiled tensor operations and specialized Gated DeltaNet memory operations. It is distinct from executing stock Hugging Face layers. The separate stock-model checks establish only the bounded results shown above.

The pretrained Qwen3.5-4B language model was not loaded. No full game frame has yet been generated through the stock Hugging Face model. The Python reference emulator checks generated tokens and does not supply the executor's next token or game frames. Python renders only bytes emitted by the RISC-V firmware.

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

## Inspect the evidence

- [Complete compact execution report](artifacts/compact-game/report.json), [frames](artifacts/compact-game/frames.json), and [input schedule with ROM bytes](artifacts/compact-game/game.json).
- [Stock-model hello result](artifacts/hf-smoke/report.json) and [game-prefix result](artifacts/hf-game-prefix/report.json).
- [Saved RISC-V firmware](artifacts/compact-game/weight-paddle.elf) and [generated token trace](artifacts/compact-game/trace.u32).
- [Firmware source](firmware/chip8.c), [original ROM generator](tools/build_game.py), and [independent artifact checks](tools/check_game.py).
- [Verification scope and review](docs/verification.md).

The timing is one observed run, not a controlled hardware benchmark. It excludes compilation and the reference run. No RTX 5070 Ti performance measurement has been made.

## Next milestones

1. **CHIP-8 execution proof.** Complete for the compact circuit path, with an accelerated replay and bounded stock-model checks.
2. **First Doom frame.** Run the actual Doom engine through the compiled RISC-V CPU and capture one frame, regardless of speed. Not yet complete.
3. **Playable Doom.** Produce continuing frames with input, then measure and improve responsiveness. Not yet complete.

[The Doom feasibility audit](docs/doom-feasibility.md) identifies the first blocker. The released CPU has 32,768 writable bytes. Doom's 320×200 indexed framebuffer alone takes 64,000 bytes. Expanding the compiled memory map and supplying Doom's platform services are required. This is a limitation of the current machine, not proof that a Doom port is impossible. No Doom executable has run here, and no WAD is included.

For the full pretrained model, the author reports an approximately 18.8 GB checkpoint and recommends a 24 GB GPU. The Mac's 24 GB is shared with the OS. A desktop RTX 5070 Ti has 16 GB VRAM and benefits from upstream CUDA support, but cannot hold that unchanged checkpoint entirely in VRAM. We have not validated quantization, offloading, or a Mac GPU port. See [the author's write-up](https://divagr.com/blog/qwenputer) and [NVIDIA's specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5070-family/).

## License and credit

New code and the original Weight Paddle ROM are MIT licensed. Qwenputer is by Divyansh Agrawal. Its notices are preserved in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Doom source and game assets have separate licenses; the audit does not redistribute either. GPT-6.1 Sol agents assisted implementation and investigation, with orchestration and independent verification recorded in this project's working trail.
