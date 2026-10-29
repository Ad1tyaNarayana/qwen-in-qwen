# Verify the CHIP-8 milestone

The complete recorded game uses the upstream compact circuit tensor executor on the Mac CPU. The separate stock Hugging Face checks use compiled RISC-V weights without pretrained language-model weights.

Run `.venv/bin/python tools/check_game.py` to check the saved evidence. It rebuilds two input schedules in temporary directories and verifies their distinct hit and miss outcomes with the reference machine. It then checks all 54,570 saved compact tokens against a fresh reference run of the saved firmware, reconstructs the emitted frames, checks frame hashes, and compares the stock-model prefix with the same trace. It also checks all 5,415 CUDA tokens, decodes their output bytes, and verifies that the resulting frame matches the first compact frame.

The check passed during implementation and again during orchestration and independent review. The original execution generated every next token from the circuit's previous output. The reference only supplied a check and stopping budget. The saved game has 12 distinct frames.

An independent GPT-6.1 Sol reviewer checked every GIF frame, the final PNG, and the HTML's embedded frame data against the saved pixels. Pause and frame selection were also exercised in the local browser. Review used the same model family requested for implementation.

The stock model's five-token hello program was rerun independently and printed `H` before halting. The 16-token game-prefix result refers to the same firmware hash as the complete compact run. It proves a boot prefix, not a rendered frame.

The subsequent Windows CUDA run generated one complete frame through stock Transformers layers using the same firmware. All 5,415 tokens matched; peak PyTorch allocation was 4,361,678,848 bytes. The single 76.37-second run includes prefill, state injection, generation, online checking, and checkpoint writes. It excludes compilation and reference generation. The executed harnesses and environment are preserved under `experiments/cuda`. This is a bounded correctness result, not a hardware benchmark or a full pretrained-model run.

The separate 64 KiB RAM prototype was independently rerun using its exported wrapper on the Mac. All 27 generated tokens matched and printed `ABXY`: upper-half writes at `0x8000` and `0xfffc` preserved the lower-half alias sentinels. This checks the expanded address decoding and bank allocation in actual stock-model execution. It does not establish Doom compatibility or exhaustive memory correctness. The prototype overlaps the existing text bridge and is limited to the standalone CPU.

Review found and corrected two reproducibility issues. The runner now saves the actual supplied firmware and its adjacent ROM and input files. A new two-frame run from a different source directory generated 10,484 matching tokens and saved the correct files. Setup now rejects modified upstream checkouts even when their HEAD is the pinned revision; this rejection was exercised with a temporary untracked file.

The preserved ELF includes compiler debug paths for this project's source files. It is retained byte for byte because the execution and stock-model prefix identify it by SHA-256. No model credentials or private SSH material are included.

The Doom audit is a source and object-compilation experiment. Its memory calculation was rerun independently, and its object section totals were recomputed during review. No Doom executable was linked or run.
