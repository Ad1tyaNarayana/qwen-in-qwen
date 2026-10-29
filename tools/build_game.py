"""Build an original CHIP-8 paddle game as a freestanding RV32IM ELF."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--controls", choices=["hit", "miss"], default="hit")
    parser.add_argument("--out", type=Path, default=ROOT / "build")
    args = parser.parse_args()
    if not 1 <= args.frames <= 255:
        parser.error("frames must be between 1 and 255")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    words = []
    labels = {}
    fixups = []

    def label(name):
        labels[name] = 0x200 + 2 * len(words)

    def emit(*ops):
        words.extend(ops)

    def jump(name, prefix=0x1000):
        fixups.append((len(words), name, prefix))
        words.append(0)

    emit(0x00E0, 0x6002, 0x610C, 0x620A, 0x6309, 0x64FF, 0x6501)
    jump("paddle", 0xA000)
    emit(0xD014)
    jump("ball", 0xA000)
    emit(0xD231)
    label("loop")
    jump("paddle", 0xA000)
    emit(0xD014)
    jump("ball", 0xA000)
    emit(0xD231, 0x6701, 0xE7A1, 0x71FF, 0x6704, 0xE7A1, 0x7101,
         0x8244, 0x8354, 0x323F)
    jump("left")
    emit(0x64FF)
    label("left")
    emit(0x3204)
    jump("top")
    emit(0x8830, 0x8815, 0x3F01)
    jump("miss")
    emit(0x6704, 0x8875, 0x3F00)
    jump("miss")
    emit(0x6401, 0x7901)
    jump("top")
    label("miss")
    emit(0x7A01, 0x620A, 0x6309, 0x64FF)
    label("top")
    emit(0x3300)
    jump("bottom")
    emit(0x6501)
    label("bottom")
    emit(0x331F)
    jump("draw")
    emit(0x65FF)
    label("draw")
    jump("paddle", 0xA000)
    emit(0xD014)
    jump("ball", 0xA000)
    emit(0xD231)
    jump("loop")
    label("paddle")
    emit(0x8080, 0x8080)
    label("ball")
    emit(0xC000)
    for offset, name, prefix in fixups:
        words[offset] = prefix | labels[name]
    rom = b"".join(word.to_bytes(2, "big") for word in words)
    (out / "weight-paddle.ch8").write_bytes(rom)
    schedule = [(2 if args.controls == "miss" and 1 <= f <= 4 else
                 (16 if args.controls == "hit" and 8 <= f <= 10 else 0))
                for f in range(args.frames)]
    (out / "game.h").write_text(
        f"#define FRAME_COUNT {args.frames}\n#define GAME_LOOP {labels['loop']}\n"
        + "static const unsigned char game_rom[] = {" + ",".join(map(str, rom)) + "};\n"
        + "static const unsigned short game_keys[] = {" + ",".join(map(str, schedule)) + "};\n")
    layout = ROOT / ".upstream/qwenputer/experimental/riscv/chipformer/rv/env/rom.ld"
    cmd = [sys.executable, "-m", "ziglang", "cc", "-target", "riscv32-freestanding-none",
           "-mcpu=generic_rv32+m", "-mabi=ilp32", "-O2", "-ffreestanding", "-nostdlib",
           "-fno-stack-protector", "-fno-pic", "-mno-relax", "-I", str(out),
           "-Wl,-T," + str(layout),
           str(ROOT / "firmware/start.S"), str(ROOT / "firmware/chip8.c"),
           "-o", str(out / "weight-paddle.elf")]
    subprocess.run(cmd, check=True, cwd=ROOT)
    import json
    (out / "game.json").write_text(json.dumps({"name": "Weight Paddle", "frames": args.frames,
        "controls": args.controls, "key_masks": schedule, "rng": "none",
        "loop_address": labels["loop"], "rom_hex": rom.hex()}, indent=2))
    print(out / "weight-paddle.elf")


if __name__ == "__main__":
    main()
