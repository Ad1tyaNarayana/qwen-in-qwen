"""Run firmware using upstream's compact tensor circuit executor and verify every token."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".upstream/qwenputer/experimental/riscv"))

import numpy as np
import torch
from chipformer.rv import core, elf, ref, spec
from chipformer.rv.fastexec import Executor


def frames(data):
    result = []
    position = 0
    while position < len(data):
        if data[position:position + 2] != b"QF":
            raise ValueError(f"bad frame header at byte {position}")
        frame_id, count = data[position + 2:position + 4]
        hits, misses = data[position + 4:position + 6]
        position += 6
        image = [0] * 256
        for _ in range(count):
            index, value = data[position:position + 2]
            image[index] = value
            position += 2
        result.append({"id": frame_id, "bytes": image, "hits": hits, "misses": misses,
                       "sha256": hashlib.sha256(bytes(image)).hexdigest()})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("firmware", nargs="?", type=Path, default=ROOT / "build/weight-paddle.elf")
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/compact-game")
    args = parser.parse_args()
    source = args.firmware.resolve().parent
    bundle = {"weight-paddle.elf": args.firmware.read_bytes()}
    if not args.reference_only:
        for name in ["weight-paddle.ch8", "game.h", "game.json"]:
            bundle[name] = (source / name).read_bytes()
        schedule = json.loads(bundle["game.json"])
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    program = elf.program(args.firmware)
    machine = ref.Machine.from_program(program)
    want = machine.run(2_000_000)
    if not machine.halted or spec.family(want[-1])[0] != "HALT" or machine.x[10] != 0:
        raise RuntimeError(f"firmware did not exit successfully; a0={machine.x[10]}")
    print(f"reference halted after {len(want) - 1} generated tokens", flush=True)
    if args.reference_only:
        result = frames(bytes(machine.out))
        (args.out / "reference-frames.json").write_text(json.dumps(result))
        print(f"reference produced {len(result)} frames, {len(set(f['sha256'] for f in result))} distinct", flush=True)
        return
    circuit = core.build()
    core.snapshot(circuit, ref.Machine.from_program(program))
    runner = Executor(circuit, device="cpu", sparse=True)
    runner.load_init()
    current = spec.tok("GO")
    embeddings = {}
    generated = []
    started = time.perf_counter()
    with torch.no_grad():
        for offset in range(0, len(want) - 1, 256):
            amount = min(256, len(want) - 1 - offset)
            chunk = []
            for _ in range(amount):
                if current not in embeddings:
                    embeddings[current] = runner.embed([current])
                residual = runner.step(embeddings[current])
                current = core.predictions(circuit, residual.numpy())[0]
                chunk.append(current)
            expected = want[offset + 1:offset + 1 + amount]
            if chunk != expected:
                mismatch = next(i for i, (a, b) in enumerate(zip(chunk, expected)) if a != b)
                raise AssertionError(f"token {offset + mismatch}: got {chunk[mismatch]}, expected {expected[mismatch]}")
            generated.extend(chunk)
            if offset % 2048 == 0:
                print(f"verified {len(generated)}/{len(want)-1} tokens", flush=True)
    elapsed = time.perf_counter() - started
    output = bytes(spec.family(t)[1] for t in generated if spec.family(t)[0] == "BYTE")
    assert output == bytes(machine.out)
    result = frames(output)
    if len(result) < 2 or len({f['sha256'] for f in result}) < 2:
        raise AssertionError("game did not produce changing frames")
    np.asarray(generated, dtype="<u4").tofile(args.out / "trace.u32")
    (args.out / "frames.json").write_text(json.dumps(result))
    report = {"mode": "upstream compact RISC-V circuit tensor executor, CPU fp32; not full Qwen",
              "upstream_revision": "ca13310e2f9f788d1dd3feeea8213c1b0c23913c",
              "firmware_sha256": hashlib.sha256(bundle["weight-paddle.elf"]).hexdigest(),
              "generated_tokens": len(generated), "mismatches": 0,
              "frames": len(result), "distinct_frames": len({f['sha256'] for f in result}),
              "elapsed_seconds_one_run": elapsed, "timed_work": "free-running circuit generation and token comparison; excludes build and reference run",
              "seed": None, "inputs": "deterministic game_keys in saved game.h", "circuit": circuit.stats(),
              "python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__}
    report["upstream_revision"] = subprocess.check_output(["git", "-C", str(ROOT / ".upstream/qwenputer"), "rev-parse", "HEAD"], text=True).strip()
    if args.firmware.read_bytes() != bundle["weight-paddle.elf"]:
        raise RuntimeError("Firmware changed during execution; rerun from a stable build.")
    report["input_schedule"] = schedule
    for name, data in bundle.items():
        (args.out / name).write_bytes(data)
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
