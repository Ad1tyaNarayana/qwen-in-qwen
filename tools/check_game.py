"""Verify input-dependent gameplay and independently replay the saved tensor-generated trace."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from run_rv import frames, ROOT
import numpy as np
from chipformer.rv import elf, ref, spec


def main():
    with tempfile.TemporaryDirectory(prefix="qwenputer-game-") as temporary:
        for control, expected in [("hit", (1, 0)), ("miss", (0, 1))]:
            output = Path(temporary) / control
            subprocess.run([sys.executable, str(ROOT / "tools/build_game.py"),
                            "--controls", control, "--out", str(output)], check=True)
            machine = ref.Machine.from_program(elf.program(output / "weight-paddle.elf"))
            machine.run(2_000_000)
            assert machine.halted and machine.x[10] == 0
            images = frames(bytes(machine.out))
            actual = images[-1]['hits'], images[-1]['misses']
            assert actual == expected, (control, actual, expected)
            print(f"reference gameplay {control}: hits={actual[0]}, misses={actual[1]}")
    directory = ROOT / "artifacts/compact-game"
    firmware = directory / "weight-paddle.elf"
    report = json.loads((directory / "report.json").read_text())
    digest = hashlib.sha256(firmware.read_bytes()).hexdigest()
    assert digest == report['firmware_sha256']
    machine = ref.Machine.from_program(elf.program(firmware))
    expected = machine.run(2_000_000)[1:]
    generated = np.fromfile(directory / "trace.u32", dtype="<u4").tolist()
    assert generated == expected, "saved tensor trace differs from independent reference"
    assert len(generated) == report['generated_tokens']
    output = bytes(spec.family(t)[1] for t in generated if spec.family(t)[0] == "BYTE")
    images = frames(output)
    assert images == json.loads((directory / "frames.json").read_text())
    assert len({f['sha256'] for f in images}) == report['distinct_frames']
    hf = json.loads((ROOT / "artifacts/hf-game-prefix/report.json").read_text())
    assert digest == hf['firmware_sha256'], "HF smoke and compact demo used different firmware"
    assert hf['tokens'] == generated[:hf['generated_tokens']]
    assert hf['mismatches'] == report['mismatches'] == 0
    print(f"saved compact execution: {len(generated)} exact tokens, {len(images)} verified frames")
    print(f"stock HF execution: {hf['generated_tokens']} exact prefix tokens, matching firmware SHA")
    cuda_dir = ROOT / "artifacts/hf-cuda-first-frame"
    cuda = json.loads((cuda_dir / "report.annotated.json").read_text())
    cuda_tokens = np.fromfile(cuda_dir / "trace.u32", dtype="<u4").tolist()
    assert cuda['completed_target'] and cuda['firmware_sha256'] == digest
    assert cuda_tokens == cuda['tokens'] == generated[:cuda['generated_tokens']]
    cuda_bytes = bytes(spec.family(t)[1] for t in cuda_tokens if spec.family(t)[0] == "BYTE")
    assert cuda_bytes == (cuda_dir / "actual-output.bin").read_bytes()
    assert frames(cuda_bytes) == json.loads((cuda_dir / "frames.json").read_text()) == images[:1]
    assert cuda['mismatches'] == 0 and cuda['frame_sha256'] == images[0]['sha256']
    print(f"stock HF CUDA: {len(cuda_tokens)} exact tokens and one complete matching frame")


if __name__ == "__main__":
    main()
