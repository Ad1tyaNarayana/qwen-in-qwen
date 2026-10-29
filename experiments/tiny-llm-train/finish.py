import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import time
from export import load_checkpoint, model_export
from pipeline import run

os.chdir(Path(__file__).resolve().parent)
if __name__ == "__main__":
    started = time.time()
    run(["train.py", "train_config.py"], "train.log")
    training_seconds = time.time() - started
    model = load_checkpoint("out/ckpt.pt")
    Path("final").mkdir(exist_ok=True)
    model_export(model, "final/model.bin", version=0)
    model_export(model, "final/model-q8.bin", version=2)
    for name in ["tok1024.bin", "tok1024.model", "tok1024.vocab"]:
        shutil.copy2(Path("data") / name, Path("final") / name)
    run(["evaluate.py"], "evaluate.log")
    import torch
    checkpoint = torch.load("out/ckpt.pt", map_location="cpu", weights_only=False)
    Path("final/training-config.json").write_text(json.dumps(checkpoint["config"], indent=2))
    Path("final/model-args.json").write_text(json.dumps(checkpoint["model_args"], indent=2))
    report = {"training_process_seconds": training_seconds, "checkpoint_step": checkpoint["iter_num"], "parameter_count": sum(p.numel() for p in model.parameters()), "artifacts": {p.name: {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in Path("final").iterdir()}}
    Path("export-evidence.json").write_text(json.dumps(report, indent=2))
    for name in ["dataset.json", "evaluation.json", "environment.json", "requirements-lock.txt", "train_config.py", "export-evidence.json", "out/loss.jsonl", "train.log", "pilot.log"]:
        shutil.copy2(name, Path("final") / Path(name).name)
    Path("done.json").write_text(json.dumps({"time": time.time(), "training_process_seconds": training_seconds}))
