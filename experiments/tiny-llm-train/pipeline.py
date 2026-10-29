import builtins
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
PY = str(ROOT / ".venv/Scripts/python.exe")

def run(args, log):
    with open(log, "w", encoding="utf-8") as output:
        process = subprocess.Popen([PY, "-u", *args], stdout=output, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONUTF8": "1"})
        while process.poll() is None:
            Path("heartbeat.json").write_text(json.dumps({"stage": log, "pid": process.pid, "time": time.time()}))
            time.sleep(5)
        if process.returncode:
            raise RuntimeError(f"{args} exited {process.returncode}; see {log}")

if __name__ == "__main__":
    try:
        import torch
        torch.set_num_threads(4)
        Path("environment.json").write_text(json.dumps({"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(), "bf16": torch.cuda.is_bf16_supported()}, indent=2))
        subprocess.run([PY, "-m", "pip", "freeze"], stdout=open("requirements-lock.txt", "w"), check=True)
        while not Path("dataset.json").exists():
            time.sleep(5)
        import tinystories
        if not Path("data/tok1024.model").exists():
            builtins.input = lambda _: "y"
            tinystories.train_vocab(1024)
        from tokenizer import Tokenizer
        Tokenizer("data/tok1024.model").export()
        tinystories.pretokenize(1024)
        run(["train.py", "train_config.py", "--out_dir=pilot", "--max_iters=50", "--eval_interval=50", "--eval_iters=5", "--log_interval=10"], "pilot.log")
        Path("pilot_done.json").write_text(json.dumps({"time": time.time()}))
    except BaseException:
        Path("pipeline_error.txt").write_text(traceback.format_exc())
        raise
