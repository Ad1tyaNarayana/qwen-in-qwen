import builtins
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import urllib.request
import tarfile

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"

def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

if __name__ == "__main__":
    Path("data").mkdir(exist_ok=True)
    archive = Path("data/TinyStories_all_data.tar.gz")
    if not archive.exists():
        with urllib.request.urlopen(URL, timeout=120) as response, open(str(archive) + ".partial", "wb") as out:
            total = 0
            while block := response.read(4 * 1024 * 1024):
                out.write(block)
                total += len(block)
                Path("heartbeat.json").write_text(json.dumps({"stage": "download", "bytes": total, "time": time.time()}))
                if shutil.disk_usage(ROOT).free < 12 * 1024**3:
                    raise RuntimeError("disk reserve reached")
        Path(str(archive) + ".partial").rename(archive)
    target = Path("data/TinyStories_all_data")
    target.mkdir(exist_ok=True)
    with tarfile.open(archive) as tar:
        for member in tar:
            if member.isfile() and member.name.endswith(".json"):
                dest = target / Path(member.name).name
                if not dest.exists():
                    if shutil.disk_usage(ROOT).free - member.size < 12 * 1024**3:
                        raise RuntimeError("disk reserve reached")
                    with tar.extractfile(member) as source, open(dest, "wb") as out:
                        shutil.copyfileobj(source, out)
                Path("heartbeat.json").write_text(json.dumps({"stage": "extract", "shard": str(dest), "time": time.time()}))
    manifest = {"url": URL, "archive_sha256": digest(archive), "archive_bytes": archive.stat().st_size, "shards": [{"file": p.name, "bytes": p.stat().st_size, "sha256": digest(p)} for p in sorted(target.glob("*.json"))]}
    Path("dataset.json").write_text(json.dumps(manifest, indent=2))
    print("DOWNLOAD_COMPLETE", flush=True)
