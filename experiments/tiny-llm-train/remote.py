import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[2] / 'tools'))
from remote_config import SSH_HOST, REMOTE_ROOT, LOCK, TRAIN_ROOT
import argparse
import base64
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("powershell")
args = parser.parse_args()
encoded = base64.b64encode(("$ProgressPreference='SilentlyContinue'; " + args.powershell).encode("utf-16le")).decode()
raise SystemExit(subprocess.run(["ssh", SSH_HOST, "powershell -NoProfile -EncodedCommand " + encoded]).returncode)
