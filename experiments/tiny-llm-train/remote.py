import argparse
import base64
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("powershell")
args = parser.parse_args()
encoded = base64.b64encode(("$ProgressPreference='SilentlyContinue'; " + args.powershell).encode("utf-16le")).decode()
raise SystemExit(subprocess.run(["ssh", "gpu", "powershell -NoProfile -EncodedCommand " + encoded]).returncode)
