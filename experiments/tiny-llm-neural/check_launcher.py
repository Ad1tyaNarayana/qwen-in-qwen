"""Exercise success and failure exit propagation through detached-process handles."""
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[2] / 'tools'))
from remote_config import SSH_HOST, REMOTE_ROOT, LOCK, TRAIN_ROOT
import base64
import json
from pathlib import Path
import subprocess

script = ('$ErrorActionPreference=\'Stop\'; $rows=@(); foreach($code in @(0,7)) {\n$arg=\'-c "import time; time.sleep(0.2); raise SystemExit(\'+ $code +\')"\'\n$p=Start-Process -FilePath \'' + REMOTE_ROOT + "/.venv/Scripts/python.exe' -ArgumentList $arg -PassThru\n$handle=$p.Handle\n$p.WaitForExit()\n$rows+=@{expected=$code; actual=$p.ExitCode; has_exited=$p.HasExited}\n}; $rows | ConvertTo-Json")
encoded = base64.b64encode(script.encode('utf-16-le')).decode()
result = subprocess.run(['ssh',SSH_HOST,'powershell -NoProfile -EncodedCommand '+encoded],capture_output=True,check=True)
rows = json.loads(result.stdout)
assert len(rows) == 2
assert all(r['expected'] == r['actual'] and r['has_exited'] for r in rows)
report = dict(controls=rows,fix='Retain Process.Handle before WaitForExit so ExitCode remains available',
              historical_hf_exit_code=None,historical_exit_code_not_inferred=True)
(Path(__file__).parent / 'evidence/launcher-exit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
