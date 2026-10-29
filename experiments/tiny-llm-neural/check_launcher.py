"""Exercise success and failure exit propagation through detached-process handles."""
import base64
import json
from pathlib import Path
import subprocess

script = '''$ErrorActionPreference='Stop'; $rows=@(); foreach($code in @(0,7)) {
$arg='-c "import time; time.sleep(0.2); raise SystemExit('+ $code +')"'
$p=Start-Process -FilePath 'C:/Users/Admin/qwenputer-games-work/.venv/Scripts/python.exe' -ArgumentList $arg -PassThru
$handle=$p.Handle
$p.WaitForExit()
$rows+=@{expected=$code; actual=$p.ExitCode; has_exited=$p.HasExited}
}; $rows | ConvertTo-Json'''
encoded = base64.b64encode(script.encode('utf-16-le')).decode()
result = subprocess.run(['ssh','gpu','powershell -NoProfile -EncodedCommand '+encoded],capture_output=True,check=True)
rows = json.loads(result.stdout)
assert len(rows) == 2
assert all(r['expected'] == r['actual'] and r['has_exited'] for r in rows)
report = dict(controls=rows,fix='Retain Process.Handle before WaitForExit so ExitCode remains available',
              historical_hf_exit_code=None,historical_exit_code_not_inferred=True)
(Path(__file__).parent / 'evidence/launcher-exit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
