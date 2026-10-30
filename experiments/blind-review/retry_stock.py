import base64,subprocess,json
from pathlib import Path
H=Path(__file__).resolve().parent;R='C:/Users/Admin/qwenputer-games-work/artifacts/blind-review-ee2dd814'
def ps(s):
 r=subprocess.run(['ssh','gpu','powershell -NoProfile -EncodedCommand '+base64.b64encode(('$ProgressPreference="SilentlyContinue";'+s).encode('utf-16le')).decode()],capture_output=True,text=True);print(r.stdout,r.stderr);r.check_returncode()
ps("Set-Content C:/Users/Admin/gpu-lock/owner.txt 'blind-review stock retry'")
script=f'''$ErrorActionPreference='Stop'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTORCH_ALLOC_CONF='expandable_segments:True'
try {{
 cmd /c "C:/Users/Admin/qwenputer-games-work/.venv/Scripts/python.exe {R}/gpu_test.py --repo C:/Users/Admin/qwenputer-games-work/repo --upstream C:/Users/Admin/qwenputer-games-work/.upstream/qwenputer --out {R}/stock-retry --mode stock > {R}/stdout-stock-retry.log 2> {R}/stderr-stock-retry.log"
 @{{exit_code=$LASTEXITCODE}} | ConvertTo-Json | Set-Content '{R}/exit-stock-retry.json'
}} catch {{ $_ | Out-String | Set-Content '{R}/runner-error.txt' }} finally {{ cmd /c 'rmdir /s /q C:\\Users\\Admin\\gpu-lock' }}
'''
(H/'retry-runner.ps1').write_text(script)
encoded=base64.b64encode(script.encode('utf-16le')).decode();ps("Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='powershell -NoProfile -EncodedCommand "+encoded+"'} | ConvertTo-Json")
