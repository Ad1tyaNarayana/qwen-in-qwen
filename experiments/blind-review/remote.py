import base64,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
REMOTE='C:/Users/Admin/qwenputer-games-work/artifacts/blind-review-ee2dd814'
def ps(s):
 r=subprocess.run(['ssh','gpu','powershell -NoProfile -EncodedCommand '+base64.b64encode(('$ProgressPreference="SilentlyContinue";'+s).encode('utf-16le')).decode()],capture_output=True,text=True);print(r.stdout);print(r.stderr,file=sys.stderr);r.check_returncode();return r.stdout
if sys.argv[1]=='launch':
 ps(f"New-Item -ItemType Directory -Path '{REMOTE}' -ErrorAction Stop | Out-Null")
 pack=subprocess.run(['tar','-cf','-','-C',str(HERE),'gpu_test.py'],capture_output=True,check=True).stdout
 subprocess.run(['ssh','gpu',f'tar -xf - -C {REMOTE}'],input=pack,check=True)
 script=f'''$ErrorActionPreference='Stop'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTORCH_ALLOC_CONF='expandable_segments:True'
try {{
 foreach($mode in @('fused','stock')) {{
  & 'C:/Users/Admin/qwenputer-games-work/.venv/Scripts/python.exe' '{REMOTE}/gpu_test.py' --repo C:/Users/Admin/qwenputer-games-work/repo --upstream C:/Users/Admin/qwenputer-games-work/.upstream/qwenputer --out '{REMOTE}/'$mode --mode $mode 1> '{REMOTE}/stdout-'$mode'.log' 2> '{REMOTE}/stderr-'$mode'.log'
  @{{exit_code=$LASTEXITCODE}} | ConvertTo-Json | Set-Content '{REMOTE}/exit-'$mode'.json'
 }}
}} finally {{ cmd /c 'rmdir /s /q C:\\Users\\Admin\\gpu-lock' }}
'''
 # Use Join-Path variables to avoid PowerShell token concatenation ambiguities.
 script=script.replace("--out '"+REMOTE+"/'$mode",'--out ("'+REMOTE+'/"+$mode)').replace("1> '"+REMOTE+"/stdout-'$mode'.log'",'1> ("'+REMOTE+'/stdout-"+$mode+".log")').replace("2> '"+REMOTE+"/stderr-'$mode'.log'",'2> ("'+REMOTE+'/stderr-"+$mode+".log")').replace("Set-Content '"+REMOTE+"/exit-'$mode'.json'",'Set-Content ("'+REMOTE+'/exit-"+$mode+".json")')
 (HERE/'runner.ps1').write_text(script)
 encoded=base64.b64encode(script.encode('utf-16le')).decode()
 ps("Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='powershell -NoProfile -EncodedCommand "+encoded+"'} | ConvertTo-Json")
elif sys.argv[1]=='status':ps(f"Get-ChildItem '{REMOTE}'; foreach($m in @('fused','stock')){{Get-Content (\"{REMOTE}/\"+$m+'/report.json') -ErrorAction SilentlyContinue; Get-Content (\"{REMOTE}/stderr-\"+$m+'.log') -Tail 15 -ErrorAction SilentlyContinue}}")
elif sys.argv[1]=='collect':
 r=subprocess.run(['ssh','gpu',f'tar -cf - -C {REMOTE} fused/report.json fused/actual.tokens.u32 fused/ram.bin stock/report.json stock-retry/report.json stock-retry/actual.tokens.u32 stdout-fused.log stderr-fused.log stdout-stock.log stderr-stock.log stdout-stock-retry.log stderr-stock-retry.log exit-fused.json exit-stock-retry.json'],capture_output=True)
 (HERE/'gpu-results.tar').write_bytes(r.stdout);print(r.stderr.decode());r.check_returncode()
 (HERE/'gpu-results').mkdir(exist_ok=True);subprocess.run(['tar','-xf',str(HERE/'gpu-results.tar'),'-C',str(HERE/'gpu-results')],check=True)
