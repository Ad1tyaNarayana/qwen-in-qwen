"""Launch through WMI only when the GPU PC has no Python processes."""
import base64
import json
from pathlib import Path
import subprocess
from gpu_lock import locked

ROOT = r'C:\Users\Admin\tiny-llm-train'
DIRECTORY = ROOT+r'\tiny-qwen35-20261030'


def launch():
    script = r'''
$ErrorActionPreference = 'Stop'
$directory = 'DIRECTORY'
if (Test-Path ($directory+'\result')) { throw 'Refusing to overwrite a previous training run' }
$command = 'cmd.exe /d /c ROOT\.venv\Scripts\python.exe -u DIRECTORY\train.py --root ROOT --config DIRECTORY\train_config.json --out DIRECTORY\result > DIRECTORY\train.log 2>&1'
'''.replace('DIRECTORY',DIRECTORY).replace('ROOT',ROOT)
    action = script+'''$p=Start-Process -FilePath 'cmd.exe' -ArgumentList $command.Substring('cmd.exe '.Length) -WorkingDirectory $directory -PassThru
$handle=$p.Handle
$p.WaitForExit()
if($p.ExitCode -ne 0) {throw ('Training failed '+$p.ExitCode)}
'''
    child = base64.b64encode(("$ErrorActionPreference='Stop'\n"+
        locked(action,'tiny-qwen35-training-20261030',60)).encode('utf-16le')).decode()
    script = "Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='powershell -NoProfile -EncodedCommand "+child+"'} | ConvertTo-Json"
    encoded = base64.b64encode(script.encode('utf-16le')).decode()
    result = subprocess.run(['ssh','gpu','powershell -NoProfile -NonInteractive -EncodedCommand '+encoded],
                            capture_output=True,text=True,check=True)
    print(result.stdout)
    evidence = Path(__file__).with_name('evidence')/'training-launch.json'
    evidence.write_text(json.dumps(dict(stdout=result.stdout,stderr=result.stderr,returncode=result.returncode),indent=2)+'\n')


if __name__ == '__main__':
    launch()
