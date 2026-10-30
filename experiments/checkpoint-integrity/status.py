"""Read-only status for this task's detached GPU batch, via ssh gpu only."""
import base64
import subprocess
script="""$ProgressPreference='SilentlyContinue'
$base='C:/Users/Admin/qwenputer-games-work/artifacts/checkpoint-integrity-20261030a'
foreach($name in @('sequence.json','mutations/report.json','scalar/heartbeat.json','scalar/report.json','ndot/heartbeat.json','ndot/report.json')) {
 $path=$base+'/'+$name
 if(Test-Path $path) {$name;Get-Content -Raw $path | ConvertFrom-Json | Select-Object complete,status,generated_tokens,total_tokens,mismatches,compute_seconds,tokens_per_second,checkpoint_offset,updated_utc,error | ConvertTo-Json}
}
Get-CimInstance Win32_Process | Where-Object {$_.Name -eq 'python.exe'} | Select-Object ProcessId,CommandLine | ConvertTo-Json
"""
subprocess.run(['ssh','-O','check','gpu'],check=True)
subprocess.run(['ssh','gpu','powershell -NoProfile -EncodedCommand '+base64.b64encode(script.encode('utf-16-le')).decode()],check=True)
