"""Read bounded M5 progress and process evidence through encoded PowerShell."""
import argparse
import base64
import subprocess

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--label',required=True)
args=parser.parse_args()
assert args.label.replace('-','').isalnum()
base='C:/Users/Admin/qwenputer-games-work/artifacts/tiny-llm-ndot-'+args.label
script=f'''$ProgressPreference='SilentlyContinue'
$base='{base}'
tasklist /FI "IMAGENAME eq python.exe"
Get-ChildItem $base -Filter '*.exit.json' | ForEach-Object {{ $_.Name; Get-Content -Raw $_.FullName }}
foreach ($name in @('main/heartbeat.json','main/report.json','stock/report.json','sequence.json')) {{
 $p=Join-Path $base $name
 if(Test-Path $p) {{ $name; Get-Content -Raw $p | ConvertFrom-Json | Select-Object complete,status,generated_tokens,mismatches,tokens_per_second,compute_seconds,checkpoint_offset,output,error | ConvertTo-Json }}
}}
Get-ChildItem $base -Filter '*.stderr.log' | Where-Object {{$_.Length -gt 0}} | ForEach-Object {{ $_.Name; Get-Content -Tail 12 $_.FullName }}
'''
encoded=base64.b64encode(script.encode('utf-16-le')).decode()
subprocess.run(['ssh','gpu','powershell -NoProfile -EncodedCommand '+encoded],check=True)
