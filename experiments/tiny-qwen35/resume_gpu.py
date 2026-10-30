"""Resume a power-interrupted run from its retained actual GPU checkpoint."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess

from gpu_lock import LOCK, locked
from launch_gpu import REMOTE, PRIVATE
from pack import ROOT, HERE


def resume(label, checkpoint):
    assert label.replace('-', '').isalnum()
    saved = json.loads(checkpoint.read_text())
    assert saved['status'] == 'valid actual GPU checkpoint' and saved['trace_mismatches'] == 0
    base = REMOTE+'/artifacts/tiny-qwen35-'+label
    owner = 'tiny-qwen35-'+label+'-full'
    offset = saved['offset']
    program = PRIVATE+'/.cache/tiny-qwen35/trained-program'
    action = f'''
if((Get-Content '{LOCK}\\owner.txt' -Raw).Trim() -notlike '{owner};*') {{throw 'Existing lock is not ours'}}
if(Get-Process python -ErrorAction SilentlyContinue) {{throw 'Python job still running'}}
if((Get-FileHash '{base}/main/checkpoint.pt' -Algorithm SHA256).Hash.ToLower() -ne '{saved['checkpoint_sha256']}') {{throw 'Checkpoint changed'}}
if((Get-FileHash '{base}/main/actual.tokens.u32' -Algorithm SHA256).Hash.ToLower() -ne '{saved['actual_trace_sha256']}') {{throw 'Stopped trace changed'}}
if(Test-Path '{base}/main/pre-shutdown') {{
 if((Get-FileHash '{base}/main/pre-shutdown/checkpoint.pt' -Algorithm SHA256).Hash.ToLower() -ne '{saved['checkpoint_sha256']}') {{throw 'Recovery archive differs'}}
}} else {{
 New-Item -ItemType Directory '{base}/main/pre-shutdown' | Out-Null
 foreach($name in @('checkpoint.pt','heartbeat.json','report.json','chunks.jsonl','actual.tokens.u32')) {{
  Copy-Item ('{base}/main/'+$name) '{base}/main/pre-shutdown/'
 }}
}}
if(Test-Path '{base}/sequence-failure.json') {{
 if(Test-Path '{base}/recovery-startup-failure.json') {{throw 'Prior recovery failure already archived'}}
 Move-Item '{base}/sequence-failure.json' '{base}/recovery-startup-failure.json'
 Copy-Item '{base}/main/report.json' '{base}/recovery-startup-report.json'
 Copy-Item '{base}/full-resumed.stdout.log' '{base}/recovery-startup.stdout.log'
}}
if(Test-Path '{base}/main/scratch') {{Remove-Item -Recurse -Force '{base}/main/scratch'}}
Copy-Item '{PRIVATE}/experiments/tiny-qwen35/evidence/recovery-checkpoint.json' '{base}/recovery-checkpoint.json'
$rows=@(Get-Content '{base}/main/chunks.jsonl' | Where-Object {{($_ | ConvertFrom-Json).end_offset -le {offset}}})
if(($rows[-1] | ConvertFrom-Json).end_offset -ne {offset}) {{throw 'No chunk boundary at checkpoint'}}
[IO.File]::WriteAllText('{base}/main/chunks.jsonl',($rows -join "`n")+"`n",(New-Object Text.UTF8Encoding($false)))
tasklist /FI "IMAGENAME eq python.exe" /FO CSV | Out-File -Append '{base}/tasklist.log'
$start=[DateTime]::Parse(((Get-Content '{LOCK}\\owner.txt' -Raw) -split 'start=')[1].Split(';')[0]).ToUniversalTime()
$remaining=[int](($start.AddMinutes(235)-[DateTime]::UtcNow).TotalSeconds)-30
if($remaining -lt 600) {{throw 'Insufficient time left in existing GPU hold'}}
$p=Start-Process -FilePath '{REMOTE}/.venv/Scripts/python.exe' -ArgumentList ('{PRIVATE}/experiments/tiny-llm-ndot-fused/story.py --repo {PRIVATE} --upstream {REMOTE}/.upstream/qwenputer --inputs {program} --out {base}/main --max-seconds '+$remaining) -WorkingDirectory '{REMOTE}' -RedirectStandardOutput '{base}/full-resumed.stdout.log' -RedirectStandardError '{base}/full-resumed.stderr.log' -PassThru
$handle=$p.Handle
$p.WaitForExit()
@{{exit_code=$p.ExitCode;finished_utc=[DateTime]::UtcNow.ToString('o');resumed=$true}} | ConvertTo-Json | Set-Content '{base}/full.exit.json'
if($p.ExitCode -ne 0) {{throw ('Resumed full stage failed: '+$p.ExitCode)}}
Copy-Item '{base}/main/report.json' '{base}/main/full-report.json'
'''
    script = f"$ErrorActionPreference='Stop'\ntry {{\ntry {{\n{action}\n}} finally {{\n"
    script += f"if((Get-Content '{LOCK}\\owner.txt' -Raw -ErrorAction SilentlyContinue) -like '{owner}*') {{cmd.exe /d /c 'rmdir /s /q {LOCK}'}}\n}}\n"
    stock = f'''
$env:PYTORCH_ALLOC_CONF='expandable_segments:True'
$p=Start-Process -FilePath '{REMOTE}/.venv/Scripts/python.exe' -ArgumentList '{PRIVATE}/experiments/tiny-llm-ndot/stock_prefix.py --upstream {REMOTE}/.upstream/qwenputer --inputs {program} --out {base}/stock --tokens 16384' -WorkingDirectory '{REMOTE}' -RedirectStandardOutput '{base}/stock.stdout.log' -RedirectStandardError '{base}/stock.stderr.log' -PassThru
$handle=$p.Handle
$p.WaitForExit()
@{{exit_code=$p.ExitCode;finished_utc=[DateTime]::UtcNow.ToString('o')}} | ConvertTo-Json | Set-Content '{base}/stock.exit.json'
if($p.ExitCode -ne 0) {{throw ('Stock stage failed: '+$p.ExitCode)}}
'''
    script += locked(stock,'tiny-qwen35-'+label+'-stock',60)
    script += f"@{{complete=$true}} | ConvertTo-Json | Set-Content '{base}/sequence.json'\n"
    script += f"}} catch {{ @{{complete=$false;error=($_ | Out-String)}} | ConvertTo-Json | Set-Content '{base}/sequence-failure.json'; throw }}\n"
    launcher = ROOT/'.cache/tiny-qwen35'/('resume-'+label+'.ps1')
    launcher.write_text(script)
    for source, target in [(launcher,PRIVATE+'/'+launcher.relative_to(ROOT).as_posix()),
                           (checkpoint,PRIVATE+'/experiments/tiny-qwen35/evidence/recovery-checkpoint.json')]:
        subprocess.run(['scp',str(source),'gpu:'+target],check=True)
    target = PRIVATE+'/'+launcher.relative_to(ROOT).as_posix()
    # Parse first; no GPU process is started by this check.
    parse = f"$e=$null;$t=$null;[Management.Automation.Language.Parser]::ParseFile('{target}',[ref]$t,[ref]$e)|Out-Null;if($e.Count){{$e|Out-String;exit 1}}"
    encoded = base64.b64encode(parse.encode('utf-16le')).decode()
    subprocess.run(['ssh','gpu','powershell -NoProfile -EncodedCommand '+encoded],check=True)
    child = base64.b64encode(f"& '{target}'".encode('utf-16le')).decode()
    command = "Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='powershell -NoProfile -ExecutionPolicy Bypass -EncodedCommand "+child+"'} | ConvertTo-Json"
    encoded = base64.b64encode(command.encode('utf-16le')).decode()
    result = subprocess.run(['ssh','gpu','powershell -NoProfile -EncodedCommand '+encoded],capture_output=True,text=True,check=True)
    assert json.loads(result.stdout)['ReturnValue'] == 0,result.stdout
    evidence = dict(offset=offset,checkpoint_sha256=saved['checkpoint_sha256'],
                    launcher_sha256=hashlib.sha256(launcher.read_bytes()).hexdigest(),
                    existing_lock_retained=True,stdout=result.stdout)
    (HERE/'evidence/recovery-launch.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(evidence,indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--label',required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    a = p.parse_args()
    resume(a.label,a.checkpoint)
