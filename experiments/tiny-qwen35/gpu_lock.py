"""PowerShell wrapper for the shared, atomic GPU lock convention."""
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[2] / 'tools'))
from remote_config import SSH_HOST, REMOTE_ROOT, LOCK, TRAIN_ROOT


def locked(action, owner, expected_minutes):
    assert owner.replace('-','').isalnum()
    assert 0 < expected_minutes <= 240
    return f'''
while($true) {{
 cmd.exe /d /c 'mkdir {LOCK} 2>nul'
 if($LASTEXITCODE -eq 0) {{break}}
 Start-Sleep -Seconds 30
}}
try {{
 ('{owner}; start='+[DateTime]::UtcNow.ToString('o')+'; expected <= {expected_minutes} minutes') | Set-Content '{LOCK}\\owner.txt'
 if(Get-Process python -ErrorAction SilentlyContinue) {{throw 'A Python job exists despite acquiring the GPU lock'}}
 {action}
}} finally {{
 $ownerText=Get-Content '{LOCK}\\owner.txt' -Raw -ErrorAction SilentlyContinue
 if(!$ownerText -or $ownerText -like '{owner}*') {{
  cmd.exe /d /c 'rmdir /s /q {LOCK}'
 }}
}}
'''
