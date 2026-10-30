"""PowerShell wrapper for the shared, atomic GPU lock convention."""
LOCK = r'C:\Users\Admin\gpu-lock'


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
