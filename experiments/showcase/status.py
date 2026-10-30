"""Read this batch's exits and active stdout without accessing another task."""
import json
from pathlib import Path
from launch import ROOT,ps

if __name__=='__main__':
    remote=json.loads((ROOT/'.cache/showcase/remote.json').read_text())
    base=remote['base']
    print(ps(f'''$ProgressPreference='SilentlyContinue'
$done=@(Get-ChildItem '{base}' -Filter '*.exit.json' | ForEach-Object {{
 @{{name=$_.BaseName;result=(Get-Content $_.FullName -Raw | ConvertFrom-Json)}}
}})
$active=@(Get-ChildItem '{base}' -Filter '*.stdout.log' | Where-Object {{
 -not (Test-Path ($_.FullName.Replace('.stdout.log','.exit.json')))
}} | ForEach-Object {{@{{name=$_.BaseName;last_line=[string](Get-Content $_.FullName -Tail 1)}}}})
@{{finished=$done;active=$active;sequence=if(Test-Path '{base}/sequence.json') {{Get-Content '{base}/sequence.json' -Raw | ConvertFrom-Json}} else {{$null}}}} | ConvertTo-Json -Depth 6
'''))
