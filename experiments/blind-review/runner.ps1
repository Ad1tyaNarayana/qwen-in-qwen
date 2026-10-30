$ErrorActionPreference='Stop'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTORCH_ALLOC_CONF='expandable_segments:True'
try {
 foreach($mode in @('fused','stock')) {
  & 'C:/Users/Admin/qwenputer-games-work/.venv/Scripts/python.exe' 'C:/Users/Admin/qwenputer-games-work/artifacts/blind-review-ee2dd814/gpu_test.py' --repo C:/Users/Admin/qwenputer-games-work/repo --upstream C:/Users/Admin/qwenputer-games-work/.upstream/qwenputer --out ("C:/Users/Admin/qwenputer-games-work/artifacts/blind-review-ee2dd814/"+$mode) --mode $mode 1> ("C:/Users/Admin/qwenputer-games-work/artifacts/blind-review-ee2dd814/stdout-"+$mode+".log") 2> ("C:/Users/Admin/qwenputer-games-work/artifacts/blind-review-ee2dd814/stderr-"+$mode+".log")
  @{exit_code=$LASTEXITCODE} | ConvertTo-Json | Set-Content ("C:/Users/Admin/qwenputer-games-work/artifacts/blind-review-ee2dd814/exit-"+$mode+".json")
 }
} finally { cmd /c 'rmdir /s /q C:\Users\Admin\gpu-lock' }
