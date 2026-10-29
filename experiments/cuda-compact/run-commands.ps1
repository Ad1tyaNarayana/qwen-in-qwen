Copy-Item experiments\cuda-compact\source\*.py tools\
Copy-Item artifacts\compact-game\weight-paddle.elf artifacts\weight-paddle.elf
Copy-Item experiments\cuda-compact\artifacts\triton-requirement.txt artifacts\triton-requirement.txt
.\.venv\Scripts\python.exe -m pip freeze > artifacts\acceleration-before.txt
.\.venv\Scripts\python.exe -m pip install --no-cache-dir --no-deps --require-hashes -r artifacts\triton-requirement.txt
.\.venv\Scripts\python.exe -m pip freeze > artifacts\acceleration-after.txt
.\.venv\Scripts\python.exe tools\acceleration_probe.py --v2
.\.venv\Scripts\python.exe tools\acceleration_probe.py
.\.venv\Scripts\python.exe tools\acceleration_compare.py
