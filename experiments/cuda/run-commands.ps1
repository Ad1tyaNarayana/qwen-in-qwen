Copy-Item experiments\cuda\source\*.py tools\
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\python.exe -m pip install --no-cache-dir transformers==5.21.0 numpy==2.5.4 pillow==13.0.0
& 'C:\Program Files\Git\cmd\git.exe' clone https://github.com/divagr18/qwenputer.git .upstream\qwenputer
& 'C:\Program Files\Git\cmd\git.exe' -C .upstream\qwenputer checkout --detach ca13310e2f9f788d1dd3feeea8213c1b0c23913c
.\.venv\Scripts\python.exe tools\gpu_probe.py
.\.venv\Scripts\python.exe tools\hf_cuda.py --device cuda --output artifacts\hf-cuda-smoke
.\.venv\Scripts\python.exe tools\hf_cuda.py --device cuda --firmware artifacts\compact-game\weight-paddle.elf --max-tokens 16 --output artifacts\hf-cuda-game-prefix
.\.venv\Scripts\python.exe tools\hf_cuda_monitored.py --device cuda --firmware artifacts\compact-game\weight-paddle.elf --max-tokens 5415 --max-seconds 300 --output artifacts\hf-cuda-first-frame
