#!/usr/bin/env python3
"""Wait for this audit's finite GPU queue, collecting each completed stage."""
import subprocess,time,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
remote=r'C:\Users\Admin\qwenputer-games-work\artifacts\audit-1'
command=f'type {remote}\\status.txt & if exist {remote}\\deep-doom\\progress.txt (type {remote}\\deep-doom\\progress.txt)'
old=None
old_stages=None
start=time.monotonic()
while time.monotonic()-start<3*3600:
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','gpu',command],capture_output=True,text=True)
    if r.returncode:raise SystemExit(r.stderr)
    status=r.stdout.replace('\r','').strip()
    stages='\n'.join(line for line in status.splitlines() if not line.strip().isdigit())
    if status!=old:
        print(time.strftime('%H:%M:%S'),status,flush=True)
        if stages!=old_stages:
            subprocess.run([sys.executable,str(Path(__file__).with_name('collect_remote.py'))],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
            subprocess.run([sys.executable,str(Path(__file__).with_name('accept_pending.py'))],cwd=ROOT,check=True)
        old=status
        old_stages=stages
    else: print(time.strftime('%H:%M:%S'),'GPU audit queue remains active; waiting for raw completion output.',flush=True)
    if 'job4 lock released' in status:break
    time.sleep(45)
else:raise SystemExit('Audit queue did not finish within the three-hour wait budget')
