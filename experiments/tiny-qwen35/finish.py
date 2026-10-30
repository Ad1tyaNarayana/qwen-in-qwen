"""Continue the milestone through retained, fail-closed verification gates."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from pack import ROOT, HERE, qwen
from build import link
from quality import quality
from operators import check as operators
from gate import gate
from launch_gpu import launch, REMOTE, PRIVATE
from verify_gpu import verify
from gpu_lock import locked
import base64

CACHE = ROOT/'.cache/tiny-qwen35'
TRAIN = 'C:/Users/Admin/tiny-llm-train/tiny-qwen35-20261030/result'


def progress(stage, **details):
    row = dict(stage=stage,utc=datetime.now(timezone.utc).isoformat(),**details)
    (CACHE/'progress.json').write_text(json.dumps(row,indent=2)+'\n')
    print(json.dumps(row),flush=True)


def remote_json(path):
    result = subprocess.run(['ssh','gpu','type '+path.replace('/','\\')],capture_output=True)
    return json.loads(result.stdout.decode('utf-8-sig')) if result.returncode == 0 else None


def wait_json(path, seconds):
    deadline = time.monotonic()+seconds
    while time.monotonic() < deadline:
        result = remote_json(path)
        if result is not None:
            return result
        time.sleep(30)
    raise TimeoutError(path)


def pull(remote, files, local):
    local.mkdir(parents=True,exist_ok=True)
    child = subprocess.Popen(['ssh','gpu','tar -cf - -C '+remote+' '+' '.join(files)],stdout=subprocess.PIPE)
    try:
        subprocess.run(['tar','-xf','-','-C',str(local)],stdin=child.stdout,check=True)
    finally:
        child.stdout.close()
    assert child.wait() == 0


def finish(label, mac_only, quantization, calibration):
    legacy = CACHE/'final-safe'
    progress('waiting-for-training')
    done = wait_json(TRAIN+'/done.json',4000)
    assert done['status'] == 'complete' and done['wall_seconds'] <= 3600
    training = CACHE/'training'
    pull(TRAIN,['hf','done.json','environment.json','samples.json','loss.jsonl','requirements-lock.txt'],training)
    for name,info in done['hf_files'].items():
        file = training/'hf'/name
        assert file.stat().st_size == info['bytes']
        assert hashlib.sha256(file.read_bytes()).hexdigest() == info['sha256']
    for name in ['done.json','environment.json','samples.json']:
        (HERE/'evidence'/('training-'+name)).write_bytes((training/name).read_bytes())
    progress('trained-quality',steps=done['steps'],training_seconds=done['training_seconds'])
    measurement = quality(training/'hf',legacy/'runtime/native',CACHE/'trained-quality',64,
                          quantization,calibration=calibration)
    program = CACHE/'trained-program'
    qwen(training/'hf',ROOT/'.cache/tiny-llm/custom/tok1024.bin',program,64,'Once',6,
         quantization,calibration)
    link(legacy/'runtime',program)
    operators(training/'hf',program)
    sanitizer = CACHE/'trained-sanitized'
    subprocess.run(['clang','-O1','-DNATIVE','-fsanitize=undefined,address',str(HERE/'runtime.c'),
                    '-o',str(sanitizer)],check=True)
    normal = subprocess.check_output([str(legacy/'runtime/native'),str(program/'model.bin')])
    result = subprocess.run([str(sanitizer),str(program/'model.bin')],capture_output=True,check=True)
    assert not result.stderr and result.stdout == normal
    (HERE/'evidence/trained-sanitizer.json').write_text(json.dumps(dict(status='pass',
        output_sha256=hashlib.sha256(normal).hexdigest(),asan=True,ubsan=True,fwrapv=False),indent=2)+'\n')
    progress('trained-software-reference',agreement=measurement['free_running_agreement'])
    if not (program/'reference.json').exists():
        subprocess.run([sys.executable,str(HERE/'reference.py'),'--build',str(program)],check=True)
    progress('waiting-for-legacy-reference')
    deadline = time.monotonic()+7200
    while not all((legacy/name/'reference.json').exists() for name in ['stories-0','custom-0']):
        if time.monotonic() > deadline:
            raise TimeoutError('Legacy reference did not finish')
        time.sleep(30)
    accepted = gate(legacy,program,CACHE/'trained-quality/quality.json')
    for source,target in [(legacy/'legacy-parity.json','legacy-parity.json'),
                           (legacy/'stories-0/reference.json','legacy-stories-reference.json'),
                           (legacy/'custom-0/reference.json','legacy-custom-reference.json'),
                           (program/'reference.json','qwen-reference.json'),
                           (program/'operators.json','trained-operators.json'),
                           (CACHE/'trained-quality/quality.json','trained-quality.json')]:
        (HERE/'evidence'/target).write_bytes(source.read_bytes())
    progress('mac-gates-pass',gate=accepted)
    if mac_only:
        return
    # Python wrapper processes can briefly remain after done.json is written.
    launch(legacy,program,CACHE/'trained-quality/quality.json',label)
    collect(label,program)


def collect(label,program):
    base = REMOTE+'/artifacts/tiny-qwen35-'+label
    progress('gpu-sequence-queued',remote=base)
    deadline = time.monotonic()+86400
    while remote_json(base+'/sequence.json') is None:
        failure = remote_json(base+'/sequence-failure.json')
        if failure is not None:
            raise RuntimeError(failure)
        for stage in ['prefix','restore','full','stock']:
            exit_result = remote_json(base+'/'+stage+'.exit.json')
            if exit_result is not None and exit_result['exit_code'] != 0:
                raise RuntimeError(dict(stage=stage,exit=exit_result,remote=base))
        if time.monotonic() > deadline:
            raise TimeoutError(base)
        time.sleep(30)
    progress('collecting-neural-results')
    export = f'{REMOTE}/.venv/Scripts/python.exe {PRIVATE}/experiments/tiny-llm-ndot/export.py --base {base}'
    action = f"& {export}\nif($LASTEXITCODE -ne 0) {{throw 'Checkpoint metadata export failed'}}\n"
    script = "$ErrorActionPreference='Stop'\n"+locked(action,'tiny-qwen35-'+label+'-export',15)
    encoded = base64.b64encode(script.encode('utf-16le')).decode()
    subprocess.run(['ssh','gpu','powershell -NoProfile -EncodedCommand '+encoded],check=True)
    files = ['main/'+name for name in ['report.json','prefix-report.json','restore-report.json','full-report.json',
             'actual.tokens.u32','actual-ram.bin','actual-output.bin','actual-registers.json','chunks.jsonl']]
    files += ['stock/report.json','stock/actual.tokens.u32','prefix.exit.json','restore.exit.json','full.exit.json',
              'stock.exit.json','checkpoint-metadata.json','sequence.json','tasklist.log','gpu-before.csv']
    actual = CACHE/('actual-'+label)
    pull(base,files,actual)
    recovery = remote_json(base+'/recovery-checkpoint.json')
    if recovery is not None:
        pull(base,['recovery-checkpoint.json',
                   'main/pre-shutdown/checkpoint.pt','main/pre-shutdown/actual.tokens.u32'],actual)
        # Keep the archive separate from the final worker's artifacts.
        (actual/'main/pre-shutdown').rename(actual/'pre-shutdown')
    result = verify(program,actual)
    progress('complete',result=result)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--label',required=True)
    p.add_argument('--mac-only',action='store_true')
    p.add_argument('--quantization',default='midpoint')
    p.add_argument('--calibration',type=Path)
    p.add_argument('--collect-only',action='store_true')
    a = p.parse_args()
    try:
        if a.collect_only:
            collect(a.label,CACHE/'trained-program')
        else:
            finish(a.label,a.mac_only,a.quantization,a.calibration.resolve() if a.calibration else None)
    except Exception as error:
        progress('failed',error=repr(error))
        raise
