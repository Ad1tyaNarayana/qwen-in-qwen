#!/usr/bin/env python3
"""Attack 5: run the project's showcase verifier against mutated copies; evidence writes redirected to a temp HERE."""
import json,shutil,sys,tempfile,traceback
from pathlib import Path
R=Path('.').resolve()
sys.path.insert(0,str(R/'experiments/showcase'));sys.path.insert(0,str(R/'experiments/neural-accelerator'))
import verify_gpu as V
from prepare import prepare
tmp=Path(tempfile.mkdtemp(prefix='vm-',dir=R/'.cache/audit'))
prepare(tmp/'package');sys.path.insert(0,str(tmp/'package'))
from chipformer.rv import spec
here=tmp/'here';(here/'evidence').mkdir(parents=True)
shutil.copy(R/'experiments/showcase/evidence/weights.json',here/'evidence/');shutil.copy(R/'experiments/showcase/gpu_worker.py',here);shutil.copy(R/'experiments/showcase/gpu_stock.py',here)
V.HERE=here
A=R/'.cache/showcase/actual-20261030a'
def J(p,f):
    d=json.loads(p.read_text(encoding='utf-8-sig'));f(d);p.write_text(json.dumps(d))
def bit(rel,off):
    def m(o):
        p=o/'sha256'/rel;b=bytearray(p.read_bytes());b[off]^=1;p.write_bytes(b)
    return m
def jm(rel,f): return lambda o:J(o/'sha256'/rel,f)
muts={'none':lambda o:None,'token_mid':bit('fused/actual.tokens.u32',40000),'token_first':bit('fused/actual.tokens.u32',0),'stock_token_mid':bit('stock/actual.tokens.u32',8000),
 'ram_mid':bit('fused/actual-ram.bin',32768),'output':bit('fused/actual-output.bin',5),
 'reg':jm('fused/actual-registers.json',lambda d:d['regs.rd'].__setitem__(5,d['regs.rd'][5]^1)),
 'mismatches':jm('fused/report.json',lambda d:d.__setitem__('mismatches',1)),
 'weight_label':jm('fused/report.json',lambda d:d.__setitem__('network_weight_sha256','0'*64)),
 'stock_weight':jm('stock/report.json',lambda d:d['model'].__setitem__('network_weight_sha256','0'*64)),
 'used_ref':jm('stock/report.json',lambda d:d.__setitem__('reference_tokens_used_as_inputs',True)),
 'chunk':lambda o:(lambda p:p.write_text(p.read_text().replace('"sha256": "','"sha256": "0',1)))(o/'sha256/fused/chunks.jsonl'),
 'exit':lambda o:(lambda p:p.write_text('{"exit_code": 1}'))(o/'sha256-full.exit.json'),
 'time_scale':jm('fused/report.json',lambda d:d.__setitem__('compute_seconds',d['compute_seconds']/2)),
 'reference_copied_over_actual':lambda o:shutil.copy(R/'.cache/showcase/sha256/reference.tokens.u32',o/'sha256/fused/actual.tokens.u32')}
res={}
for n,m in muts.items():
    o=tmp/n;o.mkdir();shutil.copytree(A/'sha256',o/'sha256')
    for p in A.glob('sha256-*.exit.json'):shutil.copy(p,o/p.name)
    m(o)
    try:
        V.verify(o,'sha256',spec);res[n]='ACCEPTED'
    except AssertionError as e:
        tb=traceback.extract_tb(e.__traceback__)[-1];res[n]=f'rejected at verify_gpu.py:{tb.lineno}'
    except Exception as e: res[n]='error '+repr(e)[:120]
print(json.dumps(res,indent=1));Path('experiments/audit/results/verifier-mutations.json').write_text(json.dumps(res,indent=1))
shutil.rmtree(tmp,ignore_errors=True)
