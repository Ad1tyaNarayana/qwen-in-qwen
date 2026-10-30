#!/usr/bin/env python3
"""Accept completed audit jobs from raw exports, without treating missing jobs as passes."""
import hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'.cache/audit/remote'
RESULT=ROOT/'experiments/audit/results'
sys.path.insert(0,str(Path(__file__).parent));import rvi
SHA='0e515bbb3109997a130c903deb40dce3447f8991e15b6637e0071c71f8b6e07d'
checks={}
def save(name,d):
    (RESULT/(name+'.json')).write_text(json.dumps(d,indent=2)+'\n')
for mode in ('unfused','fuzz','safe-fused'):
    p=BASE/mode/'report.json'
    if not p.exists():continue
    d=json.loads(p.read_text())
    if not d.get('done'):continue
    out=RESULT/(mode+'-vs-rvi.json')
    subprocess.run([sys.executable,str(Path(__file__).parent/'cmp_gpu.py'),str(p.parent),str(out)],check=True,cwd=ROOT,stdout=subprocess.DEVNULL)
    v=json.loads(out.read_text());assert v['start']==v['end']
    assert all(g['equal'] and g['ram_equal'] and g['regs']==[] for g in v['guests'].values()),mode
    if mode=='safe-fused':assert v['guests']['fresh-safe']['fault'] is None and v['guests']['fresh-safe']['exit']==0
    if mode in ('fuzz','safe-fused'):
        for name,guest in d['guests'].items():
            ref=json.loads((ROOT/'.cache/audit/guests'/name/'cmp/result.json').read_text())
            assert ref['project_sha256']==guest['trace_sha256'], name
        v['complete_project_trace_equal']=True
        save(mode+'-vs-rvi',v)
    checks[mode]=dict(passed=True,guests=len(v['guests']))
    if mode=='unfused':
        fused=json.loads((BASE/'fused/report.json').read_text())
        for name,guest in d['guests'].items():
            other=fused['guests'][name]
            assert guest['trace_sha256']==other['trace_sha256']
            assert guest['state_sha256']==other['state_sha256'], name
        checks[mode]['full_state_digest_equal_to_fused']=True
stock={
    'stock':('fresh',ROOT/'.cache/audit/fresh/cmp/project.tokens.u32'),
    'stock-deflt':('fresh',ROOT/'.cache/audit/fresh/cmp/project.tokens.u32'),
    'stock-fuzz':('fuzz1101',ROOT/'.cache/audit/guests/fuzz1101/cmp/project.tokens.u32'),
    'safe-stock':('fresh-safe',ROOT/'.cache/audit/guests/fresh-safe/cmp/project.tokens.u32'),
    'stock-qwen-deep':('qwen35',ROOT/'.cache/tiny-qwen35/actual-covariance-20261030/main/actual.tokens.u32'),
    'stock-doom-deep':('doom',ROOT/'.cache/doom-showcase/export-unified-full/actual.tokens.u32'),
}
# Resolve the retained full trace paths from the existing independent comparison inputs.
for name,(guest,ref) in stock.items():
    p=BASE/name/'report.json'
    if not p.exists():continue
    d=json.loads(p.read_text())
    if not d.get('done'):continue
    assert d['weight_sha256']==SHA and d['type_exact'] and d['hooks']==0 and not d['instance_overrides']
    actual=np.fromfile(p.parent/'tokens.u32',dtype='<u4');assert len(actual)==d['generated']
    offset=d['seed_token_offset']
    if not ref.exists():
        checks[name]=dict(passed=False,needs_reference_path=str(ref),guest=guest);continue
    expected=np.memmap(ref,dtype='<u4',mode='r')[offset:offset+len(actual)]
    assert len(expected)==len(actual) and np.array_equal(actual,expected),name
    assert hashlib.sha256(actual.tobytes()).hexdigest()==d['trace_sha256']
    q0,qn=rvi.BASE['QS'],rvi.BASE['QD']+128
    if offset==0:
        c=rvi.CPU(str(ROOT/'.cache/audit/guests'/guest/'engine.elf')).run()
        mine=np.frombuffer(c.tokens,dtype='<u4');relevant=actual[~((actual>=q0)&(actual<qn))]
        assert np.array_equal(relevant,mine[:len(relevant)])
    result=dict(passed=True,guest=guest,offset=offset,tokens=len(actual),loaded_weight_sha256=d['weight_sha256'],report=d,
                comparison='raw generated tokens vs reference; boot windows also checked against independent interpreter without division digits',
                boot_only=offset==0,reference_seeded=offset!=0)
    save(name+'-validation',result);checks[name]=dict(passed=True,tokens=len(actual),offset=offset)
for name in ('ckpt','deep-doom'):
    p=BASE/name/'report.json'
    if not p.exists():continue
    d=json.loads(p.read_text());save(name,d)
    if name=='deep-doom':
        assert d['tokens']==d['checkpoint_offset']==6342933 and 'unfused' in d['mode']
        assert d['trace_sha256']==hashlib.sha256((ROOT/'.cache/doom-showcase/export-unified-full/actual.tokens.u32').read_bytes()).hexdigest()
        assert sum(d['words_compared'].values())>0
        assert not any(d['differing_words'].values()) and d['x_equal'] and d['tok_equal'] and d['scratch_y_zero']
        checks[name]=dict(passed=True,words=d['words_compared'])
    else:checks[name]=dict(results=d)
save('pending-validation',checks)
print('Accepted GPU stages: '+', '.join(checks)+'; missing stages are not counted.')
