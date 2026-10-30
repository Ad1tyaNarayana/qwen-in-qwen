#!/usr/bin/env python3
"""Attack 7: recompute published numbers from raw evidence and raw trace files."""
import json,hashlib,statistics as st
from pathlib import Path
import numpy as np
L=lambda p:json.load(open(p))
R=[]
def ck(claim,pub,val,src,tol=0.006,kind='measured'):
    ok=abs(float(pub)-float(val))<=abs(float(pub))*tol if float(pub) else float(val)==0
    R.append(dict(claim=claim,published=pub,recomputed=val,ok=bool(ok),source=src,kind=kind))
D='experiments/doom-showcase/evidence/'
h=L('experiments/dmc-cuda/evidence/full-report.json');f=L(D+'ff-full-report.json');u=L(D+'unified-full-report.json')
fc=[json.loads(x) for x in open(D+'ff-full-chunks.jsonl')];uc=[json.loads(x) for x in open(D+'unified-full-chunks.jsonl')]
trace=np.fromfile('.cache/doom-showcase/export-unified-full/actual.tokens.u32',dtype='<u4')
ck('Doom tokens',6342933,len(trace),'raw trace file')
ck('hist generation s',3044.57,h['compute_seconds_total'],'full-report')
ck('run1 generation s',1446.28,sum(c['seconds'] for c in fc),'ff chunks sum')
ck('run2 generation s',1693.11,sum(c['seconds'] for c in uc),'unified chunks sum')
ck('2.105x',2.105,h['compute_seconds_total']/sum(c['seconds'] for c in fc),'ratio',kind='derived')
ck('1.798 cross-network',1.798,h['compute_seconds_total']/sum(c['seconds'] for c in uc),'ratio',kind='derived, unlike networks')
ck('run1 tok/s',4385.70,6342933/sum(c['seconds'] for c in fc),'')
ck('run2 tok/s',3746.33,6342933/sum(c['seconds'] for c in uc),'')
ck('hist tok/s',2083.36,6342933/h['compute_seconds_total'],'')
ck('run1 wall',1469.41,f['uninterrupted_end_to_end_seconds'],'report')
ck('run2 wall',1715.49,u['uninterrupted_end_to_end_seconds'],'report')
ck('unified/original',1.17067,u['retained_chunk_seconds']/f['retained_chunk_seconds'],'')
for n,c,vals in (('run1',fc,(227.809,227.915,229.334)),('run2',uc,(266.717,266.834,268.298))):
    x=[1e6*k['seconds']/k['amount'] for k in c if k['amount']==8192]
    ck(n+' full chunks',762,len(x),'chunks');ck(n+' min',vals[0],min(x),'chunks',.0005);ck(n+' median',vals[1],st.median(x),'chunks',.0005);ck(n+' max',vals[2],max(x),'chunks',.0005)
ck('RV instructions',3885544,L('experiments/audit/results/full-doom.json')['instructions'],'rvi.py')
t=L('experiments/dmc-feedback-fusion/evidence/timing-summary.json')
ck('render base us',479.705,t['frame2_render']['baseline']['total_us_per_token'],'timing-summary',.0005)
ck('render rom us',251.663,t['frame2_render']['rom']['total_us_per_token'],'timing-summary',.0005)
ck('render fused us',228.155,t['frame2_render']['fused']['total_us_per_token'],'timing-summary',.0005)
ck('render fused tok/s',4382.98,t['frame2_render']['fused']['tokens_per_second_median'],'timing-summary',.0005)
ck('boot fused tok/s',4380.38,t['boot']['fused']['tokens_per_second_median'],'timing-summary',.0005)
ck('~2.10x window',2.10,t['frame2_render']['cumulative_speedup'],'timing-summary',.005,'derived')
p=L('experiments/dmc-cuda-profile/evidence/profile-summary.json')['windows']['frame2_render']
ck('ROM heads us',260.136,p['device_groups_us_per_token']['data_rom_heads'],'profile-summary',.0005)
ck('ROM %',51.34,p['device_groups_percent_of_stream_span']['data_rom_heads'],'profile-summary',.0005)
ck('RAM heads us',41.289,p['device_groups_us_per_token']['ram_heads'],'profile-summary',.0005)
ck('stream span us',506.7,p['device_stream_span_us_per_token'],'profile-summary',.0005)
c=L('experiments/one-instr-design/cost-model.json')
ck('later-frame tokens',4501708.33,c['old_tokens_per_frame'],'cost-model')
ck('orig s/frame',2159,c['old_tokens_per_frame']*479.705e-6,'',.002,'projection')
ck('fused s/frame',1027,c['old_tokens_per_frame']*228.155e-6,'',.002,'projection')
ck('~52,000x',52000,c['old_tokens_per_frame']*479.705e-6*24,'',.01,'projection')
ck('phases sum',46858641,45865842+443715+237404+309655+2025,'CONTINUING.md')
ck('walk tokens',46858641,L('experiments/one-instr-design/counts-walk.json')['tokens'],'counts-walk')
ck('1.624x',1.624,c['ratio'],'cost-model',.001,'estimate')
s=L('experiments/tiny-llm-ndot-fused/evidence/story-validation.json');r=L('experiments/audit/results/full-stories.json')
ck('story tokens',53296563,r['gpu_tokens'],'raw trace')
ck('story min/pos',6.2394,s['compute_seconds']/60/38,'validation',.0005)
ck('story tok/s',3746.445,53296563/s['compute_seconds'],'',.0001)
q=L('experiments/tiny-qwen35/evidence/neural-validation.json');rq=L('experiments/audit/results/full-qwen35.json')
ck('qwen tokens',35836005,rq['gpu_tokens'],'raw trace');ck('qwen instr',25121038,rq['instructions'],'rvi.py');ck('qwen ndots',2257920,rq['ndots'],'rvi.py')
ck('qwen min/pos',27.82,q['compute_seconds']/60/6,'validation');ck('qwen min/new',33.38,q['compute_seconds']/60/5,'validation')
m=L('experiments/showcase/evidence/summary.json')['mnist']
ck('MNIST time ratio',4.152,m['observed_gpu_speedup'],'summary',.0005);ck('MNIST token ratio',4.1521,785128/189092,'rvi.py',.0001)
ck('MNIST instr ratio',4.6202,560569/121329,'rvi.py',.0001);
lab=Path('.cache/showcase/mnist-model/test-labels.bin').read_bytes();pred=Path('.cache/showcase/mnist-model/test-predictions.bin').read_bytes()
ck('MNIST int acc %',96.05,100*sum(a==b for a,b in zip(lab,pred))/10000,'raw predictions',.0001)
b=L('experiments/tiny-llm-ndot/evidence/baseline.json')
ck('orig unfused proj min/pos',25.90,b['original_unfused_projected_minutes_per_position'],'baseline',.001,'projection')
ck('fused/unfused boot',1.5610,419.083/268.472,'README table',.0005,'derived')
ck('1.86x (unlike backends)',1.86,14.06/7.5641,'README numbers',.005,'derived: fused original vs unfused unified')
ck('3.4x (projected vs measured)',3.4,b['original_unfused_projected_minutes_per_position']/7.5641,'',.015,'derived: projection/measurement')
ck('~5.3x (projection/projection)',5.3,b['original_unfused_projected_minutes_per_position']/4.853,'',.012,'derived: both projections; 38-pos measured ratio would be 4.15')
print(sum(x['ok'] for x in R),'of',len(R),'ok')
for x in R:
    if not x['ok']: print('MISMATCH',x)
json.dump(R,open('experiments/audit/results/claims.json','w'),indent=1)
print('stock speed ratio',3746.33/16.6, 35836005/16.6/86400,'days for tiny-qwen in stock;',53296563/16.6/86400,'days stories',6342933/16.6/3600,'hours doom')
print('stock coverage %',16384/35836005*100,32768/53296563*100,4096/6342933*100)
print('python interp vs GPU',1693.1/L('experiments/audit/results/full-doom.json')['seconds'],s['compute_seconds']/r['seconds'])
