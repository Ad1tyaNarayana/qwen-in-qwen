"""Independent raw-artifact arithmetic, byte checks and article number ledger."""
import hashlib,json,re,statistics,sys
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
R=Path(__file__).resolve().parents[2];O=Path(__file__).resolve().parent
j=lambda p:json.loads((R/p).read_text())
rows=[];raw_results={}
def check(claim,got,want,source,places=None):
 ok=(round(got,places)==want) if places is not None else got==want
 rows.append(dict(claim=claim,actual=got,article_value=want,source=source,pass_=ok));assert ok,(claim,got,want)
# All full LLM outputs checked directly; token BYTE range from pinned specification.
sys.path.insert(0,str(R/'.cache/tiny-llm/ndot-story38-20261030/reference-package'))
from chipformer.rv import spec
for name,actual,fixture,n,positions,seconds,minutes in [
 ('scalar','.cache/checkpoint-integrity/actual-20261030a/scalar','.cache/tiny-llm/neural-stories',16213774,5,4361.42,14.5381),
 ('ndot','.cache/checkpoint-integrity/actual-20261030a/ndot','.cache/tiny-llm/ndot5-20261030',5423298,5,1446.63,4.8221),
 ('story','.cache/tiny-llm/ndot-story38-20261030-actual/main','.cache/tiny-llm/ndot-story38-20261030',53296563,38,14225.90,6.2394),
 ('qwen','.cache/tiny-qwen35/actual-covariance-20261030/main','.cache/tiny-qwen35/trained-program',35836005,6,10014.17,27.82)]:
 a=np.memmap(R/actual/'actual.tokens.u32',mode='r',dtype='<u4');b=np.memmap(R/fixture/'reference.tokens.u32',mode='r',dtype='<u4');assert len(a)==len(b)
 errors=sum(int(np.count_nonzero(a[i:i+1048576]!=b[i:i+1048576])) for i in range(0,len(a),1048576))
 check(name+' trace tokens',len(a),n,actual+'/actual.tokens.u32');check(name+' mismatches',errors,0,fixture+'/reference.tokens.u32')
 ram=(R/actual/'actual-ram.bin').read_bytes();assert ram==(R/fixture/'reference-ram.bin').read_bytes();check(name+' final RAM bytes',len(ram),65536,actual+'/actual-ram.bin')
 regs=j(actual+'/actual-registers.json');ref=j(fixture+'/reference.json');assert set(regs)=={'regs.rs1','regs.rs2','regs.rd'} and all(v==ref['final_registers'] for v in regs.values());check(name+' register copies',len(regs),3,actual+'/actual-registers.json')
 base,size=spec.FAMILIES['BYTE'];out=bytearray()
 for i in range(0,len(a),1048576):
  c=a[i:i+1048576];out.extend((c[(c>=base)&(c<base+256)]-base).astype('u1').tobytes())
 assert out==(R/actual/'actual-output.bin').read_bytes()==(R/fixture/'reference-output.bin').read_bytes()
 records=re.findall(rb'^@(\d+) (\d+) (\d+) (\d+) (\d+)\n',out,re.M);check(name+' positions',len(records),positions,actual+'/actual.tokens.u32')
 chunks=[json.loads(s) for s in (R/actual/'chunks.jsonl').read_text().splitlines()];last=0;sec=0
 for c in chunks:
  end=c['end_offset'];assert end-last==c['amount']>0;assert c['compute_seconds']>=sec
  assert hashlib.sha256(a[last:end].tobytes()).hexdigest()==c['sha256'];last=end;sec=c['compute_seconds']
 assert last==len(a) and sec==j(actual+'/report.json')['compute_seconds']
 check(name+' generation seconds',sec,seconds,actual+'/chunks.jsonl',2)
 check(name+' min/position',sec/positions/60,minutes,actual+'/chunks.jsonl',2 if name=='qwen' else 4)
 raw_results[name]=dict(tokens=len(a),seconds=sec,prediction_ids=[int(x[2]) for x in records],text=b''.join(re.findall(rb'^>(.*)\n',out,re.M)).decode(),chunks=len(chunks),all_chunk_hashes_exact=True)
check('scalar trace tokens/s',16213774/raw_results['scalar']['seconds'],3717.54,'scalar raw count/chunk timer',2)
check('NDOT trace tokens/s',5423298/raw_results['ndot']['seconds'],3748.93,'NDOT raw count/chunk timer',2)
check('scalar/NDOT ratio',raw_results['scalar']['seconds']/raw_results['ndot']['seconds'],3.0149,'both complete raw chunk journals',4)
check('Qwen min/new token including prefill',raw_results['qwen']['seconds']/5/60,33.38,'qwen raw chunk journal',2)
for name,gent,wall,rate,us in [('ff-full',1446.28,1469.41,4385.70,228.01),('unified-full',1693.11,1715.49,3746.33,266.93)]:
 d='.cache/doom-showcase/export-'+name;report=j(d+'/report.json');a=np.memmap(R/d/'actual.tokens.u32',mode='r',dtype='<u4');chunks=[json.loads(s) for s in (R/d/'chunks.jsonl').read_text().splitlines()]
 sec=sum(c['seconds'] for c in chunks);full=[c['seconds']*1e6/c['amount'] for c in chunks if c['amount']==8192]
 check(name+' tokens',len(a),6342933,d+'/actual.tokens.u32');check(name+' generation seconds',sec,gent,d+'/chunks.jsonl',2);check(name+' tokens/s',len(a)/sec,rate,d+'/chunks.jsonl',2);check(name+' us/token',sec*1e6/len(a),us,d+'/chunks.jsonl',2);check(name+' wall seconds',report['uninterrupted_end_to_end_seconds'],wall,d+'/report.json',2);check(name+' full chunks',len(full),762,d+'/chunks.jsonl')
 check(name+' frame pixels',(R/d/'frame-0001.rgb565').stat().st_size//2,76800,d+'/frame-0001.rgb565')
 for label,got,want in zip(['min','median','max'],[min(full),statistics.median(full),max(full)], [227.809,227.915,229.334] if name=='ff-full' else [266.717,266.834,268.298]):check(name+' '+label+' us/token',got,want,d+'/chunks.jsonl',3)
# Quality agreement from individual records, not summary fields.
quality=j('experiments/tiny-qwen35/evidence/covariance-quality.json');matches=total=0
for c in quality['cases']:
 ids=[r[2] for r in c['integer_records']][len(c['prompt_tokens'])-1:]
 floats=c['float_records']
 # Inspect exact record format rather than assuming.
 if floats and isinstance(floats[0],dict):fids=[r['prediction'] for r in floats]
 else:fids=[r[2] for r in floats]
 # float_records start with the first post-prompt prediction.
 ids=ids[:len(fids)];assert len(ids)==len(fids)==c['generated_predictions']
 m=sum(x==y for x,y in zip(ids,fids));assert m==c['free_running_matches'];matches+=m;total+=len(ids)
check('quality matched predictions',matches,291,'covariance-quality.json per-prompt integer_records vs float_continuation');check('quality denominator',total,327,'same');check('quality prompts',len(quality['cases']),6,'same')
stock=np.fromfile(R/'.cache/tiny-qwen35/actual-covariance-20261030/stock/actual.tokens.u32',dtype='<u4')
ref=np.memmap(R/'.cache/tiny-qwen35/trained-program/reference.tokens.u32',mode='r',dtype='<u4')
check('Qwen stock prefix tokens',len(stock),16384,'.cache/tiny-qwen35/actual-covariance-20261030/stock/actual.tokens.u32')
check('Qwen stock prefix mismatches',int(np.count_nonzero(stock!=ref[:len(stock)])),0,'same, against reference trace')
(O/'raw-checks.json').write_text(json.dumps(raw_results,indent=2));(O/'number-checks.json').write_text(json.dumps(rows,indent=2))
lines=['# Article number checks','','Computed from retained raw files where present. Wall time is a retained clock report, not independently measurable after the run. Quality records were recomputed, not generation rerun.','','| Claim | Recomputed value | Article value | Evidence |','|---|---:|---:|---|']
for row in rows:lines.append(f"| {row['claim']} | {row['actual']} | {row['article_value']} | `{row['source']}` |")
(O/'NUMBERS.md').write_text('\n'.join(lines)+'\n');print('All',len(rows),'number checks pass');print(json.dumps(raw_results,indent=2))
