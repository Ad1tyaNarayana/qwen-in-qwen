"""Run retained checks with output redirected exclusively into blind-review."""
import hashlib,importlib.util,json,os,subprocess,sys,tempfile,traceback
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
(OUT/'tmp').mkdir(exist_ok=True);tempfile.tempdir=str(OUT/'tmp')
def load(name,relative):
 s=importlib.util.spec_from_file_location(name,ROOT/relative);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
results={}
def attempt(name,f):
 try:results[name]={'status':'pass','result':f()}
 except Exception as e:results[name]={'status':'fail','error':repr(e),'traceback':traceback.format_exc()}
 (OUT/'local-checks.json').write_text(json.dumps(results,indent=2));print(name,results[name]['status'],flush=True)
def restore():
 path=ROOT/'experiments/checkpoint-integrity/test_restore.py'
 source=path.read_text().replace("out=ROOT/'experiments/checkpoint-integrity/evidence/cpu-restore.json'",f'out=Path({str(OUT/"cpu-restore.json")!r})')
 ns={'__name__':'blind_restore','__file__':str(path)};exec(compile(source,str(path),'exec'),ns);ns['main']();return json.loads((OUT/'cpu-restore.json').read_text())
attempt('checkpoint_cpu_real_entrypoints',restore)
def measure():
 m=load('measurement_verify','experiments/checkpoint-integrity/verify.py');m.EVIDENCE=OUT/'measurement';m.EVIDENCE.mkdir(exist_ok=True)
 (m.EVIDENCE/'scalar-reference-repair.json').write_bytes((ROOT/'experiments/checkpoint-integrity/evidence/scalar-reference-repair.json').read_bytes());m.main()
 return json.loads((m.EVIDENCE/'controlled-measurement.json').read_text())
attempt('controlled_measurement_full_raw_verifier',measure)
for name in ('ff-full','unified-full'):
 attempt('doom_raw_'+name,lambda name=name:load('doom_verify','experiments/doom-showcase/verify_runs.py').verify(name))
def historical():
 paths=['experiments/tiny-qwen35/evidence/gpu.json','experiments/tiny-llm-ndot-fused/evidence/story-run.json']+[f'experiments/showcase/{d}/evidence/gpu-report.json' for d in ['life','sha256','tictactoe','mnist','mnist-scalar']]+['experiments/doom-showcase/evidence/ff-full-report.json','experiments/doom-showcase/evidence/unified-full-report.json']
 rows=[]
 for name in paths:
  report=json.loads((ROOT/name).read_text());scripts=report.get('scripts',{})
  for path,digest in scripts.items():
   if not isinstance(digest,str):continue
   rel=path.replace('\\','/');current=ROOT/rel
   if not current.is_file():continue
   b=subprocess.run(['git','show','e7bd077:'+rel],cwd=ROOT,capture_output=True)
   rows.append(dict(report=name,file=rel,bound=digest,current_matches=hashlib.sha256(current.read_bytes()).hexdigest()==digest,historical_matches=b.returncode==0 and hashlib.sha256(b.stdout).hexdigest()==digest))
 return rows
attempt('historical_source_bindings_at_e7bd077',historical)
