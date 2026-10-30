"""Negative controls on retained verifiers; historical hashes resolved at documented commit."""
import contextlib,hashlib,importlib.util,io,json,subprocess,sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent

def load(name,path):
 s=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def historical_sha(p):
 p=Path(p)
 if p.suffix=='.py' and p.is_relative_to(ROOT/'experiments') and not p.is_relative_to(OUT):
  b=subprocess.check_output(['git','show','e7bd077:'+p.relative_to(ROOT).as_posix()],cwd=ROOT)
  return hashlib.sha256(b).hexdigest()
 return hashlib.sha256(p.read_bytes()).hexdigest()

def linked_tree(source,dest,omit):
 dest.mkdir(parents=True,exist_ok=True)
 for p in source.iterdir():
  q=dest/p.name
  if p.name in omit:continue
  if q.exists() or q.is_symlink():continue
  q.symlink_to(p.resolve(),target_is_directory=p.is_dir())
results={}
# Empty journal; all other retained bytes unchanged, including the actual full trace.
source=ROOT/'.cache/tiny-llm/ndot-story38-20261030-actual';dest=OUT/'controls/story-empty-journal'
linked_tree(source,dest,{'main'});linked_tree(source/'main',dest/'main',{'chunks.jsonl'})
(dest/'main/chunks.jsonl').write_text('')
m=load('story_check','experiments/tiny-llm-ndot-fused/verify_story.py');m.sha=historical_sha
for label,actual in [('story_unmodified',source),('story_empty_journal',dest)]:
 sys.argv=['verify_story.py','--inputs',str(ROOT/'.cache/tiny-llm/ndot-story38-20261030'),'--actual',str(actual),'--out',str(OUT/(label+'.json'))]
 try:
  with contextlib.redirect_stdout(io.StringIO()):m.main()
  results[label]={'accepted':True}
 except Exception as e:results[label]={'accepted':False,'error':repr(e)}
# Omit unified Doom RAM and register export: verifier still returns successfully.
m=load('doom_check','experiments/doom-showcase/verify_runs.py')
dest=OUT/'controls/doom'
linked_tree(ROOT/'.cache/doom-showcase/export-unified-full',dest/'export-unified-full',{'final-gpu-ram.bin','actual-registers.json'})
m.CACHE=dest
try:
 r=m.verify('unified-full');results['doom_missing_final_state']={'accepted':True,'result':r}
except Exception as e:results['doom_missing_final_state']={'accepted':False,'error':repr(e)}
(OUT/'verifier-controls.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
