import sys,types,time,json,threading,os,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.upstream/qwenputer/experimental/riscv'))
import torch
from chipformer.rv import core,ref,elf,spec

def main():
 timer=threading.Timer(60,lambda:os._exit(124));timer.daemon=True;timer.start()
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 out=ROOT/'artifacts/acceleration-probe'
 module=types.ModuleType('fastexec_standalone_probe');source=out/'fastexec-standalone.py';exec(compile(source.read_text(),str(source),'exec'),module.__dict__)
 c=core.build();program=elf.program(ROOT/'artifacts/weight-paddle.elf');wanted=ref.Machine.from_program(program).run()[1:5416];core.snapshot(c,ref.Machine.from_program(program))
 engines={}
 with torch.no_grad():
  for label,v2 in [('triton',True),('torch-sparse-cudagraph',False)]:
   fr=module.FreeRun(c,1,'cuda',compile_=False,sparse=True,v2=v2);fr.load(0,spec.tok('GO'));assert fr.run(64)[0].tolist()==wanted[:64];engines[label]=fr
  samples={label:[] for label in engines}
  for repeat in range(5):
   for label,fr in engines.items():
    fr.load(0,spec.tok('GO'));torch.cuda.synchronize();t=time.perf_counter();got=fr.run(5415)[0].tolist();elapsed=time.perf_counter()-t;assert got==wanted;samples[label].append(elapsed);print(label,repeat,elapsed,flush=True)
  report={'tokens_each_run':5415,'mismatches':0,'repeats_each_path':5,'schedule':'alternating Triton then uncompiled torch sparse, each reset to identical snapshot','samples_seconds':samples,'median_seconds':{label:statistics.median(xs) for label,xs in samples.items()},'range_seconds':{label:[min(xs),max(xs)] for label,xs in samples.items()},'timed_work':'CUDA graph replays with device token decoding, self-fed embeddings, trace output copies and final CPU transfer; excludes setup and graph capture','bottleneck_profile':{}}
  fr=engines['triton'];fr.load(0,spec.tok('GO'))
  try:
   from torch.profiler import profile,ProfilerActivity
   with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as prof:fr.run(64)
   events=prof.key_averages()
   report['bottleneck_profile']={'activities':'CPU and CUDA requested; separate64-token run excluded from measurements','top_events':[{'key':e.key,'count':e.count,'cpu_time_total_us':e.cpu_time_total,'device_time_total_us':e.device_time_total} for e in sorted(events,key=lambda e:e.device_time_total,reverse=True)[:12]]}
  except Exception as e:report['bottleneck_profile']={'unavailable':repr(e)}
 (out/'comparison.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True);timer.cancel()

if __name__=='__main__':main()
