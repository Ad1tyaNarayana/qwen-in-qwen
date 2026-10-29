import argparse,json,os,sys,time,threading,traceback,types,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.upstream/qwenputer/experimental/riscv'))
import numpy as np
import torch
from chipformer.rv import core,elf,ref,spec,fastexec

def main():
 p=argparse.ArgumentParser();p.add_argument('--v2',action='store_true');args=p.parse_args()
 out=ROOT/'artifacts/acceleration-probe';out.mkdir(parents=True,exist_ok=True)
 label='triton' if args.v2 else 'torch-sparse-cudagraph'
 timer=threading.Timer(240,lambda:os._exit(124));timer.daemon=True;timer.start()
 torch.set_num_threads(4)
 torch.backends.cuda.matmul.allow_tf32=False
 report={'mode':label,'torch':torch.__version__,'device':torch.cuda.get_device_name(),'gpu':True,'dtype':'float32','allow_tf32':False,'reference_tokens_used_as_inputs':False}
 try:
  if args.v2:
   import triton,triton.language as tl
   report['triton']=triton.__version__
   import triton_probe_kernel
   report['arithmetic_probe']=triton_probe_kernel.check()
  t=time.perf_counter();c=core.build();program=elf.program(ROOT/'artifacts/weight-paddle.elf')
  expected=ref.Machine.from_program(program).run()[1:]
  core.snapshot(c,ref.Machine.from_program(program))
  src=Path(fastexec.__file__).read_text()
  patched=src.replace('nf, nb = len(C.OUT_NAMES), C.NPAY', 'output_names = list(S.FAMILIES) + S.CONTROL_NAMES\n        nf, nb = len(output_names), C.NPAY').replace('enumerate(C.OUT_NAMES)', 'enumerate(output_names)')
  assert patched != src
  (out/'fastexec-standalone.py').write_text(patched)
  module=types.ModuleType('fastexec_standalone_probe');exec(compile(patched,str(out/'fastexec-standalone.py'),'exec'),module.__dict__)
  report['patch']='FreeRun embedding table omits merged-chat-only output family Q, preserving every standalone family id and row. Circuit arithmetic and kernels unchanged.'
  report['source_sha256']=hashlib.sha256(src.encode()).hexdigest();report['patched_source_sha256']=hashlib.sha256(patched.encode()).hexdigest()
  print('Building',label,flush=True)
  fr=module.FreeRun(c,1,'cuda',compile_=False,sparse=True,v2=args.v2)
  fr.load(0,spec.tok('GO'))
  report['setup_seconds']=time.perf_counter()-t
  print('Setup seconds',report['setup_seconds'],flush=True)
  with torch.no_grad():
   got=fr.run(64)[0].tolist();assert got==expected[:64],(got,expected[:64])
   fr.load(0,spec.tok('GO'));torch.cuda.synchronize();t=time.perf_counter()
   got=fr.run(5415)[0].tolist();elapsed=time.perf_counter()-t
  assert got==expected[:5415],next((i,a,b) for i,(a,b) in enumerate(zip(got,expected)) if a!=b)
  np.asarray(got,dtype='<u4').tofile(out/(label+'.trace.u32'))
  report.update({'generated_tokens':len(got),'mismatches':0,'seconds_one_run':elapsed,'timed_work':'FreeRun CUDA graph replay, next-token decoding/embedding, output copies and final CPU token transfer; excludes setup/capture warm-up/reference run','cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'complete_first_frame':True})
 except Exception as e:
  report['error']=repr(e);report['traceback']=traceback.format_exc();print(report['traceback'],flush=True)
 finally:
  (out/(label+'.json')).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True);timer.cancel()
 if 'error' in report:sys.exit(1)

if __name__=='__main__':main()
