import os,sys,time,json,threading,hashlib,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.upstream/qwenputer/experimental/riscv'))
import numpy as np
import torch,triton
from PIL import Image
from chipformer.rv import core,ref,elf,spec,fastexec

def decode_frames(data):
 frames=[];offset=0
 while offset<len(data):
  assert data[offset:offset+2]==b'QF'
  frame_id,count,hits,misses=data[offset+2:offset+6];offset+=6;pixels=[0]*256
  for _ in range(count):
   index,value=data[offset:offset+2];pixels[index]=value;offset+=2
  frames.append({'id':frame_id,'bytes':pixels,'hits':hits,'misses':misses,'sha256':hashlib.sha256(bytes(pixels)).hexdigest()})
 return frames

def main():
 timer=threading.Timer(55,lambda:os._exit(124));timer.daemon=True;timer.start()
 out=ROOT/'artifacts/compact-game-cuda';out.mkdir(exist_ok=True)
 report={'mode':'upstream compact circuit Triton Executor2, fp32 CUDA graph; not stock HF','is_doom':False,'wall_limit_seconds':55}
 try:
  torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;assert spec.RAM_BASE==0 and spec.RAM_WORDS==8192 and spec.DROM_BASE==0x20000 and spec.DROM_WORDS==32768
  started=time.perf_counter();c=core.build();firmware=ROOT/'artifacts/weight-paddle.elf';program=elf.program(firmware);reference=ref.Machine.from_program(program);wanted=reference.run()[1:];assert reference.halted and reference.fault is None and len(wanted)==54570
  core.snapshot(c,ref.Machine.from_program(program));source=Path(fastexec.__file__).read_text();patched=source.replace('nf, nb = len(C.OUT_NAMES), C.NPAY','output_names = list(S.FAMILIES) + S.CONTROL_NAMES\n        nf, nb = len(output_names), C.NPAY').replace('enumerate(C.OUT_NAMES)','enumerate(output_names)');assert source!=patched
  (out/'fastexec-standalone.py').write_text(patched);module=types.ModuleType('fastexec_standalone_full_game');exec(compile(patched,str(out/'fastexec-standalone.py'),'exec'),module.__dict__)
  fr=module.FreeRun(c,1,'cuda',compile_=False,v2=True);fr.load(0,spec.tok('GO'));report['setup_seconds']=time.perf_counter()-started
  with torch.no_grad():
   t=time.perf_counter();assert fr.run(1)[0].tolist()==wanted[:1];report['capture_warmup_seconds']=time.perf_counter()-t
   fr.load(0,spec.tok('GO'));torch.cuda.synchronize();t=time.perf_counter();generated=fr.run(len(wanted))[0].tolist();elapsed=time.perf_counter()-t
  assert generated==wanted
  output=bytes(spec.family(t)[1] for t in generated if spec.family(t)[0]=='BYTE');assert output==bytes(reference.out)
  frames=decode_frames(output);assert len(frames)==12 and len({f['sha256'] for f in frames})==12
  trace=np.asarray(generated,dtype='<u4').tobytes();(out/'trace.u32').write_bytes(trace);(out/'actual-output.bin').write_bytes(output);(out/'frames.json').write_text(json.dumps(frames));(out/'weight-paddle.elf').write_bytes(firmware.read_bytes())
  images=[]
  for f in frames:
   image=Image.new('RGB',(64,32),'#101b22');image.putdata([(166,247,207) if f['bytes'][y*8+x//8]&(128>>(x&7)) else (16,27,34) for y in range(32) for x in range(64)]);images.append(image.resize((1024,512),Image.Resampling.NEAREST))
  images[-1].save(out/'frame.png');images[0].save(out/'replay.gif',save_all=True,append_images=images[1:],duration=250,loop=0)
  report.update({'upstream_revision':'ca13310e2f9f788d1dd3feeea8213c1b0c23913c','firmware_sha256':hashlib.sha256(firmware.read_bytes()).hexdigest(),'generated_tokens':len(generated),'mismatches':0,'frames':len(frames),'distinct_frames':12,'elapsed_seconds_one_run':elapsed,'timed_work':'FreeRun CUDA graph replay, self-fed token embeddings, trace copies and final CPU transfer; excludes setup, graph capture, reference execution, token comparison and rendering','reference_tokens_used_as_inputs':False,'trace_sha256':hashlib.sha256(trace).hexdigest(),'torch':torch.__version__,'triton':triton.__version__,'gpu':torch.cuda.get_device_name(),'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'graph':{'chip_layers':c.n_layers(),'dims':len(c.dims)},'inputs':'same deterministic game_keys compiled into the saved ELF; prerecorded replay','source_adaptation':'Only merged-chat Q family omitted from standalone FreeRun embedding table; circuit kernels unchanged','replay_display_fps':4,'replay_is_live_execution':False,'total_seconds':time.perf_counter()-started})
 except Exception as e:
  import traceback
  report['error']=repr(e);report['traceback']=traceback.format_exc();print(report['traceback'],flush=True)
 finally:
  (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True);timer.cancel()
 if 'error' in report:sys.exit(1)

if __name__=='__main__':main()
