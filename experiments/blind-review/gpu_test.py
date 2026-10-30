"""Independent guest: signed dot followed by reflected CRC recurrence; no reference execution."""
import argparse, hashlib, importlib.util, json, os, struct, sys, time, traceback
from pathlib import Path
sys.dont_write_bytecode=True
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--upstream',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--mode',choices=['fused','stock'],required=True);a=p.parse_args()
a.out.mkdir(parents=True,exist_ok=False)
os.environ['TRITON_CACHE_DIR']=str(a.out/'triton');os.environ['TORCHINDUCTOR_CACHE_DIR']=str(a.out/'inductor')
r={'status':'running','mode':a.mode,'reference_execution_disabled':True}
def write(): (a.out/'report.json').write_text(json.dumps(r,indent=2))
write()
try:
 import numpy as np, torch
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
 cap=(15 if a.mode=='stock' else 8)*2**30;torch.cuda.set_per_process_memory_fraction(cap/torch.cuda.get_device_properties(0).total_memory)
 r.update(allocator_cap_bytes=cap,minimum_free_bytes=torch.cuda.mem_get_info()[0],torch=torch.__version__,gpu=torch.cuda.get_device_name())
 def guard(required=0):
  torch.cuda.empty_cache();free,_=torch.cuda.mem_get_info();r['minimum_free_bytes']=min(free,r['minimum_free_bytes']);assert free-required>=512*2**20,(free,required)
 guard()
 s=importlib.util.spec_from_file_location('prep',a.repo/'experiments/memory-map/prepare.py');prep=importlib.util.module_from_spec(s);s.loader.exec_module(prep);prep.prepare(a.upstream,a.out/'scratch',64,4096)
 sys.path[:0]=[str(a.out/'scratch'),str(a.repo/'experiments/neural-accelerator'),str(a.repo/'experiments/tiny-llm-ndot-fused'),str(a.repo/'experiments/tiny-llm-ndot')]
 from chipformer.rv import asm,spec,ref
 import reference
 from constructor import build,snapshot
 # Hand-encode RV32; neither project assembler nor reference semantics produce expected results.
 words=[]
 def I(rd,rs,imm,f=0): words.append((imm&4095)<<20|rs<<15|f<<12|rd<<7|0x13)
 def R(rd,a,b,f,f7=0): words.append(f7<<25|b<<20|a<<15|f<<12|rd<<7|0x33)
 def lui(rd,v): words.append((v&0xfffff000)|rd<<7|0x37)
 lui(1,0x01000000);lui(2,0x02000000);I(3,0,-17)
 words.append(0x20<<25|1<<20|2<<15|3<<7|0x0b) # NDOT4 x3, x2, x1
 lui(4,0xedb88000);I(4,4,0x320);I(5,0,32)
 loop=len(words)
 I(6,3,1,7);R(6,0,6,0,0x20);R(6,6,4,7);I(3,3,1,5);R(3,3,6,4);I(5,5,-1)
 off=(loop-len(words))*4;o=off&8191
 words.append((o>>12&1)<<31|(o>>5&63)<<25|5<<15|1<<12|(o>>1&15)<<8|(o>>11&1)<<7|0x63)
 words.append(3<<20|1<<15|2<<12|16<<7|0x23) # sw x3,16(x1)
 I(17,0,1)
 for shift in [0,8,16,24]: I(10,3,shift,5);I(10,10,255,7);words.append(0x73)
 I(17,0,93);I(10,0,0);words.append(0x73)
 weights=[-128,127,-7,19];values=[-32768,32767,-12345,23456]
 dot=(-17+sum(w*v for w,v in zip(weights,values)))&0xffffffff
 crc=dot
 for _ in range(32): crc=(crc>>1)^(0xedb88320 if crc&1 else 0)
 expected=crc.to_bytes(4,'little')
 data=struct.pack('<4h',*values)+bytes(12)
 program=asm.Program(words,data,{'_start':spec.ROM_BASE},data_base=spec.RAM_BASE,rodata=struct.pack('<4b',*weights))
 boot=reference.Machine.from_program(program);boot.maximum=4
 def forbidden(*args,**kwargs): raise AssertionError('reference execution prohibited')
 ref.Machine.record=forbidden;reference.Machine.record=forbidden;ref.Machine.run=forbidden;reference.Machine.run=forbidden
 circuit=build(4);r.update(constructor=circuit.source_manifest,expected_dot=dot,expected_crc=crc,expected_output_hex=expected.hex(),program_words=words)
 write();start=time.perf_counter()
 with torch.inference_mode():
  if a.mode=='fused':
   from compact import freerun_class
   from fusion import install
   snapshot(circuit,boot);fr=freerun_class(a.out)(circuit,1,'cuda',compile_=False,v2=True);fused=install(fr)
   def reset():
    fr.load(0,spec.tok('GO'));fr.ex.reset();fr.ex.y.zero_();fr.tok.zero_()
    for y in fused['scratch_buffers']:y.zero_()
   reset();fr.capture();reset();tokens=[]
   for i in range(4096):
    token=int(fr.run(1)[0][0]);tokens.append(token)
    if token==spec.tok('HALT'):break
   else:raise AssertionError('No HALT within budget')
   sys.path.insert(0,str(a.repo/'experiments/dmc-cuda/source/helpers'));from state import read_ram
   ram=read_ram(fr.ex,spec.RAM_BASE,65536);(a.out/'ram.bin').write_bytes(ram)
   assert ram[16:20]==expected,(ram[16:20].hex(),expected.hex())
   from worker import read_registers
   regs=read_registers(fr,circuit);assert regs['regs.rs1'][3]==crc
   r['registers']=regs;r['ram_result_hex']=ram[16:20].hex()
  else:
   from stock import build_model,boot_cache
   from transformers import StoppingCriteria,StoppingCriteriaList
   model,meta=build_model(circuit,'cuda',memory_guard=guard)
   h=hashlib.sha256()
   for name,t in sorted(model.state_dict().items()):
    for block in t.detach().view(-1).split(4*1024*1024):h.update(block.cpu().contiguous().view(torch.uint16).numpy().astype('<u2',copy=False).tobytes())
   r['actual_loaded_weight_sha256']=h.hexdigest();assert h.hexdigest()=='0e515bbb3109997a130c903deb40dce3447f8991e15b6637e0071c71f8b6e07d'
   class Guard(StoppingCriteria):
    def __call__(self,*args,**kwargs):guard();return False
   cache=boot_cache(model,circuit,boot)
   tokens=model.generate(input_ids=torch.tensor([[spec.VOCAB,spec.tok('GO')]],device='cuda'),past_key_values=cache,max_new_tokens=4096,do_sample=False,eos_token_id=spec.tok('HALT'),pad_token_id=spec.VOCAB,stopping_criteria=StoppingCriteriaList([Guard()]))[0,2:].tolist()
   assert tokens[-1]==spec.tok('HALT')
  raw=np.asarray(tokens,dtype='<u4').tobytes();(a.out/'actual.tokens.u32').write_bytes(raw)
  output=bytes(v for f,v in map(spec.family,tokens) if f=='BYTE' and v<256)
  assert output==expected,(output.hex(),expected.hex())
  assert not any(spec.family(t)[0]=='FLT' for t in tokens)
  r.update(status='pass',tokens=len(tokens),output_hex=output.hex(),trace_sha256=hashlib.sha256(raw).hexdigest(),seconds_including_executor_construction=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
except Exception as e:r.update(status='fail',error=repr(e),traceback=traceback.format_exc());traceback.print_exc()
finally:write();print(json.dumps(r,indent=2),flush=True)
sys.exit(r['status']!='pass')
