import torch,triton
import triton.language as tl

@triton.jit
def add_kernel(x,y,z,N:tl.constexpr,BLOCK:tl.constexpr):
 i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
 mask=i<N
 tl.store(z+i,tl.load(x+i,mask=mask,other=0)+tl.load(y+i,mask=mask,other=0),mask=mask)

def check():
 x=torch.arange(1024,device='cuda',dtype=torch.float32);y=x*2;z=torch.empty_like(x)
 add_kernel[(triton.cdiv(x.numel(),128),)](x,y,z,x.numel(),BLOCK=128)
 torch.cuda.synchronize();assert z.is_cuda and torch.equal(z,x*3)
 return {'generated_kernel_cuda_tensor':z.is_cuda,'float32_elements_exact':z.numel()}
