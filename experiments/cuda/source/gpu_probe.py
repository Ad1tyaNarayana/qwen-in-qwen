import json,sys,torch,platform
x=torch.arange(16,device='cuda',dtype=torch.bfloat16).reshape(4,4)
y=x@x
assert x.is_cuda and y.is_cuda
assert y.cpu().float().tolist() == [[56.,62.,68.,74.],[152.,174.,196.,218.],[248.,286.,324.,362.],[344.,398.,452.,506.]]
torch.cuda.synchronize()
print(json.dumps({'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'cuda_available':torch.cuda.is_available(),'device':torch.cuda.get_device_name(),'capability':torch.cuda.get_device_capability(),'compiled_arches':torch.cuda.get_arch_list(),'input_is_cuda':x.is_cuda,'result_is_cuda':y.is_cuda,'bf16_matmul_exact':True},indent=2),flush=True)
