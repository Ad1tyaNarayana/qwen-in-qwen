"""Measure real C attention operators against the installed HF implementation."""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch
from transformers import Qwen3_5ForCausalLM, DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import l2norm

HERE = Path(__file__).resolve().parent


def check(checkpoint, image):
    torch.set_num_threads(1)
    meta = json.loads((image/'pack.json').read_text())
    libpath = image/'operators.dylib'
    command = ['clang','-O1','-fwrapv','-shared','-fPIC',str(HERE/'operator_bridge.c'),'-o',str(libpath)]
    subprocess.run(command,check=True)
    lib = C.CDLL(str(libpath.resolve()))
    ptr = C.POINTER(C.c_int32)
    lib.test_bind.argtypes = [C.c_void_p]
    lib.test_linear.argtypes = [C.c_int,ptr,ptr]
    lib.test_full.argtypes = [C.c_int,C.c_int,ptr,ptr]
    lib.test_l2.argtypes = [ptr,C.c_int]
    lib.test_rms.argtypes = [ptr,ptr,C.c_int,C.c_uint32,C.c_uint32]
    lib.test_clips.restype = C.c_uint32
    blob = C.create_string_buffer((image/'model.bin').read_bytes())
    lib.test_bind(blob)
    model = Qwen3_5ForCausalLM.from_pretrained(checkpoint,torch_dtype=torch.float32,
                                             attn_implementation='eager').eval()
    rng = np.random.default_rng(331)

    def pointer(array):
        return array.ctypes.data_as(ptr)

    q = (rng.normal(size=16)*4096).round().astype(np.int32)
    expected = l2norm(torch.tensor(q.astype(np.float32)/4096)).numpy()
    lib.test_l2(pointer(q),16)
    l2_error = float(np.max(np.abs(q/32768-expected)))
    results = []
    for layer_index in [0,3]:
        lib.test_bind(blob)
        cache = DynamicCache(config=model.config)
        hf_layer = model.model.layers[layer_index]
        rows = []
        for pos in range(8):
            inp = (rng.normal(size=model.config.hidden_size)*4096).round().astype(np.int32)
            actual = np.zeros_like(inp)
            h = torch.tensor(inp.astype(np.float32)/4096).reshape(1,1,-1)
            with torch.inference_mode():
                if layer_index == 0:
                    expected = hf_layer.linear_attn(h,cache_params=cache)
                    lib.test_linear(layer_index,pointer(inp),pointer(actual))
                else:
                    positions = torch.tensor([[pos]])
                    rope = model.model.rotary_emb(h,positions)
                    expected,_ = hf_layer.self_attn(h,position_embeddings=rope,
                                                   attention_mask=None,past_key_values=cache)
                    lib.test_full(layer_index,pos,pointer(inp),pointer(actual))
            target = expected.numpy().reshape(-1)
            diff = actual.astype(np.float64)/4096-target
            rows.append(dict(position=pos,max_absolute_error=float(np.max(np.abs(diff))),
                             rmse=float(np.sqrt(np.mean(diff**2))),
                             float_rms=float(np.sqrt(np.mean(target**2)))))
        results.append(dict(layer=layer_index,kind=meta['graph'][layer_index]['kind'],positions=rows,
                            numeric_clips=lib.test_clips()))
    report = dict(status='measured',scope='C operators versus unmodified HF, not neural tolerance',
                  checkpoint=str(checkpoint),l2_max_absolute_error=l2_error,layers=results,
                  source_sha256=hashlib.sha256((HERE/'runtime.c').read_bytes()).hexdigest(),
                  model_sha256=hashlib.sha256((image/'model.bin').read_bytes()).hexdigest())
    (image/'operators.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--image',type=Path,required=True)
    a = p.parse_args()
    check(a.checkpoint,a.image)
