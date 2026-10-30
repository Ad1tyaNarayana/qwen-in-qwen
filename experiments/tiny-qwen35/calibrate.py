"""Collect CPU-only input covariance for offline weight rounding."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import Qwen3_5ForCausalLM

from pack import encode, ROOT


def calibrate(checkpoint, tokenizer, stories, out):
    torch.set_num_threads(1)
    model = Qwen3_5ForCausalLM.from_pretrained(checkpoint,torch_dtype=torch.float32,
                                             attn_implementation='eager').eval()
    tokens = json.loads(tokenizer.read_text())
    texts = json.loads(stories.read_text())[:64]
    H,counts,handles = {},{},[]

    def hook(name):
        def record(module, args):
            x = args[0].detach().float().reshape(-1,args[0].shape[-1])
            if name not in H:
                H[name] = torch.zeros((x.shape[1],x.shape[1]))
                counts[name] = 0
            H[name] += x.T@x
            counts[name] += len(x)
        return record

    for name,module in model.named_modules():
        if isinstance(module,torch.nn.Linear):
            handles.append(module.register_forward_pre_hook(hook(name+'.weight')))
    with torch.inference_mode():
        for text in texts:
            ids = encode(text,tokens)[:64]
            model(input_ids=torch.tensor([ids]),use_cache=False)
    for handle in handles:
        handle.remove()
    metadata = dict(checkpoint_sha256=hashlib.sha256((checkpoint/'model.safetensors').read_bytes()).hexdigest(),
                    stories_sha256=hashlib.sha256(stories.read_bytes()).hexdigest(),
                    excerpts=len(texts),context=64,counts=counts,
                    scope='CPU-only floating calibration; only quantized weights enter DROM, no guest state')
    torch.save(dict(H={k:v/counts[k] for k,v in H.items()},**metadata),out)
    out.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--tokenizer',type=Path,required=True)
    p.add_argument('--stories',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a = p.parse_args()
    calibrate(a.checkpoint,a.tokenizer,a.stories,a.out)
