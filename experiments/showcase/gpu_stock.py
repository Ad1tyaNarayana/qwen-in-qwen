"""Run ordinary stock generate and hash the actual loaded BF16 network first."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/neural-accelerator'))
sys.path.insert(0,str(ROOT/'experiments/tiny-llm-ndot'))
import stock
import stock_prefix
base_build=stock.build_model


def build(*args,**kwargs):
    model,meta=base_build(*args,**kwargs)
    digest=hashlib.sha256()
    import torch
    for name,value in sorted(model.state_dict().items()):
        flat=value.detach().view(-1)
        assert flat.dtype==torch.bfloat16
        for part in flat.split(4*1024*1024):
            digest.update(part.cpu().contiguous().view(torch.uint16).numpy().astype('<u2',copy=False).tobytes())
    weights=json.loads((ROOT/'experiments/showcase/evidence/weights.json').read_text())
    assert digest.hexdigest()==weights['sha256'],(digest.hexdigest(),weights['sha256'])
    meta['network_weight_sha256']=digest.hexdigest()
    meta['wrapper_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return model,meta


if __name__=='__main__':
    stock.build_model=build
    stock_prefix.main()
