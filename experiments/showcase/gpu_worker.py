"""Use the accepted fused checkpoint worker and bind its generic constructor."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/tiny-llm-ndot-fused'))
import story
base_build=story.build


def build(args,meta):
    result=base_build(args,meta)
    fr,fused,spec,metadata=result
    weights=json.loads((ROOT/'experiments/showcase/evidence/weights.json').read_text())
    actual=fused['circuit'].source_manifest
    # Prepared source bytes use platform line endings; generated source text is identical.
    assert {k:v for k,v in actual.items() if k!='base_sha256'}=={k:v for k,v in weights['constructor'].items() if k!='base_sha256'}
    metadata['scripts']['experiments/showcase/gpu_worker.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    metadata['network_weight_sha256']=weights['sha256']
    metadata['weight_binding']='stock-emitted BF16 hash; identical generic constructor source manifest, no guest specialization'
    return result


if __name__=='__main__':
    story.worker.build=build
    story.worker.main()
