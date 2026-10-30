"""Run the unchanged checkpointed story worker with the accepted fused adapter."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/tiny-llm-ndot'))
import worker
build_unfused=worker.build


def build(a,meta):
    fr,info,spec,metadata=build_unfused(a,meta)
    from fusion import install
    fused=install(fr)
    assert len(fused['scratch_buffers'])==1
    fr.capture()
    from runtime import reset
    from chipformer.rv import elf
    import reference
    initial=reference.Machine.from_program(elf.program(a.inputs/'engine.elf'))
    initial.maximum=4
    reset(fr,initial,fused)
    fused['circuit']=info['circuit']
    metadata['mode']='unified W=4 unchanged circuit with data-ROM batching and single-kernel feedback'
    for path in (Path(__file__),Path(__file__).with_name('fusion.py'),Path(__file__).with_name('runtime.py'),
                 ROOT/'experiments/dmc-feedback-fusion/fusion.py'):
        metadata['scripts'][str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    metadata['rom_batches']=[dict(indices=b['indices'],value_heads=b['V'],widths=b['widths']) for b in fused['batches'].values()]
    return fr,fused,spec,metadata


if __name__=='__main__':
    worker.build=build
    worker.main()
