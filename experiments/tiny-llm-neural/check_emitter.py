"""Check streamed packaging against the original emitter on three circuit layers."""
import copy
import hashlib
import inspect
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT / '.cache/tiny-llm/neural-stories/reference-package'))
from chipformer.rv import core
from chipformer.q35 import emit
from chipformer.q35.tiny import BASE,pick_layout

c = core.build()
layout = pick_layout(len(c.dims))
config = dict(BASE,hidden_size=layout.hidden,linear_num_key_heads=c.target.key_heads,
              linear_num_value_heads=c.target.value_heads,intermediate_size=c.target.intermediate)
source = inspect.getsource(emit.emit_layers)
source = source.replace('def emit_layers(', 'def iter_layers(')
source = source.replace('            out[p + k] = t', '        yield {p + k:t for k,t in layer.items()}')
source = source.replace('    return out', '')
namespace = dict(emit.__dict__)
exec(compile(source,'<streamed original emit_layers>','exec'),namespace)
reports = []
torch.set_num_threads(4)
for index in (0,22,43):
    g = copy.copy(c)
    g.nodes = [copy.copy(n) for n in c.nodes]
    for node in g.nodes:
        node.level -= 2*index
    g.n_layers = lambda:1
    original = emit.emit_layers(g,layout,config,0,'model.',None,set(c.keep))
    streamed, = namespace['iter_layers'](g,layout,config,0,'model.',None,set(c.keep))
    assert original.keys() == streamed.keys()
    digest = hashlib.sha256()
    for key in original:
        a,b = original[key].to(torch.bfloat16),streamed[key].to(torch.bfloat16)
        assert torch.equal(a.view(torch.int16), b.view(torch.int16)), key
        digest.update(a.view(torch.int16).numpy().tobytes())
    reports.append(dict(layer=index,tensors=len(original),bf16_word_mismatches=0,sha256=digest.hexdigest()))
    del original,streamed,a,b
report = dict(scope='Original unchanged emitter versus layer-streaming dictionary packaging, first/middle/last layer at full geometry',
              layers=reports,source_sha256=hashlib.sha256(source.encode()).hexdigest())
(ROOT / 'experiments/tiny-llm-neural/evidence/emitter.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
