"""Compare every emitted BF16 word against the saved passing constructor."""
import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import traceback

from prepare import ROOT, prepare


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', tensors=[], baseline=str(a.baseline))
    try:
        prepare(a.out / 'scratch')
        import torch
        torch.set_num_threads(4)
        from chipformer.q35 import emit
        from chipformer.q35.tiny import BASE, pick_layout
        from chipformer.rv import spec
        from chipformer.rv.tiny import embedding_matrix
        from transformers import Qwen3_5TextConfig
        from constructor import build
        module_spec = importlib.util.spec_from_file_location('passing_constructor', a.baseline / 'constructor.py')
        old = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(old)
        accepted = json.loads(Path(__file__).with_name('stock-m4.json').read_text())
        assert hashlib.sha256((a.baseline / 'constructor.py').read_bytes()).hexdigest() == accepted['constructor']['constructor_sha256']
        assert hashlib.sha256((a.baseline / 'integer_counters.py').read_bytes()).hexdigest() == accepted['constructor']['integer_counters_sha256']
        counter_spec = importlib.util.spec_from_file_location('integer_counters', a.baseline / 'integer_counters.py')
        baseline_counter = importlib.util.module_from_spec(counter_spec)
        counter_spec.loader.exec_module(baseline_counter)
        import integer_counters
        sys.modules['integer_counters'] = baseline_counter
        try:
            before = old.build(4, stock_integer_counters=True)
        finally:
            sys.modules['integer_counters'] = integer_counters
        after = build(4)
        assert before.source_manifest['generated_sha256'] == after.source_manifest['generated_sha256']
        layout = pick_layout(len(after.dims))
        config = Qwen3_5TextConfig(**dict(BASE, hidden_size=layout.hidden,
            vocab_size=-(-(spec.VOCAB+1)//64)*64, num_hidden_layers=after.n_layers()+1,
            layer_types=['linear_attention']*after.n_layers()+['full_attention'],
            linear_num_key_heads=after.target.key_heads, linear_num_value_heads=after.target.value_heads,
            intermediate_size=after.target.intermediate))
        assert before.stats() == after.stats() and before.keep == after.keep
        source = inspect.getsource(emit.emit_layers).replace('def emit_layers(', 'def iter_layers(')
        source = source.replace('            out[p + k] = t', '        yield {p + k: t for k, t in layer.items()}').replace('    return out', '')
        ns = dict(emit.__dict__)
        exec(compile(source, '<weight comparison emitter>', 'exec'), ns)
        def layers(c):
            return ns['iter_layers'](c, layout, config.to_dict(), 0, 'model.', None, set(c.keep))
        def compare(name, x, y):
            x, y = x.to(torch.bfloat16), y.to(torch.bfloat16)
            assert torch.equal(x.view(torch.int16), y.view(torch.int16)), name
            digest = hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest()
            report['tensors'].append(dict(name=name, shape=list(x.shape), sha256=digest, unequal_words=0))
        for i, (left, right) in enumerate(zip(layers(before), layers(after), strict=True)):
            assert left.keys() == right.keys()
            for name in left:
                compare(name, left[name], right[name])
            print(json.dumps(dict(layer=i, tensors=len(report['tensors']))), flush=True)
        compare('embedding', embedding_matrix(before, layout, config.vocab_size), embedding_matrix(after, layout, config.vocab_size))
        report.update(status='pass', graph_generated_sha256=after.source_manifest['generated_sha256'],
                      constructor=after.source_manifest, geometry=after.stats(),
                      final_norm_equal=before.keep == after.keep, zero_final_layer_equal=True,
                      config=config.to_dict())
    except Exception as error:
        report.update(status='fail', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        (a.out / 'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
