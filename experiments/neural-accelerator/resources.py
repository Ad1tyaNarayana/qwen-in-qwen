"""Exact stock meta-model accounting and explicitly conditional compact cost estimates."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import tempfile

from prepare import ROOT, prepare

HERE = Path(__file__).resolve().parent
GIB = 2**30


def write_chart(result):
    rows = result['variants']
    free_budget = result['gpu_inventory']['free_mib']/1024-1
    maximum = max(row['estimated_peak_reserved_bytes']/GIB for row in rows)*1.12
    left, plot = 132, 500
    x = lambda value: left+plot*value/maximum
    height = 58+44*len(rows)
    parts = [f'<svg viewBox="0 0 728 {height}" role="img" aria-label="Stock peak reserved memory compared with sampled VRAM budget">']
    for tick in (0, 8, 16, 24, 32):
        px = x(tick)
        parts += [f'<line x1="{px}" x2="{px}" y1="28" y2="{height-22}" class="grid"/>',
                  f'<text x="{px}" y="{height-5}" text-anchor="middle">{tick}</text>']
    labels = {'original': 'Original', 'w4-before': 'W=4 before', 'w4-after': 'W=4 after', 'w8-after': 'W=8 after'}
    for index, row in enumerate(rows):
        value = row['estimated_peak_reserved_bytes']/GIB
        y = 40+44*index
        label_x = x(value)-8 if value < free_budget else x(value)+8
        anchor = 'end' if value < free_budget else 'start'
        parts += [f'<text x="0" y="{y+5}">{labels[row["variant"]]}</text>',
                  f'<rect x="{left}" y="{y-11}" width="{x(value)-left}" height="22" rx="3" fill="var(--chart-{index+1})"/>',
                  f'<text x="{label_x}" y="{y+5}" text-anchor="{anchor}">{value:.2f}</text>']
    budget_x = x(free_budget)
    parts += [f'<line x1="{budget_x}" x2="{budget_x}" y1="24" y2="{height-22}" class="budget"/>',
              f'<text x="{budget_x}" y="13" text-anchor="middle">{free_budget:.2f} GiB available with headroom</text>', '</svg>']
    html = '''<!doctype html><html><head><meta charset="utf-8"><style>
html,body{margin:0;font-family:var(--font-sans);color:var(--foreground)}
svg{display:block;width:100%;height:''' + str(height) + '''px}text{fill:var(--foreground);font-size:12px}
.grid{stroke:var(--border);stroke-width:1}.budget{stroke:var(--warning);stroke-width:2;stroke-dasharray:4 3}
p{font-size:12px;line-height:1.5;margin:4px 0;color:var(--muted-foreground)}
</style></head><body>''' + ''.join(parts) + f'''<p>Stock peak reserved memory in GiB. Original is measured; ndot variants are calibrated estimates.</p>
<p>The sampled free VRAM is {result['gpu_inventory']['free_mib']/1024:.2f} GiB. Required free headroom is 1 GiB; the allocator cap is 14 GiB. No new GPU allocation was attempted.</p></body></html>'''
    (HERE/'resources.html').write_text(html)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--include-w8', action='store_true')
    args = parser.parse_args()
    reports = {'w4-before': json.loads((HERE/'ir-w4-before-reduction.json').read_text()),
               'w4-after': json.loads((HERE/'ir-w4.json').read_text())}
    if args.include_w8:
        reports['w8-after'] = json.loads((HERE/'ir-w8.json').read_text())
    assert all(report['status'] == 'pass' for report in reports.values())
    assert reports['w4-after']['source_manifest']['reduced'] is True
    if args.include_w8:
        assert reports['w8-after']['source_manifest']['reduced'] is True
    inventory = json.loads((HERE/'gpu-inventory.json').read_text())
    prepare(Path(tempfile.mkdtemp(prefix='resources-', dir=ROOT/'.cache/neural-accelerator')))
    import torch
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    from chipformer.q35.ir import Head
    from chipformer.q35.tiny import BASE, pick_layout
    from chipformer.rv import core, spec as S
    c = core.build()
    original = c.stats()
    original.update(hidden=pick_layout(len(c.dims)).hidden,
                    logical_value_heads=sum(len(node.vheads()) for node in c.nodes if isinstance(node, Head)))
    variants = dict(original=original, **{key: report['stats'] for key, report in reports.items()})
    prior = json.loads((HERE.parent/'tiny-llm-neural/evidence/hf.json').read_text())
    rate = json.loads((HERE.parent/'tiny-llm-neural/evidence/gpu.json').read_text())['tokens_per_second']
    counts = json.loads((HERE.parent/'tiny-llm-rv32/evidence/evidence.json').read_text())
    rows = []
    for name, geometry in variants.items():
        config = dict(BASE, hidden_size=geometry['hidden'],
            vocab_size=-(-(S.VOCAB+1)//64)*64,
            num_hidden_layers=geometry['layers']+1,
            layer_types=['linear_attention']*geometry['layers']+['full_attention'],
            linear_num_key_heads=c.target.key_heads, linear_num_value_heads=c.target.value_heads,
            intermediate_size=c.target.intermediate, eos_token_id=S.tok('HALT'),
            pad_token_id=S.VOCAB, bos_token_id=S.VOCAB)
        with torch.device('meta'):
            model = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**config))
        parameters = sum(parameter.numel() for parameter in model.parameters())
        assert all(parameter.device.type == 'meta' for parameter in model.parameters())
        del model
        gc.collect()
        rows.append(dict(variant=name, geometry=geometry, config=config,
                         parameters=parameters, bf16_parameter_bytes=2*parameters))
    assert rows[0]['parameters'] == prior['parameters'] == 6212045824
    # Each stock linear layer allocates 32 fp32 128x128 recurrent matrices.
    state_per_layer = c.target.value_heads*128*128*4
    original_bytes = rows[0]['bf16_parameter_bytes']
    other_runtime = prior['cuda_peak_allocated_bytes']-original_bytes-original['layers']*state_per_layer
    reserve_gap = prior['cuda_peak_reserved_bytes']-prior['cuda_peak_allocated_bytes']
    category_us = dict(ffn=55.779, drom_heads=53.190, drom_apply=20.464, ram=41.333,
                       other_heads=15.201, code_rom=10.013, other_apply=23.750, feedback=5.782)
    profiler_span = 239.208
    unmodeled = profiler_span-sum(category_us.values())
    for row in rows:
        name = row['variant']
        geometry = row['geometry']
        layer_ratio = geometry['layers']/original['layers']
        width_ratio = geometry['dims']/original['dims']
        overhead = geometry['layers']*state_per_layer + other_runtime*max(1, geometry['hidden']/original['hidden'])
        allocated = row['bf16_parameter_bytes']+overhead
        reserved = allocated+reserve_gap
        row['estimated_peak_allocated_bytes'] = int(allocated)
        row['estimated_peak_reserved_bytes'] = int(reserved)
        row['fits_16_gib_with_1_gib_headroom_estimate'] = reserved+GIB <= 16*GIB
        row['within_14_gib_allocator_cap_estimate'] = reserved <= 14*GIB
        row['fits_sampled_free_with_1_gib_headroom_estimate'] = reserved+GIB <= inventory['free_mib']*2**20
        ratios = {key: 1.0 for key in category_us}
        ratios.update(ffn=geometry['ffn_units']/original['ffn_units'],
                      other_apply=layer_ratio*width_ratio, feedback=width_ratio)
        growth = (sum(category_us[key]*ratios[key] for key in category_us)+unmodeled*layer_ratio)/profiler_span
        us = 1e6/rate*growth
        row['compact_projection'] = dict(category_ratios=ratios, growth=growth, us_per_transition=us,
                                         baseline_trace_tokens_per_second=rate, positions=[])
        if name != 'original':
            width = 8 if name.startswith('w8') else 4
            for position in (4, 63):
                measured = counts['positions'][position]
                groups = measured['hypothetical_dot'][str(width)]['groups']
                tokens = [measured['trace_tokens']-measured['dot_body_tokens']+per_group*groups
                          for per_group in (7, 8)]
                row['compact_projection']['positions'].append(dict(position=position,
                    trace_tokens=tokens, minutes_per_llm_token=[value*us/60e6 for value in tokens]))
    result = dict(status='pass', scope='Exact meta parameter counts; estimated runtime memory and compact latency only',
        variants=rows, torch=torch.__version__, transformers=__import__('transformers').__version__,
        gpu_inventory=inventory,
        calibration=dict(path='../tiny-llm-neural/evidence/hf.json',
            peak_allocated_bytes=prior['cuda_peak_allocated_bytes'],
            peak_reserved_bytes=prior['cuda_peak_reserved_bytes'],
            original_bf16_bytes=original_bytes, state_per_layer_bytes=state_per_layer,
            other_runtime_bytes=other_runtime, reserve_gap_bytes=reserve_gap),
        cost_model=dict(profiler_span_us=profiler_span, category_us=category_us,
            unmodeled_us=unmodeled, baseline_rate_source='../tiny-llm-neural/evidence/gpu.json',
            assumptions=[
                'Scale FFN category by actual unit count, treating all units as equally costly.',
                'Head shapes, head counts, projection nonzero counts and RAM copies are unchanged; head and DROM-apply costs stay fixed.',
                'Residual/history apply scales by layers times physical residual dimensions; feedback scales by dimensions.',
                'Unmodeled profiler overhead scales by layers. Normalize to the 239.208 us profiler span before scaling the measured stories rate.',
                'Replace matrix dot-loop work only, with 7 to 8 tokens per aligned group; retain measured nondot work. Attention remains scalar.',
                'No aligned guest rebuild, refresh recount, compact timing, stock execution or whole-LLM prediction was performed here.']),
        memory_assumptions=[
            'Match the original stock config, complete chip vocabulary, tied embeddings and required zero full-attention layer.',
            'Add exact recurrent-state bytes per layer to calibrated nonparameter overhead; scale remaining overhead upward for wider hidden sizes only.',
            'Reuse the original reserved-minus-allocated gap. Fits flags are estimates, not an allocation or runtime proof.',
            'Apply the approved 14 GiB cap and at least 1 GiB free headroom before any future GPU allocation.'],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in [HERE/'ir-w4-before-reduction.json', HERE/'ir-w4.json',
                                   HERE.parent/'tiny-llm-neural/evidence/hf.json',
                                   HERE.parent/'tiny-llm-neural/evidence/gpu.json',
                                   HERE/'gpu-inventory.json',
                                   HERE.parent/'tiny-llm-rv32/evidence/evidence.json']
                      + ([HERE/'ir-w8.json'] if args.include_w8 else [])})
    (HERE/'resources.json').write_text(json.dumps(result, indent=2)+'\n')
    write_chart(result)
    for row in rows:
        print(row['variant'], row['parameters'], row['bf16_parameter_bytes'],
              f"{row['estimated_peak_allocated_bytes']/GIB:.3f} GiB estimated allocated peak",
              row['compact_projection']['positions'], flush=True)


if __name__ == '__main__':
    main()
