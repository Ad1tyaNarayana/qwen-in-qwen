"""Materialize the ordinary full-vocabulary stock model one emitted layer at a time."""
import gc
import inspect

import torch


def build_model(circuit, device):
    from transformers import GenerationConfig, Qwen3_5ForCausalLM, Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
    from chipformer.q35 import emit
    from chipformer.q35.tiny import BASE, CODE_GAIN, pick_layout
    from chipformer.rv import spec
    from chipformer.rv.tiny import embedding_matrix
    layout = pick_layout(len(circuit.dims))
    count = circuit.n_layers()
    vocabulary = -(-(spec.VOCAB + 1) // 64) * 64
    config = Qwen3_5TextConfig(**dict(BASE, hidden_size=layout.hidden, vocab_size=vocabulary,
        num_hidden_layers=count + 1,
        layer_types=['linear_attention'] * count + ['full_attention'],
        linear_num_key_heads=circuit.target.key_heads,
        linear_num_value_heads=circuit.target.value_heads,
        intermediate_size=circuit.target.intermediate, eos_token_id=spec.tok('HALT'),
        pad_token_id=spec.VOCAB, bos_token_id=spec.VOCAB))
    with torch.device('meta'):
        model = Qwen3_5ForCausalLM(config)
    parameters = sum(p.numel() for p in model.parameters())
    source = inspect.getsource(emit.emit_layers)
    replacements = [
        ('def emit_layers(', 'def iter_layers('),
        ('            out[p + k] = t', '        yield {p + k: t for k, t in layer.items()}'),
        ('    return out', ''),
    ]
    for before, after in replacements:
        assert source.count(before) == 1, before
        source = source.replace(before, after)
    namespace = dict(emit.__dict__)
    exec(compile(source, '<streamed stock emit_layers>', 'exec'), namespace)
    loaded = set()
    for weights in namespace['iter_layers'](circuit, layout, config.to_dict(), 0,
                                           'model.', None, set(circuit.keep)):
        weights = {k: v.to(dtype=torch.bfloat16, device=device) for k, v in weights.items()}
        _, unexpected = model.load_state_dict(weights, strict=False, assign=True)
        assert not unexpected
        loaded.update(weights)
        del weights
    gc.collect()
    norm = torch.full((layout.hidden,), -1.0)
    for dim in circuit.keep:
        norm[layout.dim(dim)] = CODE_GAIN
    emit.check_bf16('final norm', norm)
    weights = {'model.embed_tokens.weight': embedding_matrix(circuit, layout, vocabulary).to(dtype=torch.bfloat16, device=device),
               'model.norm.weight': norm.to(dtype=torch.bfloat16, device=device)}
    _, unexpected = model.load_state_dict(weights, strict=False, assign=True)
    assert not unexpected
    loaded.update(weights)
    for name in set(model.state_dict()) - loaded:
        if name == 'lm_head.weight':
            continue
        assert name.startswith(f'model.layers.{count}.')
        parent, leaf = name.rsplit('.', 1)
        original = getattr(model.get_submodule(parent), leaf)
        setattr(model.get_submodule(parent), leaf, torch.nn.Parameter(
            torch.zeros(original.shape, dtype=torch.bfloat16, device=device), requires_grad=False))
    model.tie_weights()
    model.model.rotary_emb = Qwen3_5TextRotaryEmbedding(config).to(device)
    model.layout = layout
    model.generation_config = GenerationConfig(eos_token_id=spec.tok('HALT'), pad_token_id=spec.VOCAB, do_sample=False)
    model.eval()
    assert all(p.device.type == device and p.dtype == torch.bfloat16 for p in model.parameters())
    del weights
    gc.collect()
    return model, dict(parameters=parameters, bf16_parameter_bytes=parameters * 2,
                       hidden=layout.hidden, config=config.to_dict(), streamed_emitter=source)


def boot_cache(model, circuit, machine):
    from chipformer.rv import hf, spec
    from constructor import snapshot
    snapshot(circuit, machine)
    device = next(model.parameters()).device
    out = model(input_ids=torch.tensor([[spec.VOCAB]], device=device), use_cache=True)
    hf.inject(circuit, out.past_key_values)
    return out.past_key_values
