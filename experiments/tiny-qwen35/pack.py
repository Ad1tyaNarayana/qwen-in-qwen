"""Pack llama2.c or an HF Qwen3.5 state dictionary as data for one runtime."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys

from format import Image, layout, LLAMA, QWEN_FULL, QWEN_DELTA

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / 'experiments/tiny-llm-rv32'


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tokenizer(image, path):
    raw = path.read_bytes()
    cursor = 4
    starts, lengths, scores, pieces = [], [], [], bytearray()
    for _ in range(image.header['vocab']):
        score, n = struct.unpack_from('<fi',raw,cursor)
        cursor += 8
        if not math.isfinite(score) or not 0 <= n <= 4096 or cursor+n > len(raw):
            raise ValueError('Invalid tokenizer')
        starts.append(len(pieces))
        lengths.append(n)
        scores.append(score)
        pieces.extend(raw[cursor:cursor+n])
        cursor += n
    if cursor != len(raw):
        raise ValueError('Tokenizer vocabulary mismatch')
    image.header.update(vocab_offsets=image.put('I',starts), vocab_lengths=image.put('I',lengths),
                        vocab_bytes=image.put('B',pieces))
    return dict(scores=scores, pieces=[list(pieces[s:s+n]) for s,n in zip(starts,lengths)])


def encode(prompt, tok):
    current = sys.modules.get('pack')
    sys.modules['pack'] = load_module('legacy_pack',OLD/'pack.py')
    try:
        encoder = load_module('legacy_encoder', OLD/'build.py')
    finally:
        if current is None:
            del sys.modules['pack']
        else:
            sys.modules['pack'] = current
    tokens = encoder.encode(prompt,tok)
    return tokens


def finish(image, out, prompt, steps, tok, meta):
    tokens = encode(prompt,tok)
    if not len(tokens) <= steps <= image.header['context']:
        raise ValueError('Require prompt length <= steps <= context')
    image.header.update(prompt=image.put('I',tokens),prompt_len=len(tokens),steps=steps)
    memory = layout(image)
    raw = image.finish()
    if len(raw) > 4194304-16384:
        raise ValueError('DROM budget exceeded')
    out.mkdir(parents=True,exist_ok=True)
    (out/'model.bin').write_bytes(raw)
    (out/'tokenizer.json').write_text(json.dumps(tok))
    meta.update(header=image.header, graph=image.layers, ram=memory, prompt=prompt,
                prompt_tokens=tokens, steps=steps, rom_bytes=len(raw),
                rom_sha256=hashlib.sha256(raw).hexdigest())
    (out/'pack.json').write_text(json.dumps(meta,indent=2)+'\n')
    return meta


def llama(model, tokpath, out, context, prompt, steps, kv_bits=16, kv_frac=10):
    oldpack = load_module('legacy_pack',OLD/'pack.py')
    legacy = out/'legacy'
    meta = oldpack.pack(model,tokpath,legacy,context,kv_bits,kv_frac,1024)
    image = Image(meta['layers'])
    base = image.put('B',(legacy/'model.bin').read_bytes())
    offsets = {k:v+base for k,v in meta['offsets'].items()}
    d, hidden, kd = meta['dim'],meta['hidden'],meta['dim']*meta['kvheads']//meta['heads']
    image.header.update(dim=d,vocab=meta['vocab'],context=context,kv_bits=kv_bits,
                        kv_frac=kv_frac,norm_eps=168,final_norm=offsets['rms_final'])
    for name in ['emb','emb_scale','cls','cls_scale','exp','sigmoid',
                 'vocab_offsets','vocab_lengths','vocab_bytes']:
        image.header[name] = offsets[name]
    for i in range(meta['layers']):
        l = dict(kind=LLAMA,hidden=hidden,heads=meta['heads'],kvheads=meta['kvheads'],
                 head_dim=d//meta['heads'],key_heads=0,value_heads=0,key_dim=0,value_dim=0,
                 rotary=d//meta['heads'],rope=offsets['rope'],eps=168,
                 inv_sqrt=round(4096/math.sqrt(d//meta['heads'])),
                 pre_norm=offsets['rms_att']+4*i*d,post_norm=offsets['rms_ffn']+4*i*d)
        for name, old, rows, cols in [('q','wq',d,d),('k','wk',kd,d),('v','wv',kd,d),
                                      ('o','wo',d,d),('gate','w1',hidden,d),
                                      ('up','w3',hidden,d),('down','w2',d,hidden)]:
            l[name] = offsets[old]+i*rows*cols
            l[name+'s'] = offsets[old+'_scale']+4*i*rows
        image.layers.append(l)
    tok = json.loads((legacy/'tokenizer.json').read_text())
    return finish(image,out,prompt,steps,tok,dict(kind='llama2.c',legacy=meta,
                  source_sha256=hashlib.sha256(model.read_bytes()).hexdigest()))


def qwen(checkpoint, tokpath, out, context, prompt, steps, quantization='midpoint', calibration=None):
    import torch
    from transformers import Qwen3_5ForCausalLM
    model = Qwen3_5ForCausalLM.from_pretrained(checkpoint,torch_dtype=torch.float32,
                                             attn_implementation='eager').eval()
    config = model.config
    if not (0 < config.hidden_size <= 1024 and 0 < config.intermediate_size <= 4096 and
            0 < config.num_hidden_layers <= 32 and 259 <= config.vocab_size <= 8192 and
            0 < context <= 512 and config.num_attention_heads % config.num_key_value_heads == 0 and
            0 < config.linear_conv_kernel_dim <= 16 and
            0 < config.linear_key_head_dim <= 1024 and 0 < config.linear_value_head_dim <= 1024 and
            config.num_attention_heads*config.head_dim <= 4096 and
            2*config.linear_num_key_heads*config.linear_key_head_dim +
            config.linear_num_value_heads*config.linear_value_head_dim <= 4096):
        raise ValueError('Qwen geometry exceeds the bounded integer profile')
    if not 0 < config.rms_norm_eps <= 0.001:
        raise ValueError('RMS epsilon is outside the fixed-point profile')
    if config.hidden_act != 'silu' or config.attention_bias:
        raise ValueError('Only bias-free SiLU Qwen3.5 text models are supported')
    if config.rope_parameters['rope_type'] != 'default':
        raise ValueError('Only default partial text RoPE is supported')
    if context > config.max_position_embeddings:
        raise ValueError('Context exceeds trained sequence limit')
    state = model.state_dict()
    methods = quantization.split('+')
    if any(method not in ['row','midpoint','minmse','equalize','covariance'] for method in methods):
        raise ValueError('Unknown quantization method')
    covariance = None
    if 'covariance' in methods:
        if calibration is None:
            raise ValueError('Covariance quantization requires --calibration')
        covariance = torch.load(calibration,map_location='cpu',weights_only=True)
        assert covariance['checkpoint_sha256'] == hashlib.sha256((checkpoint/'model.safetensors').read_bytes()).hexdigest()
    gains = {}
    if 'equalize' in methods:
        for i,kind in enumerate(config.layer_types):
            if kind != 'linear_attention':
                continue
            prefix = f'model.layers.{i}.linear_attn.'
            key,conv = prefix+'in_proj_qkv.weight',prefix+'conv1d.weight'
            projection = state[key].float()
            kernel = state[conv].float()
            gain = (kernel.abs().amax(dim=(1,2))/projection.norm(dim=1).clamp(min=1e-8)).sqrt().clamp(0.0625,16)
            state[key] = projection*gain[:,None]
            state[conv] = kernel/gain[:,None,None]
            gains[str(i)] = dict(minimum=float(gain.min()),maximum=float(gain.max()),mean=float(gain.mean()))
    image = Image(config.num_hidden_layers)
    d = config.hidden_size
    image.header.update(dim=d,vocab=config.vocab_size,context=context,kv_bits=16,kv_frac=10,
                        norm_eps=max(1,round(config.rms_norm_eps*2**24)),activation_limit=2**23)
    errors = {}

    def matrix(key):
        value = state[key].float().cpu()
        peak = value.abs().amax(dim=1)
        scales = (peak/127*2**24).round().clamp(min=1).to(torch.int64)
        if 'minmse' in methods:
            best = scales.clone()
            error = torch.full_like(peak,float('inf'))
            for factor in [0.96,0.97,0.98,0.99,1.0,1.01,1.02,1.03,1.04]:
                candidate = (scales.float()*factor).round().clamp(min=1).to(torch.int64)
                step = candidate[:,None].float()/2**24
                trial = (value/step).round().clamp(-127,127)*step
                mse = (value-trial).square().mean(dim=1)
                select = mse < error
                best = torch.where(select,candidate,best)
                error = torch.minimum(error,mse)
            scales = best
        if scales.max().item() >= 2**31:
            raise ValueError(f'Scale overflow {key}')
        quant = (value/(scales[:,None].float()/2**24)).round().clamp(-127,127).to(torch.int8)
        if covariance is not None:
            name = 'lm_head.weight' if key == 'model.embed_tokens.weight' else key
            H = covariance['H'][name].double()
            H = H/H.diag().mean().clamp(min=1e-12)
            if key == 'model.embed_tokens.weight':
                H = (H+torch.eye(H.shape[0]))/2
            H = (H+H.T)/2+0.01*torch.eye(H.shape[0])
            order = H.diag().argsort(descending=True,stable=True)
            H = H[order][:,order]
            upper = torch.linalg.cholesky(torch.linalg.inv(H),upper=True)
            working = value[:,order].double().clone()
            step = scales.double()/2**24
            result = torch.empty_like(working,dtype=torch.int8)
            for j in range(working.shape[1]):
                rounded = (working[:,j]/step).round().clamp(-127,127)
                result[:,j] = rounded.to(torch.int8)
                error = (working[:,j]-rounded*step)/upper[j,j]
                working[:,j:] -= error[:,None]*upper[j,j:][None,:]
            quant[:,order] = result
        errors[key] = float((value-quant.float()*scales[:,None]/2**24).square().mean().sqrt())
        return image.put('b',quant.flatten().tolist()), image.put('i',scales.tolist())

    def fixed(key, add=0, exp=False, frac=12):
        value = state[key].float().cpu()
        if exp:
            value = value.exp()
        value = ((value+add)*2**frac).round()
        limit = 2**31-1 if frac == 20 else 2**23
        if not value.isfinite().all() or value.abs().max() > limit:
            raise ValueError(f'Fixed-point weight range exceeded {key}')
        return image.put('i',value.to(torch.int64).flatten().tolist())

    image.header['emb'],image.header['emb_scale'] = matrix('model.embed_tokens.weight')
    if config.tie_word_embeddings:
        image.header['cls'],image.header['cls_scale'] = image.header['emb'],image.header['emb_scale']
    else:
        image.header['cls'],image.header['cls_scale'] = matrix('lm_head.weight')
    image.header['final_norm'] = fixed('model.norm.weight',1)
    center = 0.5 if 'midpoint' in methods else 0
    image.header['exp'] = image.put('I',[round(math.exp(-(i+center)/1024)*2**20) for i in range(16385)])
    image.header['sigmoid'] = image.put('H',[round(32768/(1+math.exp(-(i-2048+center)/128))) for i in range(4097)])
    image.header['softplus'] = image.put('i',[round(math.log1p(math.exp((i-2048+center)/128))*2**20) for i in range(4097)])
    rotary = int(config.head_dim*config.rope_parameters.get('partial_rotary_factor',1))
    if rotary % 2 or not 0 < rotary <= config.head_dim:
        raise ValueError('Rotary dimension must be positive and even')
    rope = []
    theta = config.rope_parameters['rope_theta']
    for pos in range(context):
        for i in range(0,rotary,2):
            angle = pos/theta**(i/rotary)
            rope.extend([round(math.cos(angle)*32767),round(math.sin(angle)*32767)])
    rope_offset = image.put('h',rope)
    for i, kind in enumerate(config.layer_types):
        prefix = f'model.layers.{i}.'
        l = dict(kind=QWEN_DELTA if kind == 'linear_attention' else QWEN_FULL,
                 hidden=config.intermediate_size,heads=config.num_attention_heads,
                 kvheads=config.num_key_value_heads,head_dim=config.head_dim,
                 key_heads=config.linear_num_key_heads,value_heads=config.linear_num_value_heads,
                 key_dim=config.linear_key_head_dim,value_dim=config.linear_value_head_dim,
                 kernel=config.linear_conv_kernel_dim,eps=image.header['norm_eps'],
                 pre_norm=fixed(prefix+'input_layernorm.weight',1),
                 post_norm=fixed(prefix+'post_attention_layernorm.weight',1))
        if l['value_heads'] % l['key_heads']:
            raise ValueError('Value heads must be divisible by key heads')
        for name,hf in [('gate','gate_proj'),('up','up_proj'),('down','down_proj')]:
            l[name],l[name+'s'] = matrix(prefix+'mlp.'+hf+'.weight')
        if l['kind'] == QWEN_FULL:
            for name,hf in [('q','q_proj'),('k','k_proj'),('v','v_proj'),('o','o_proj')]:
                l[name],l[name+'s'] = matrix(prefix+'self_attn.'+hf+'.weight')
            l.update(rotary=rotary,rope=rope_offset,inv_sqrt=round(4096/math.sqrt(config.head_dim)),
                     q_norm=fixed(prefix+'self_attn.q_norm.weight',1),
                     k_norm=fixed(prefix+'self_attn.k_norm.weight',1))
        else:
            for name,hf in [('q','in_proj_qkv'),('z','in_proj_z'),('b','in_proj_b'),
                            ('a','in_proj_a'),('o','out_proj')]:
                l[name],l['ascale' if name == 'a' else name+'s'] = matrix(prefix+'linear_attn.'+hf+'.weight')
            l.update(conv=fixed(prefix+'linear_attn.conv1d.weight'),
                     dt=fixed(prefix+'linear_attn.dt_bias'),A=fixed(prefix+'linear_attn.A_log',exp=True,frac=20),
                     gated_norm=fixed(prefix+'linear_attn.norm.weight'),
                     inv_sqrt=round(4096/math.sqrt(config.linear_key_head_dim)))
        image.layers.append(l)
    tok = tokenizer(image,tokpath)
    return finish(image,out,prompt,steps,tok,dict(kind='Qwen3_5ForCausalLM',config=config.to_dict(),
                  parameters=sum(p.numel() for p in model.parameters()),weight_rmse=errors,
                  quantization=quantization,conv_equalization=gains,
                  calibration_sha256=hashlib.sha256(calibration.read_bytes()).hexdigest() if calibration else None))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind',choices=['llama','qwen'],required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--tokenizer',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--context',type=int,default=64)
    p.add_argument('--steps',type=int,default=5)
    p.add_argument('--prompt',default='Once')
    p.add_argument('--quantization',default='midpoint',help='Qwen LUT and matrix quantization method')
    p.add_argument('--calibration',type=Path)
    a = p.parse_args()
    if a.kind == 'llama':
        result = llama(a.model,a.tokenizer,a.out.resolve(),a.context,a.prompt,a.steps)
    else:
        result = qwen(a.model,a.tokenizer,a.out.resolve(),a.context,a.prompt,a.steps,a.quantization,a.calibration)
    print(json.dumps({k:v for k,v in result.items() if k not in ['graph','legacy','config','weight_rmse','header']},indent=2))
