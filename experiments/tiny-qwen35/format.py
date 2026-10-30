"""Version-one DROM graph ABI and deterministic guest arena layout."""
import struct

MAGIC = 0x354E5751
ARENA_BYTES = 63120
HEADER = '''magic version bytes dim vocab context layers graph kv_bits kv_frac
norm_eps final_norm emb emb_scale cls cls_scale exp sigmoid softplus
vocab_offsets vocab_lengths vocab_bytes prompt prompt_len steps ram_bytes
x xb xb2 q tmp hb hb2 logits att quant z alpha beta activation_limit'''.split()
LAYER = '''kind hidden heads kvheads head_dim key_heads value_heads key_dim value_dim
kernel rotary rope inv_sqrt eps pre_norm post_norm q_norm k_norm gated_norm
conv dt A state conv_state keys values
q qs k ks v vs o os gate gates up ups down downs z zs b bs a ascale'''.split()
LLAMA, QWEN_FULL, QWEN_DELTA = 1, 2, 3


def words(fields, obj):
    return struct.pack('<' + 'I' * len(fields), *(obj.get(k, 0) for k in fields))


class Image:
    def __init__(self, layers):
        self.data = bytearray(4 * (len(HEADER) + layers * len(LAYER)))
        self.header = dict(magic=MAGIC, version=1, layers=layers, graph=4*len(HEADER))
        self.layers = []

    def put(self, fmt, values):
        self.data.extend(bytes(-len(self.data) % 8))
        offset = len(self.data)
        self.data.extend(struct.pack('<' + str(len(values)) + fmt, *values))
        return offset

    def finish(self):
        self.header['bytes'] = len(self.data)
        self.data[:4*len(HEADER)] = words(HEADER, self.header)
        for i, layer in enumerate(self.layers):
            offset = self.header['graph'] + i * 4 * len(LAYER)
            self.data[offset:offset+4*len(LAYER)] = words(LAYER, layer)
        return bytes(self.data)


def layout(image):
    h = image.header
    d, context = h['dim'], h['context']
    cursor = 0

    def alloc(n, size=4, alignment=4):
        nonlocal cursor
        cursor = (cursor + alignment-1) & ~(alignment-1)
        offset = cursor
        cursor += n * size
        return offset

    work = max([d] + [max(2*l['key_heads']*l['key_dim']+l['value_heads']*l['value_dim'],
                         l['heads']*l['head_dim']) for l in image.layers])
    qsize = max(d, max((2*l['heads']*l['head_dim'] if l['kind'] == QWEN_FULL else
                        2*l['key_heads']*l['key_dim']+l['value_heads']*l['value_dim'])
                       for l in image.layers))
    hidden = max(l['hidden'] for l in image.layers)
    qwen = any(l['kind'] != LLAMA for l in image.layers)
    for name, n in [('x',d), ('xb',max(d,work)), ('xb2',d), ('q',qsize), ('tmp',work),
                    ('hb',hidden), ('hb2',hidden), ('logits',h['vocab']), ('att',context)]:
        h[name] = alloc(n)
    h['quant'] = alloc(max(hidden,work,qsize), 2, 8)
    if qwen:
        h['z'] = alloc(work)
        h['alpha'] = alloc(max(l['value_heads'] for l in image.layers))
        h['beta'] = alloc(max(l['value_heads'] for l in image.layers))
    states = []
    for l in image.layers:
        begin = cursor
        if l['kind'] == QWEN_DELTA:
            l['state'] = alloc(l['value_heads']*l['key_dim']*l['value_dim'])
            convdim = 2*l['key_heads']*l['key_dim']+l['value_heads']*l['value_dim']
            l['conv_state'] = alloc(convdim*(l['kernel']-1))
        else:
            n = context*l['kvheads']*l['head_dim']
            l['keys'] = alloc(n,h['kv_bits']//8)
            l['values'] = alloc(n,h['kv_bits']//8)
        states.append(cursor-begin)
    h['ram_bytes'] = cursor
    if cursor > ARENA_BYTES:
        raise ValueError(f'Guest arena needs {cursor} > {ARENA_BYTES} bytes')
    return dict(arena_used=cursor, arena_capacity=ARENA_BYTES, layer_state_bytes=states)
