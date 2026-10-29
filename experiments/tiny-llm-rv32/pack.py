#!/usr/bin/env python3
"""Convert the legacy llama2.c checkpoint and tokenizer to a bounded ROM blob."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

RAM_MARGIN = 1024

def sha(b):
    return hashlib.sha256(b).hexdigest()


def pack(model, tokenizer, out, context, kv_bits=16, kv_frac=10, stack_bytes=1024):
    if kv_bits not in (8, 16) or not 0 <= kv_frac <= 12 or stack_bytes < 512 or stack_bytes % 16:
        raise ValueError('Invalid KV format or stack reservation')
    raw, tok = model.read_bytes(), tokenizer.read_bytes()
    d, h, layers, heads, kvheads, signed_vocab, seq = struct.unpack_from('<7i', raw)
    vocab = abs(signed_vocab)
    if not (0 < d <= 1024 and 0 < h <= 4096 and 0 < layers <= 32 and
            0 < heads <= d and 0 < kvheads <= heads and d % heads == 0 and
            heads % kvheads == 0 and d // heads % 2 == 0 and
            259 <= vocab <= 8192 and 0 < context <= min(seq, 512)):
        raise ValueError('Unsupported checkpoint geometry/context')
    kd = d * kvheads // heads
    blob = bytearray()
    offsets = {}

    def put(name, fmt, values):
        blob.extend(bytes(-len(blob) % 4))
        offsets[name] = len(blob)
        blob.extend(struct.pack('<' + str(len(values)) + fmt, *values))

    cursor = 28

    def take(n):
        nonlocal cursor
        values = struct.unpack_from('<' + str(n) + 'f', raw, cursor)
        cursor += n * 4
        if not all(math.isfinite(v) for v in values):
            raise ValueError('Nonfinite weights')
        return values

    errors = {}

    def matrix(name, rows, cols):
        values = take(rows * cols)
        q, scales = [], []
        err = 0.0
        for r in range(rows):
            row = values[r * cols:(r + 1) * cols]
            scale = max(abs(v) for v in row) / 127 or 1 / (1 << 24)
            s = max(1, round(scale * (1 << 24)))
            if s >= 1 << 31:
                raise ValueError('Scale exceeds signed Q24')
            scales.append(s)
            for v in row:
                vq = max(-127, min(127, round(v / (s / (1 << 24)))))
                q.append(vq)
                err += (v - vq * s / (1 << 24)) ** 2
        put(name, 'b', q)
        put(name + '_scale', 'i', scales)
        errors[name] = math.sqrt(err / len(values))

    def norm(name, n):
        values = [round(v * 4096) for v in take(n)]
        if any(abs(v) >= 1 << 31 for v in values):
            raise ValueError('Norm scale overflow')
        put(name, 'i', values)

    matrix('emb', vocab, d)
    norm('rms_att', layers * d)
    for name, rows, cols in [('wq', layers * d, d), ('wk', layers * kd, d),
                             ('wv', layers * kd, d), ('wo', layers * d, d)]:
        matrix(name, rows, cols)
    norm('rms_ffn', layers * d)
    matrix('w1', layers * h, d)
    matrix('w2', layers * d, h)
    matrix('w3', layers * h, d)
    norm('rms_final', d)
    take(seq * (d // heads))
    if signed_vocab < 0:
        matrix('cls', vocab, d)
    else:
        offsets['cls'], offsets['cls_scale'] = offsets['emb'], offsets['emb_scale']
    if cursor != len(raw):
        raise ValueError(f'Checkpoint size mismatch {cursor} != {len(raw)}')
    rope = []
    for pos in range(context):
        for i in range(0, d // heads, 2):
            a = pos / 10000 ** (i / (d // heads))
            rope += [round(math.cos(a) * 32767), round(math.sin(a) * 32767)]
    put('rope', 'h', rope)
    put('exp', 'I', [round(math.exp(-i / 128) * (1 << 20)) for i in range(2049)])
    put('sigmoid', 'H', [round(32768 / (1 + math.exp(-(i - 2048) / 128))) for i in range(4097)])
    put('inv_sqrt_head', 'i', [round(4096 / math.sqrt(d // heads))])
    tc = 4
    pieces, scores, lengths, starts = bytearray(), [], [], []
    for i in range(vocab):
        score, n = struct.unpack_from('<fi', tok, tc)
        tc += 8
        if not math.isfinite(score) or not 0 <= n <= 4096 or tc + n > len(tok):
            raise ValueError('Invalid tokenizer')
        piece = tok[tc:tc + n]
        tc += n
        starts.append(len(pieces))
        lengths.append(n)
        pieces.extend(piece)
        scores.append(score)
    if tc != len(tok):
        raise ValueError('Tokenizer size mismatch')
    put('vocab_offsets', 'I', starts)
    put('vocab_lengths', 'I', lengths)
    put('vocab_bytes', 'B', list(pieces))
    scratch = 4 * (5 * d + 2 * h + vocab + context) + 2 * max(d, h)
    kvbytes = 2 * (kv_bits // 8) * layers * context * kd
    if scratch + kvbytes + stack_bytes + 256 + RAM_MARGIN > 65536:
        raise ValueError(f'RAM needs {scratch + kvbytes + stack_bytes + 256 + RAM_MARGIN} bytes with margin; reduce --context')
    if len(blob) + 16384 > 4 * 1024 * 1024:
        raise ValueError('Data ROM exceeds capacity')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'model.bin').write_bytes(blob)
    header = ['#pragma once', '#include <stdint.h>']
    for name, value in dict(DIM=d, HIDDEN=h, LAYERS=layers, HEADS=heads,
                            KVHEADS=kvheads, VOCAB=vocab, CONTEXT=context,
                            KVDIM=kd, KV_BITS=kv_bits, KV_FRAC=kv_frac, MODEL_BYTES=len(blob)).items():
        header.append(f'#define {name} {value}')
    for name, offset in offsets.items():
        header.append(f'#define O_{name} {offset}')
    (out / 'model.h').write_text('\n'.join(header) + '\n')
    meta = dict(dim=d, hidden=h, layers=layers, heads=heads, kvheads=kvheads,
                vocab=vocab, context=context, source_seq_len=seq, kv_bytes=kvbytes,
                scratch_bytes=scratch, stack_reserved=stack_bytes, kv_bits=kv_bits,
                kv_fractional_bits=kv_frac, minimum_ram_margin=RAM_MARGIN, rom_bytes=len(blob),
                input_sha256=sha(raw), tokenizer_sha256=sha(tok), rom_sha256=sha(blob),
                weight_rmse=errors, offsets=offsets)
    (out / 'pack.json').write_text(json.dumps(meta, indent=2) + '\n')
    (out / 'tokenizer.json').write_text(json.dumps(dict(scores=scores,
        pieces=[list(pieces[s:s+n]) for s, n in zip(starts, lengths)])))
    return meta


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--context', type=int, default=64)
    p.add_argument('--kv-bits', type=int, choices=[8, 16], default=16)
    p.add_argument('--kv-frac', type=int, default=10)
    p.add_argument('--stack-bytes', type=int, default=1024)
    a = p.parse_args()
    print(json.dumps(pack(a.model, a.tokenizer, a.out, a.context, a.kv_bits, a.kv_frac, a.stack_bytes), indent=2))
