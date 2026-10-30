#ifndef TINY_QWEN_RUNTIME_H
#define TINY_QWEN_RUNTIME_H
#include <stdint.h>
#define ARENA_BYTES 63120
#define MAGIC 0x354e5751u
typedef struct {
    uint32_t magic, version, bytes, dim, vocab, context, layers, graph, kv_bits, kv_frac;
    uint32_t norm_eps, final_norm, emb, emb_scale, cls, cls_scale, exp, sigmoid, softplus;
    uint32_t vocab_offsets, vocab_lengths, vocab_bytes, prompt, prompt_len, steps, ram_bytes;
    uint32_t x, xb, xb2, q, tmp, hb, hb2, logits, att, quant, z, alpha, beta, activation_limit;
} Model;
typedef struct {
    uint32_t kind, hidden, heads, kvheads, head_dim, key_heads, value_heads, key_dim, value_dim;
    uint32_t kernel, rotary, rope, inv_sqrt, eps, pre_norm, post_norm, q_norm, k_norm, gated_norm;
    uint32_t conv, dt, A, state, conv_state, keys, values;
    uint32_t q, qs, k, ks, v, vs, o, os, gate, gates, up, ups, down, downs, z, zs, b, bs, a, ascale;
} Layer;
_Static_assert(sizeof(Model) == 40*4, "DROM model ABI");
_Static_assert(sizeof(Layer) == 46*4, "DROM layer ABI");
#endif
