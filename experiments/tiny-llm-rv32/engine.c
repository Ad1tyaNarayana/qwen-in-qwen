/* SPDX-License-Identifier: MIT
 * Transformer equations adapted from llama2.c, Copyright (c) 2023 Andrej.
 * See NOTICE for the upstream license.
 */
#include <stdint.h>
#include "model.h"

static const unsigned char *rom;
static int32_t x[DIM], xb[DIM], xb2[DIM], q[DIM], tmp[DIM];
static int32_t hb[HIDDEN], hb2[HIDDEN], logits[VOCAB], att[CONTEXT];
static int16_t quant[HIDDEN > DIM ? HIDDEN : DIM];
#if KV_BITS == 8
typedef int8_t kv_t;
#else
typedef int16_t kv_t;
#endif
static kv_t keys[LAYERS * CONTEXT * KVDIM], values[LAYERS * CONTEXT * KVDIM];
static uint32_t kv_clips;
#define PTR(type, name) ((const type *)(rom + O_##name))

static uint32_t isqrt(uint64_t n) {
    uint64_t r = 0, b = (uint64_t)1 << 62;
    while (b > n) b >>= 2;
    while (b) {
        if (n >= r + b) { n -= r + b; r = (r >> 1) + b; }
        else r >>= 1;
        b >>= 2;
    }
    return (uint32_t)r;
}

static void norm(int32_t *o, const int32_t *a, const int32_t *w) {
    uint64_t ss = 0;
    for (int i = 0; i < DIM; i++) ss += (int64_t)a[i] * a[i];
    uint32_t root = isqrt(ss / DIM + 168);
    for (int i = 0; i < DIM; i++) o[i] = (int64_t)a[i] * w[i] / root;
}

__attribute__((noinline)) void matvec(int32_t *o, const int32_t *a,
                                      const int8_t *w, const int32_t *sc, int n, int rows) {
    uint32_t peak = 0;
    for (int j = 0; j < n; j++) {
        uint32_t v = a[j] < 0 ? -(int64_t)a[j] : a[j];
        if (v > peak) peak = v;
    }
    int shift = 0;
    uint32_t limit = 2147483647u / (127u * (unsigned)n);
    if (limit > 16383) limit = 16383;
    while (peak > limit) { peak = (peak + 1) >> 1; shift++; }
    for (int j = 0; j < n; j++) quant[j] = a[j] >> shift;
    for (int i = 0; i < rows; i++) {
        int32_t sum = 0;
        const int8_t *row = w + i * n;
        for (int j = 0; j < n; j++) {
            __asm__ volatile(".global dot_loop_begin\ndot_loop_begin:" ::: "memory");
            sum += row[j] * quant[j];
            __asm__ volatile(".global dot_loop_end\ndot_loop_end:" ::: "memory");
        }
        o[i] = ((int64_t)sum * sc[i]) >> (24 - shift);
    }
}

static kv_t cache_value(int32_t v) {
    const int max = (1 << (KV_BITS - 1)) - 1, min = -(1 << (KV_BITS - 1));
    v >>= 12 - KV_FRAC;
    if (v > max) { kv_clips++; return max; }
    if (v < min) { kv_clips++; return min; }
    return v;
}

static void rotate(int32_t *a, int n, int pos) {
    const int16_t *r = PTR(int16_t, rope) + pos * (DIM / HEADS);
    for (int i = 0; i < n; i += 2) {
        int j = i % (DIM / HEADS);
        int32_t u = a[i], v = a[i + 1];
        a[i] = ((int64_t)u * r[j] - (int64_t)v * r[j + 1]) >> 15;
        a[i + 1] = ((int64_t)u * r[j + 1] + (int64_t)v * r[j]) >> 15;
    }
}

#define MV(out, in, name, layer, rows, cols) \
    matvec(out, in, PTR(int8_t, name) + (layer) * (rows) * (cols), \
           PTR(int32_t, name##_scale) + (layer) * (rows), cols, rows)

static int forward(int token, int pos, uint32_t *hash) {
    const int8_t *embedding = PTR(int8_t, emb) + token * DIM;
    int32_t scale = PTR(int32_t, emb_scale)[token];
    for (int i = 0; i < DIM; i++) x[i] = ((int64_t)embedding[i] * scale) >> 12;
    for (int l = 0; l < LAYERS; l++) {
        norm(xb, x, PTR(int32_t, rms_att) + l * DIM);
        MV(q, xb, wq, l, DIM, DIM);
        MV(tmp, xb, wk, l, KVDIM, DIM);
        rotate(q, DIM, pos);
        rotate(tmp, KVDIM, pos);
        int off = (l * CONTEXT + pos) * KVDIM;
        for (int i = 0; i < KVDIM; i++) keys[off + i] = cache_value(tmp[i]);
        MV(tmp, xb, wv, l, KVDIM, DIM);
        for (int i = 0; i < KVDIM; i++) values[off + i] = cache_value(tmp[i]);
        for (int head = 0; head < HEADS; head++) {
            int hs = DIM / HEADS, kh = head / (HEADS / KVHEADS) * hs;
            int32_t max = INT32_MIN;
            for (int t = 0; t <= pos; t++) {
                int64_t sum = 0;
                int ko = (l * CONTEXT + t) * KVDIM + kh;
                for (int j = 0; j < hs; j++) sum += (int64_t)q[head * hs + j] * keys[ko + j];
                att[t] = ((sum >> KV_FRAC) * PTR(int32_t, inv_sqrt_head)[0]) >> 12;
                if (att[t] > max) max = att[t];
            }
            uint32_t total = 0;
            for (int t = 0; t <= pos; t++) {
                int64_t delta = ((int64_t)max - att[t]) >> 5;
                att[t] = PTR(uint32_t, exp)[delta > 2048 ? 2048 : delta];
                total += att[t];
            }
            for (int t = 0; t <= pos; t++) att[t] = ((uint64_t)att[t] << 15) / total;
            for (int j = 0; j < hs; j++) {
                int64_t sum = 0;
                for (int t = 0; t <= pos; t++)
                    sum += (int64_t)att[t] * values[(l * CONTEXT + t) * KVDIM + kh + j];
                xb[head * hs + j] = sum >> (KV_FRAC + 3);
            }
        }
        MV(xb2, xb, wo, l, DIM, DIM);
        for (int i = 0; i < DIM; i++) x[i] += xb2[i];
        norm(xb, x, PTR(int32_t, rms_ffn) + l * DIM);
        MV(hb, xb, w1, l, HIDDEN, DIM);
        MV(hb2, xb, w3, l, HIDDEN, DIM);
        for (int i = 0; i < HIDDEN; i++) {
            int idx = (hb[i] >> 5) + 2048;
            if (idx < 0) idx = 0;
            if (idx > 4096) idx = 4096;
            int32_t silu = ((int64_t)hb[i] * PTR(uint16_t, sigmoid)[idx]) >> 15;
            hb[i] = ((int64_t)silu * hb2[i]) >> 12;
        }
        MV(xb, hb, w2, l, DIM, HIDDEN);
        for (int i = 0; i < DIM; i++) x[i] += xb[i];
    }
    norm(x, x, PTR(int32_t, rms_final));
    MV(logits, x, cls, 0, VOCAB, DIM);
    int best = 0;
    uint32_t hv = 2166136261u;
    for (int i = 0; i < VOCAB; i++) {
        if (logits[i] > logits[best]) best = i;
        hv = (hv ^ (uint32_t)logits[i]) * 16777619u;
    }
    *hash = hv;
    return best;
}

#ifdef NATIVE
#include <stdio.h>
#include <stdlib.h>
static void byte(unsigned c) { putchar(c); }
#else
extern const unsigned char model_blob[];
static void byte(unsigned c) {
    register unsigned a0 __asm__("a0") = c;
    register unsigned a7 __asm__("a7") = 1;
    __asm__ volatile("ecall" : "+r"(a0) : "r"(a7) : "memory");
}
#endif

static void number(uint32_t n) {
    char b[10]; int k = 0;
    do { b[k++] = '0' + n % 10; n /= 10; } while (n);
    while (k) byte(b[--k]);
}

static void decode(int previous, int token) {
    uint32_t n = PTR(uint32_t, vocab_lengths)[token];
    const unsigned char *p = PTR(unsigned char, vocab_bytes) + PTR(uint32_t, vocab_offsets)[token];
    if (previous == 1 && n && *p == ' ') { p++; n--; }
    if (n == 6 && p[0] == '<' && p[1] == '0' && p[2] == 'x' && p[5] == '>') {
        unsigned c = 0;
        for (int i = 3; i < 5; i++) c = c * 16 + (p[i] <= '9' ? p[i] - '0' : (p[i] & ~32) - 'A' + 10);
        byte(c);
    } else for (uint32_t i = 0; i < n; i++) byte(p[i]);
}

#include "workload.h"
__attribute__((noinline)) void position_begin(int pos) { __asm__ volatile("" : : "r"(pos) : "memory"); }
__attribute__((noinline)) void position_end(int pos) { __asm__ volatile("" : : "r"(pos) : "memory"); }

int engine_main(void) {
#ifndef NATIVE
    rom = model_blob;
#endif
    int token = prompt_tokens[0];
    for (int pos = 0; pos < RUN_STEPS; pos++) {
        position_begin(pos);
        uint32_t hash;
        int next = forward(token, pos, &hash);
        position_end(pos);
        byte('@'); number(pos); byte(' '); number(token); byte(' '); number(next);
        byte(' '); number(hash); byte(' '); number(kv_clips); byte('\n');
        if (pos + 1 < PROMPT_LEN) next = prompt_tokens[pos + 1];
        byte('>'); decode(token, next); byte('\n');
        token = next;
    }
    return 0;
}

#ifdef NATIVE
int main(int argc, char **argv) {
    if (argc != 2) return 2;
    FILE *f = fopen(argv[1], "rb");
    unsigned char *b = malloc(MODEL_BYTES);
    if (!f || !b || fread(b, 1, MODEL_BYTES, f) != MODEL_BYTES) return 2;
    fclose(f); rom = b;
    int result = engine_main(); free(b); return result;
}
#endif
