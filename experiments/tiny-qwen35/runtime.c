/* SPDX-License-Identifier: MIT
 * Llama integer arithmetic adapted from experiments/tiny-llm-rv32/engine.c.
 * See ../tiny-llm-rv32/NOTICE for the original llama2.c attribution.
 */
#include "runtime.h"
#ifndef NATIVE
#include "ndot.h"
#endif

static const unsigned char *rom;
static const Model *m;
static _Alignas(8) unsigned char arena[ARENA_BYTES];
static uint32_t kv_clips, numeric_clips;
#define R(type, offset) ((const type *)(rom + (offset)))
#define B(offset) ((int32_t *)(arena + (offset)))
#define X(name) B(m->name)
#define LLAMA 1
#define QWEN_FULL 2
#define QWEN_DELTA 3
#define BOUND (1 << 23)

static int32_t bounded(int64_t v) {
    if (v > BOUND) { numeric_clips++; return BOUND; }
    if (v < -BOUND) { numeric_clips++; return -BOUND; }
    return v;
}

static int32_t bounded_high(int64_t v) {
    if (v > (1 << 26)) { numeric_clips++; return 1 << 26; }
    if (v < -(1 << 26)) { numeric_clips++; return -(1 << 26); }
    return v;
}

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

static void norm(int32_t *out, const int32_t *in, uint32_t weights, int n, uint32_t eps) {
    uint64_t ss = 0;
    for (int i = 0; i < n; i++) ss += (int64_t)in[i]*in[i];
    uint32_t root = isqrt(ss / n + eps);
    if (!root) root = 1;
    for (int i = 0; i < n; i++) {
        int64_t value = (int64_t)in[i]*R(int32_t,weights)[i]/root;
        out[i] = m->activation_limit ? bounded(value) : (int32_t)value;
    }
}

__attribute__((noinline)) void matvec(int32_t *out, const int32_t *in,
                                    uint32_t weights, uint32_t scales, int n, int rows) {
    uint32_t peak = 0;
    int16_t *quant = (int16_t *)(arena + m->quant);
    for (int j = 0; j < n; j++) {
        uint32_t v = in[j] < 0 ? -(int64_t)in[j] : in[j];
        if (v > peak) peak = v;
    }
    int shift = 0;
    uint32_t limit = 2147483647u/(127u*(unsigned)n);
    if (limit > 16383) limit = 16383;
    while (peak > limit) { peak = (peak+1) >> 1; shift++; }
    for (int j = 0; j < n; j++) quant[j] = in[j] >> shift;
    for (int i = 0; i < rows; i++) {
        const int8_t *row = R(int8_t,weights) + i*n;
        uint32_t acc = 0;
        int j = 0;
        if (((uintptr_t)row & 3) == 0) {
            for (; j+4 <= n; j += 4) {
#ifdef NATIVE
                for (int lane = 0; lane < 4; lane++)
                    acc += (uint32_t)((int32_t)row[j+lane]*quant[j+lane]);
#else
                acc = ndot4(acc, row+j, quant+j);
#endif
            }
        }
        for (; j < n; j++) acc += (uint32_t)((int32_t)row[j]*quant[j]);
        int64_t product = (int64_t)(int32_t)acc*R(int32_t,scales)[i];
        int64_t value = product >> (24-shift);
        out[i] = m->activation_limit ? bounded(value) : (int32_t)value;
    }
}

#define MV(out,in,name,rows,cols) matvec(out,in,l->name,l->name##s,cols,rows)
static int lut_index(int32_t a) {
    int idx = (a >> 5)+2048;
    return idx < 0 ? 0 : idx > 4096 ? 4096 : idx;
}
static int32_t sigmoid(int32_t a) { return R(uint16_t,m->sigmoid)[lut_index(a)]; }
static int32_t silu(int32_t a) { return ((int64_t)a*sigmoid(a)) >> 15; }

static int32_t cache_value(int32_t v) {
    int max = (1 << (m->kv_bits-1))-1, min = -(1 << (m->kv_bits-1));
    v >>= 12-m->kv_frac;
    if (v > max) { kv_clips++; return max; }
    if (v < min) { kv_clips++; return min; }
    return v;
}
static void store_cache(uint32_t offset, int i, int32_t v) {
    v = cache_value(v);
    if (m->kv_bits == 8) ((int8_t *)(arena+offset))[i] = v;
    else ((int16_t *)(arena+offset))[i] = v;
}
static int32_t read_cache(uint32_t offset, int i) {
    if (m->kv_bits == 8) return ((int8_t *)(arena+offset))[i];
    return ((int16_t *)(arena+offset))[i];
}

static void rotate(int32_t *a, int n, int pos, const Layer *l) {
    const int16_t *r = R(int16_t,l->rope)+pos*l->rotary;
    for (int head = 0; head < n; head += l->head_dim) {
        for (uint32_t j = 0; j < l->rotary/2; j++) {
            int uidx = head+(l->kind == LLAMA ? 2*j : j);
            int vidx = head+(l->kind == LLAMA ? 2*j+1 : j+l->rotary/2);
            int32_t u = a[uidx], v = a[vidx], c = r[2*j], s = r[2*j+1];
            a[uidx] = ((int64_t)u*c-(int64_t)v*s) >> 15;
            a[vidx] = ((int64_t)u*s+(int64_t)v*c) >> 15;
        }
    }
}

static void attention(const Layer *l, int pos) {
    int d = m->dim, hs = l->head_dim, qdim = l->heads*hs, kd = l->kvheads*hs;
    int32_t *xb = X(xb), *q = X(q), *tmp = X(tmp), *z = X(z), *att = X(att);
    MV(q,xb,q,l->kind == LLAMA ? qdim : 2*qdim,d);
    if (l->kind == QWEN_FULL) {
        for (uint32_t h = 0; h < l->heads; h++) {
            for (int j = 0; j < hs; j++) {
                z[h*hs+j] = q[2*h*hs+hs+j];
                q[h*hs+j] = q[2*h*hs+j];
            }
            norm(q+h*hs,q+h*hs,l->q_norm,hs,l->eps);
        }
    }
    MV(tmp,xb,k,kd,d);
    if (l->kind == QWEN_FULL)
        for (int h = 0; h < kd; h += hs) norm(tmp+h,tmp+h,l->k_norm,hs,l->eps);
    rotate(q,qdim,pos,l);
    rotate(tmp,kd,pos,l);
    for (int j = 0; j < kd; j++) store_cache(l->keys,pos*kd+j,tmp[j]);
    MV(tmp,xb,v,kd,d);
    for (int j = 0; j < kd; j++) store_cache(l->values,pos*kd+j,tmp[j]);
    for (uint32_t head = 0; head < l->heads; head++) {
        int kh = head/(l->heads/l->kvheads)*hs;
        int32_t max = INT32_MIN;
        for (int t = 0; t <= pos; t++) {
            int64_t sum = 0;
            for (int j = 0; j < hs; j++) sum += (int64_t)q[head*hs+j]*read_cache(l->keys,t*kd+kh+j);
            int64_t score = ((sum >> m->kv_frac)*l->inv_sqrt) >> 12;
            att[t] = m->activation_limit ? bounded(score) : (int32_t)score;
            if (att[t] > max) max = att[t];
        }
        uint32_t total = 0;
        for (int t = 0; t <= pos; t++) {
            int shift = m->activation_limit ? 2 : 5;
            int limit = m->activation_limit ? 16384 : 2048;
            int64_t delta = ((int64_t)max-att[t]) >> shift;
            att[t] = R(uint32_t,m->exp)[delta > limit ? limit : delta];
            total += att[t];
        }
        for (int t = 0; t <= pos; t++) att[t] = ((uint64_t)att[t] << 15)/total;
        for (int j = 0; j < hs; j++) {
            int64_t sum = 0;
            for (int t = 0; t <= pos; t++) sum += (int64_t)att[t]*read_cache(l->values,t*kd+kh+j);
            xb[head*hs+j] = sum >> (m->kv_frac+3);
            if (l->kind == QWEN_FULL) xb[head*hs+j] = ((int64_t)xb[head*hs+j]*sigmoid(z[head*hs+j])) >> 15;
        }
    }
    MV(X(xb2),xb,o,d,qdim);
}

static void l2norm(int32_t *a, int n, uint32_t eps) {
    uint64_t ss = eps;
    for (int j = 0; j < n; j++) ss += (int64_t)a[j]*a[j];
    uint32_t root = isqrt(ss);
    for (int j = 0; j < n; j++) a[j] = (int64_t)a[j]*32768/root;
}

static void delta_attention(const Layer *l) {
    int d = m->dim, kd = l->key_heads*l->key_dim, vd = l->value_heads*l->value_dim;
    int channels = 2*kd+vd, kernel = l->kernel;
    int32_t *qkv = X(q), *z = X(z), *alpha = X(alpha), *beta = X(beta), *out = X(tmp);
    int32_t *state = B(l->state), *history = B(l->conv_state);
    MV(qkv,X(xb),q,channels,d);
    MV(z,X(xb),z,vd,d);
    matvec(alpha,X(xb),l->a,l->ascale,d,l->value_heads);
    MV(beta,X(xb),b,l->value_heads,d);
    for (int c = 0; c < channels; c++) {
        int32_t input = bounded(qkv[c]);
        int64_t sum = (int64_t)input*R(int32_t,l->conv)[c*kernel+kernel-1];
        for (int j = 0; j < kernel-1; j++) {
            sum += (int64_t)history[c*(kernel-1)+j]*R(int32_t,l->conv)[c*kernel+j];
            history[c*(kernel-1)+j] = j+1 == kernel-1 ? input : history[c*(kernel-1)+j+1];
        }
        int32_t high = bounded_high(sum >> 4);
        qkv[c] = ((int64_t)high*sigmoid(high >> 8)) >> 15;
    }
    for (uint32_t h = 0; h < l->key_heads; h++) {
        l2norm(qkv+h*l->key_dim,l->key_dim,1099512);
        l2norm(qkv+kd+h*l->key_dim,l->key_dim,1099512);
    }
    for (uint32_t h = 0; h < l->value_heads; h++) {
        int kh = h/(l->value_heads/l->key_heads)*l->key_dim;
        int32_t *q = qkv+kh, *k = qkv+kd+kh, *v = qkv+2*kd+h*l->value_dim;
        int32_t *s = state+h*l->key_dim*l->value_dim;
        int64_t dt = (int64_t)alpha[h]+R(int32_t,l->dt)[h];
        int64_t sp = R(int32_t,m->softplus)[lut_index(bounded(dt))];
        if (dt > 65536) sp = (int64_t)bounded(dt)*256;
        int64_t decay = ((int64_t)R(int32_t,l->A)[h]*sp) >> 20;
        int64_t idx = decay >> 10;
        int32_t decay_q20 = R(uint32_t,m->exp)[idx > 16384 ? 16384 : idx];
        int32_t b = sigmoid(beta[h]);
        for (uint32_t i = 0; i < l->key_dim*l->value_dim; i++) s[i] = ((int64_t)s[i]*decay_q20) >> 20;
        for (uint32_t j = 0; j < l->value_dim; j++) {
            int64_t mem = 0;
            for (uint32_t i = 0; i < l->key_dim; i++) mem += (int64_t)s[i*l->value_dim+j]*k[i];
            int32_t correction = bounded_high(((int64_t)v[j]-(mem >> 15))*b >> 15);
            for (uint32_t i = 0; i < l->key_dim; i++)
                s[i*l->value_dim+j] = bounded_high((int64_t)s[i*l->value_dim+j]+(((int64_t)k[i]*correction) >> 15));
            int64_t value = 0;
            for (uint32_t i = 0; i < l->key_dim; i++) value += (int64_t)s[i*l->value_dim+j]*q[i];
            out[h*l->value_dim+j] = bounded_high(((value >> 15)*l->inv_sqrt) >> 12);
        }
        norm(out+h*l->value_dim,out+h*l->value_dim,l->gated_norm,l->value_dim,l->eps << 16);
        for (uint32_t j = 0; j < l->value_dim; j++) {
            int at = h*l->value_dim+j;
            out[at] = bounded(((int64_t)out[at]*silu(z[at])) >> 12);
        }
    }
    MV(X(xb2),out,o,d,vd);
}

static int forward(int token, int pos, uint32_t *hash) {
    int d = m->dim;
    int32_t *x = X(x), *xb = X(xb), *hb = X(hb), *hb2 = X(hb2), *logits = X(logits);
    const int8_t *embedding = R(int8_t,m->emb)+token*d;
    int32_t scale = R(int32_t,m->emb_scale)[token];
    for (int i = 0; i < d; i++) {
        int64_t value = ((int64_t)embedding[i]*scale) >> 12;
        x[i] = m->activation_limit ? bounded(value) : (int32_t)value;
    }
    const Layer *graph = R(Layer,m->graph);
    for (uint32_t i = 0; i < m->layers; i++) {
        const Layer *l = graph+i;
        norm(xb,x,l->pre_norm,d,l->eps);
        if (l->kind == QWEN_DELTA) delta_attention(l);
        else attention(l,pos);
        for (int j = 0; j < d; j++) {
            int64_t v = (int64_t)x[j]+X(xb2)[j];
            x[j] = l->kind == LLAMA ? (int32_t)v : bounded(v);
        }
        norm(xb,x,l->post_norm,d,l->eps);
        MV(hb,xb,gate,l->hidden,d);
        MV(hb2,xb,up,l->hidden,d);
        for (uint32_t j = 0; j < l->hidden; j++) {
            int64_t v = ((int64_t)silu(hb[j])*hb2[j]) >> 12;
            hb[j] = l->kind == LLAMA ? (int32_t)v : bounded(v);
        }
        MV(xb,hb,down,d,l->hidden);
        for (int j = 0; j < d; j++) {
            int64_t v = (int64_t)x[j]+xb[j];
            x[j] = l->kind == LLAMA ? (int32_t)v : bounded(v);
        }
    }
    norm(x,x,m->final_norm,d,m->norm_eps);
    matvec(logits,x,m->cls,m->cls_scale,d,m->vocab);
    int best = 0;
    uint32_t hv = 2166136261u;
    for (uint32_t i = 0; i < m->vocab; i++) {
        if (logits[i] > logits[best]) best = i;
        hv = (hv^(uint32_t)logits[i])*16777619u;
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
    do { b[k++] = '0'+n%10; n /= 10; } while (n);
    while (k) byte(b[--k]);
}
static void decode(int previous, int token) {
    uint32_t n = R(uint32_t,m->vocab_lengths)[token];
    const unsigned char *p = R(unsigned char,m->vocab_bytes)+R(uint32_t,m->vocab_offsets)[token];
    if (previous == 1 && n && *p == ' ') { p++; n--; }
    if (n == 6 && p[0] == '<' && p[1] == '0' && p[2] == 'x' && p[5] == '>') {
        unsigned c = 0;
        for (int i = 3; i < 5; i++) c = c*16+(p[i] <= '9' ? p[i]-'0' : (p[i]&~32)-'A'+10);
        byte(c);
    } else for (uint32_t i = 0; i < n; i++) byte(p[i]);
}
__attribute__((noinline)) void position_begin(int pos) { __asm__ volatile("" : : "r"(pos) : "memory"); }
__attribute__((noinline)) void position_end(int pos) { __asm__ volatile("" : : "r"(pos) : "memory"); }
int engine_main(void) {
#ifndef NATIVE
    rom = model_blob;
#endif
    m = (const Model *)rom;
    if (m->magic != MAGIC || m->version != 1 || m->ram_bytes > ARENA_BYTES ||
        m->steps > m->context || !m->prompt_len || m->prompt_len > m->steps ||
        (m->activation_limit && m->activation_limit != BOUND)) return 2;
    int token = R(uint32_t,m->prompt)[0];
    for (uint32_t pos = 0; pos < m->steps; pos++) {
        position_begin(pos);
        uint32_t hash;
        int next = forward(token,pos,&hash);
        position_end(pos);
        byte('@'); number(pos); byte(' '); number(token); byte(' '); number(next);
        byte(' '); number(hash); byte(' '); number(kv_clips); byte('\n');
        if (pos+1 < m->prompt_len) next = R(uint32_t,m->prompt)[pos+1];
        byte('>'); decode(token,next); byte('\n');
        token = next;
    }
    if (numeric_clips) { byte('!'); number(numeric_clips); byte('\n'); }
    return 0;
}
#ifdef NATIVE
int main(int argc, char **argv) {
    if (argc != 2) return 2;
    FILE *f = fopen(argv[1],"rb");
    if (!f) return 2;
    unsigned char *data = malloc(4194304);
    if (!data) return 2;
    size_t size = fread(data,1,4194304,f);
    fclose(f);
    if (size < sizeof(Model)) return 2;
    rom = data;
    if (((const Model *)data)->bytes != size) return 2;
    int result = engine_main();
    free(data);
    return result;
}
#endif
