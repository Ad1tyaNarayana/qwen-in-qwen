/* SPDX-License-Identifier: MIT */
#define NATIVE
#define main runtime_native_main
#include "runtime.c"
#undef main

void test_bind(const unsigned char *image) {
    rom = image;
    m = (const Model *)rom;
    for (int i = 0; i < ARENA_BYTES; i++) arena[i] = 0;
    kv_clips = numeric_clips = 0;
}
void test_linear(int layer, const int32_t *input, int32_t *output) {
    const Layer *l = R(Layer,m->graph)+layer;
    for (uint32_t i = 0; i < m->dim; i++) X(xb)[i] = input[i];
    delta_attention(l);
    for (uint32_t i = 0; i < m->dim; i++) output[i] = X(xb2)[i];
}
void test_full(int layer, int position, const int32_t *input, int32_t *output) {
    const Layer *l = R(Layer,m->graph)+layer;
    for (uint32_t i = 0; i < m->dim; i++) X(xb)[i] = input[i];
    attention(l,position);
    for (uint32_t i = 0; i < m->dim; i++) output[i] = X(xb2)[i];
}
void test_l2(int32_t *input, int n) { l2norm(input,n,17); }
void test_rms(int32_t *output, const int32_t *input, int n, uint32_t weights, uint32_t eps) {
    norm(output,input,weights,n,eps);
}
uint32_t test_clips(void) { return numeric_clips; }
void test_copy(uint32_t offset, int32_t *output, int n) {
    for (int i = 0; i < n; i++) output[i] = B(offset)[i];
}
