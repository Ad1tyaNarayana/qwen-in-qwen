#ifndef QWENPUTER_NDOT_H
#define QWENPUTER_NDOT_H
#include <stdint.h>
static inline uint32_t ndot4(uint32_t acc, const int8_t *w, const int16_t *a) {
    __asm__ volatile (".insn r 0x0b, 0, 0x20, %0, %1, %2"
                      : "+&r"(acc) : "r"(w), "r"(a) : "memory");
    return acc;
}
static inline uint32_t ndot8(uint32_t acc, const int8_t *w, const int16_t *a) {
    __asm__ volatile (".insn r 0x0b, 1, 0x20, %0, %1, %2"
                      : "+&r"(acc) : "r"(w), "r"(a) : "memory");
    return acc;
}
#endif
