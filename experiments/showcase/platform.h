#ifndef SHOWCASE_PLATFORM_H
#define SHOWCASE_PLATFORM_H
#include <stdint.h>
#ifdef NATIVE
#include <stdio.h>
static void byte(unsigned c) { putchar((int)c); }
#else
#include "ndot.h"
static void byte(unsigned c) {
 register unsigned a0 __asm__("a0") = c;
 register unsigned a7 __asm__("a7") = 1;
 __asm__ volatile("ecall" : "+r"(a0) : "r"(a7) : "memory");
}
#endif
static void text(const char *s) { while (*s) byte((unsigned char)*s++); }
static void number(unsigned n) { if(n>=10) number(n/10); byte('0'+n%10); }
#ifdef NATIVE
int engine_main(void);
int main(void) { return engine_main(); }
#endif
#endif
