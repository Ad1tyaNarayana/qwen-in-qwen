/* SPDX-License-Identifier: MIT */
#include "game.h"

typedef unsigned char u8;
typedef unsigned short u16;
static u8 ram[4096], v[16], pixels[256];
static u16 stack[16], pc = 0x200, index_reg;
static unsigned sp, keys;

static void putbyte(unsigned value) {
    register unsigned a0 __asm__("a0") = value;
    register unsigned a7 __asm__("a7") = 1;
    __asm__ volatile("ecall" : "+r"(a0) : "r"(a7) : "memory");
}

static void frame(unsigned id) {
    unsigned count = 0;
    for (unsigned i = 0; i < 256; i++) count += pixels[i] != 0;
    putbyte('Q'); putbyte('F'); putbyte(id); putbyte(count); putbyte(v[9]); putbyte(v[10]);
    for (unsigned i = 0; i < 256; i++) {
        if (pixels[i]) { putbyte(i); putbyte(pixels[i]); }
    }
}

static int step(void) {
    unsigned op = ram[pc] * 256u + ram[pc + 1];
    unsigned x = op >> 8 & 15, y = op >> 4 & 15, n = op & 15, nn = op & 255;
    pc += 2;
    switch (op >> 12) {
    case 0:
        if (op == 0xe0) for (unsigned i = 0; i < 256; i++) pixels[i] = 0;
        else if (op == 0xee) pc = stack[--sp];
        else return 1;
        break;
    case 1: pc = op & 4095; break;
    case 2: stack[sp++] = pc; pc = op & 4095; break;
    case 3: if (v[x] == nn) pc += 2; break;
    case 4: if (v[x] != nn) pc += 2; break;
    case 5: if (v[x] == v[y]) pc += 2; break;
    case 6: v[x] = nn; break;
    case 7: v[x] += nn; break;
    case 8: {
        unsigned a = v[x], b = v[y], result;
        switch (n) {
        case 0: v[x] = b; break;
        case 1: v[x] |= b; break;
        case 2: v[x] &= b; break;
        case 3: v[x] ^= b; break;
        case 4: result = a + b; v[x] = result; v[15] = result > 255; break;
        case 5: v[x] = a - b; v[15] = a >= b; break;
        case 6: v[x] = a >> 1; v[15] = a & 1; break;
        case 7: v[x] = b - a; v[15] = b >= a; break;
        case 14: v[x] = a << 1; v[15] = a >> 7; break;
        default: return 2;
        }
        break;
    }
    case 9: if (v[x] != v[y]) pc += 2; break;
    case 10: index_reg = op & 4095; break;
    case 11: pc = (op & 4095) + v[0]; break;
    case 13: {
        unsigned px = v[x] & 63, py = v[y] & 31, collision = 0;
        for (unsigned row = 0; row < n && py + row < 32; row++) {
            unsigned sprite = ram[index_reg + row];
            for (unsigned bit = 0; bit < 8 && px + bit < 64; bit++) {
                if (!(sprite & (128 >> bit))) continue;
                unsigned address = (py + row) * 8 + (px + bit) / 8;
                unsigned mask = 128 >> ((px + bit) & 7);
                collision |= pixels[address] & mask;
                pixels[address] ^= mask;
            }
        }
        v[15] = collision != 0;
        break;
    }
    case 14:
        if (nn == 0x9e) { if (keys & (1u << v[x])) pc += 2; }
        else if (nn == 0xa1) { if (!(keys & (1u << v[x]))) pc += 2; }
        else return 3;
        break;
    case 15:
        if (nn == 0x1e) index_reg += v[x];
        else if (nn == 0x55) for (unsigned j = 0; j <= x; j++) ram[index_reg + j] = v[j];
        else if (nn == 0x65) for (unsigned j = 0; j <= x; j++) v[j] = ram[index_reg + j];
        else return 4;
        break;
    default: return 5;
    }
    return 0;
}

int main(void) {
    for (unsigned i = 0; i < sizeof(game_rom); i++) ram[0x200 + i] = game_rom[i];
    for (unsigned f = 0; f < FRAME_COUNT; f++) {
        keys = game_keys[f];
        unsigned budget = 96;
        do {
            if (step()) return 10;
            if (!--budget) return 11;
        } while (pc != GAME_LOOP);
        frame(f);
    }
    return 0;
}
