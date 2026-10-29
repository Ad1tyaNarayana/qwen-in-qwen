"""Isolated reference-machine extension. The independent oracle is not imported."""
from chipformer.rv import ref, spec as S

CUSTOM = dict(ref.CUSTOM, NDOT4=70, NDOT8=71)
U = dict(ref.U, NDOT4=70, NDOT8=71)
UOPS = ref.UOPS + ['NDOT4', 'NDOT8']
WRITES_RD = ref.WRITES_RD | {'NDOT4', 'NDOT8'}


def __getattr__(name):
    return getattr(ref, name)


def decode(word, maximum=8):
    if word & 127 == 0x0b and word >> 25 == 0x20:
        width = {0: 4, 1: 8}.get(word >> 12 & 7)
        if width is not None and width <= maximum:
            return ref.Inst(f'NDOT{width}', word >> 7 & 31, word >> 15 & 31, word >> 20 & 31)
        return ref.Inst('ILLEGAL')
    return ref.decode(word)


def rom_word(word, pc, maximum=8):
    ins = decode(word, maximum)
    if ins.op.startswith('NDOT'):
        return U[ins.op] | ins.rd << 11 | ins.rs1 << 19 | ins.rs2 << 27
    return ref.rom_word(word, pc)


class Machine(ref.Machine):
    maximum = 8

    def record(self):
        if self.n == S.N_MAX or self.halted:
            return super().record()
        word = self.rom[(self.pc - S.ROM_BASE) // 4]
        ins = decode(word, self.maximum)
        if not ins.op.startswith('NDOT'):
            return super().record()
        width = int(ins.op[4:])
        a, b, accumulator = self.x[ins.rs1], self.x[ins.rs2], self.x[ins.rd]
        self.steps += 1
        cause = ('misaligned load' if a & (width-1) or b & (2*width-1) else
                 'load out of range' if not (
                     S.DROM_BASE <= a <= S.DROM_BASE + 4*S.DROM_WORDS - width and
                     S.RAM_BASE <= b <= S.RAM_BASE + 4*S.RAM_WORDS - 2*width) else None)
        if cause:
            self.halted = True
            self.fault = (S.FAULTS.index(cause), self.pc)
            return S.fault_record(*self.fault)
        for i in range(width):
            w = self.drom_word(a+i) >> (8*((a+i) & 3)) & 255
            v = self.ram[(b-S.RAM_BASE+2*i)//4] >> (8*((b+2*i) & 3)) & 65535
            accumulator += ref.sx(w, 8) * ref.sx(v, 16)
        self.n += 1
        if ins.rd:
            self.x[ins.rd] = accumulator & 0xffffffff
            return S.value_record(accumulator)
        return [S.tok('NT')]
