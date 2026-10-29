"""Independent byte-array ISA oracle. No compiler or reference imports."""
from dataclasses import dataclass

RAM_BASE, RAM_SIZE = 0x01000000, 65536
DROM_BASE, DROM_SIZE = 0x02000000, 4194304


@dataclass(frozen=True)
class Outcome:
    regs: tuple[int, ...]
    fault: int | None
    pc_delta: int


def execute(word, old_regs, ram, drom, maximum=8):
    regs = tuple(old_regs)
    fields = ((word >> 25) & 127, (word >> 12) & 7)
    if word & 127 != 11 or fields[0] != 32 or fields[1] not in (0, 1):
        return Outcome(regs, 0, 0)
    n = 4 << fields[1]
    if n > maximum:
        return Outcome(regs, 0, 0)
    rd, r1, r2 = (word >> 7) & 31, (word >> 15) & 31, (word >> 20) & 31
    a, b = regs[r1], regs[r2]
    if a % n or b % (2 * n):
        return Outcome(regs, 2, 0)
    if not (DROM_BASE <= a and a + n <= DROM_BASE + DROM_SIZE
            and RAM_BASE <= b and b + 2*n <= RAM_BASE + RAM_SIZE):
        return Outcome(regs, 4, 0)
    total = regs[rd]
    for lane in range(n):
        wi = a - DROM_BASE + lane
        ai = b - RAM_BASE + 2*lane
        w = drom[wi] if wi < len(drom) else 0
        lo = ram[ai] if ai < len(ram) else 0
        hi = ram[ai+1] if ai+1 < len(ram) else 0
        activation = lo + 256*hi
        total += (w if w < 128 else w - 256) * (
            activation if activation < 32768 else activation - 65536)
    result = list(regs)
    if rd:
        result[rd] = total % (1 << 32)
    return Outcome(tuple(result), None, 4)


def serialize(outcome, word, pc):
    """Encode numeric chip token IDs independently of spec.value_record."""
    if outcome.fault is not None:
        return [229892 + outcome.fault] + [229900 + ((pc >> s) & 15)
                                          for s in range(28, -1, -4)] + [230321]
    rd = (word >> 7) & 31
    if not rd:
        return [230316]
    v = outcome.regs[rd]
    signed = v if v < 2147483648 else v - 4294967296
    if -16384 <= signed < 16384:
        return [32768 + (v & 32767)]
    if -536870912 <= signed < 536870912:
        return [v & 32767, 65536 + ((v >> 15) & 32767)]
    return [v & 32767, (v >> 15) & 32767, 98304 + (v >> 30)]
