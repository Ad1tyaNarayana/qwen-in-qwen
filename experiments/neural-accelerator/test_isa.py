"""Independent directed and seeded ISA checks, including raw record IDs."""
import hashlib
import json
from pathlib import Path
import random
import struct
import tempfile

from prepare import ROOT, prepare


def encoding(n=8, rd=3, r1=1, r2=2):
    return 0x4000000b | ({4: 0, 8: 1, 16: 2}[n] << 12) | rd << 7 | r1 << 15 | r2 << 20


def main():
    scratch = Path(tempfile.mkdtemp(prefix='isa-', dir=ROOT / '.cache/neural-accelerator'))
    manifest = prepare(scratch)
    from chipformer.rv import ref, spec as S
    import reference as R
    import oracle as O
    rng = random.Random(20261029)
    ram = bytearray(rng.randbytes(65536))
    drom = bytearray(rng.randbytes(4194304))
    counts = {}

    def check(word, regs, maximum=8, category='directed'):
        before_ram = hashlib.sha256(ram).digest()
        expected = O.execute(word, regs, ram, drom, maximum)
        machine = R.Machine([word], list(struct.unpack('<16384I', ram)), list(regs), 0, drom=drom)
        machine.maximum = maximum
        tokens = machine.record()
        assert tuple(machine.x) == expected.regs, (category, word, regs, machine.x, expected)
        assert machine.fault == (None if expected.fault is None else (expected.fault, S.ROM_BASE))
        assert machine.pc == S.ROM_BASE + expected.pc_delta
        assert tokens == O.serialize(expected, word, S.ROM_BASE), (tokens, O.serialize(expected, word, S.ROM_BASE))
        assert machine.steps == 1
        assert hashlib.sha256(struct.pack('<16384I', *machine.ram)).digest() == before_ram
        counts[category] = counts.get(category, 0) + 1
        return machine

    base = [0]*32
    base[1:4] = [S.DROM_BASE, S.RAM_BASE, 0xffffffff]
    for f7 in range(1, 128):
        for f3 in range(8):
            word = (f7 << 25) | (f3 << 12) | (encoding() & 0x1fff8fff & ~(127 << 25) & ~(7 << 12))
            check(word, base, category='reserved_encodings')
    for f3 in range(5):
        word = 11 | f3 << 12
        assert R.decode(word) == ref.decode(word)
        assert R.rom_word(word, S.ROM_BASE) == ref.rom_word(word, S.ROM_BASE)
    for n in (4, 8):
        for rd in range(32):
            for r1, r2 in ((1, 2), (rd, 2), (1, rd), (rd, rd), (0, 2), (1, 0)):
                regs = base.copy()
                check(encoding(n, rd, r1, r2), regs, category='aliases')
        for a in (S.DROM_BASE, S.DROM_BASE+4194304-n, S.DROM_BASE+4194304,
                  S.DROM_BASE-n, S.RAM_BASE, 0, 0x20000, 0xfffffff8, 0xfffffffc):
            for b in (S.RAM_BASE, S.RAM_BASE+65536-2*n, S.RAM_BASE+65536,
                      S.RAM_BASE-2*n, S.DROM_BASE, 0, 0xfffffff0):
                regs = base.copy()
                regs[1:3] = [a, b]
                check(encoding(n), regs, category='spans')
        for low_a in range(64):
            for low_b in range(64):
                regs = base.copy()
                regs[1:3] = [S.DROM_BASE+low_a, S.RAM_BASE+low_b]
                check(encoding(n), regs, category='alignment_residues')
        for bank in range(512):
            regs = base.copy()
            regs[1] = S.DROM_BASE + bank*8192 + rng.randrange(8192//n)*n
            regs[2] = S.RAM_BASE + rng.randrange(65536//(2*n))*2*n
            check(encoding(n), regs, category='drom_banks')
        for offset in range(0, 65536, 2*n):
            regs = base.copy()
            regs[2] += offset
            check(encoding(n), regs, category='ram_rows_banks_slots')
        for _ in range(1000):
            regs = [0] + [rng.getrandbits(32) for _ in range(31)]
            regs[1] = S.DROM_BASE+rng.randrange(4194304//n)*n
            regs[2] = S.RAM_BASE+rng.randrange(65536//(2*n))*2*n
            check(encoding(n), regs, category='random')
        old = [0, 1, 0x3fff, 0x4000, 0x1fffffff, 0x20000000,
               0x7fffffff, 0x80000000, 0xbfffffff, 0xc0000000, 0xffffc000, 0xffffffff]
        for w, a in ((-128, -32768), (127, 32767), (-128, 32767), (127, -32768), (0, 0)):
            drom[:n] = bytes([w & 255])*n
            ram[:2*n] = struct.pack('<h', a)*n
            for acc in old:
                regs = base.copy()
                regs[3] = acc
                check(encoding(n), regs, category='extremes_wrap_chunks')
        for i in range(n):
            drom[i] = (-128 if i % 2 else 127) & 255
            ram[2*i:2*i+2] = struct.pack('<h', -32768 if i % 2 else 32767)
        check(encoding(n), base, category='alternating')
        for kind, size in ((0, 1), (1, 2), (2, 4)):
            for lane in range(0, 2*n, size):
                regs = base.copy()
                regs[4:6] = [S.RAM_BASE+lane, 0xa583f17e]
                store = 0x23 | kind << 12 | 4 << 15 | 5 << 20
                m = R.Machine([store, encoding(n)], list(struct.unpack('<16384I', ram)), regs, 0, drom=drom)
                m.record()
                actual_ram = struct.pack('<16384I', *m.ram)
                expected = O.execute(encoding(n), regs, actual_ram, drom)
                assert m.record() == O.serialize(expected, encoding(n), S.ROM_BASE+4)
                assert tuple(m.x) == expected.regs
                counts['prior_stores'] = counts.get('prior_stores', 0)+1
    check(encoding(8), base, maximum=4, category='disabled_width')
    check(encoding(16), base, category='disabled_width')
    for n in (4, 8):
        for k in range(1, 8):
            regs = base.copy()
            regs[3] = (1 << (4*k))-1
            drom[:n] = bytes([1]+[0]*(n-1))
            ram[:2*n] = bytes([1])+bytes(2*n-1)
            check(encoding(n), regs, category='nibble_carry_chains')
        for acc in (0, 0x7fffffff, 0xffffffff):
            regs = base.copy()
            regs[3] = acc
            drom[:n] = bytes([1]*n)
            ram[:2*n] = struct.pack('<'+'h'*n, *([32767, -32767]*(n//2)))
            check(encoding(n), regs, category='cancellation')
    regs = base.copy()
    machine = R.Machine([encoding()]*128, list(struct.unpack('<16384I', ram)), regs.copy(), 0, drom=drom)
    commits = refreshes = 0
    while commits < 128:
        if machine.n == S.N_MAX:
            assert machine.record() == [S.tok('R', machine.anchor+machine.n)]
            refreshes += 1
            continue
        expected = O.execute(encoding(), regs, ram, drom)
        pc = machine.pc
        assert machine.record() == O.serialize(expected, encoding(), pc)
        regs = list(expected.regs)
        assert machine.x == regs and machine.pc == pc+4
        commits += 1
    counts['chained_dots'] = commits
    counts['refreshes_in_chain'] = refreshes
    regs = base.copy()
    regs[1] += 1
    machine = R.Machine([0]*5+[encoding()], list(struct.unpack('<16384I', ram)), regs, 5, drom=drom)
    assert machine.record() == O.serialize(O.execute(encoding(), regs, ram, drom), encoding(), S.ROM_BASE+20)
    assert machine.fault == (2, S.ROM_BASE+20) and machine.pc == S.ROM_BASE+20
    counts['nonzero_fault_pc'] = 1
    # Every signed operand pair is checked through independent byte interpretations.
    import numpy as np
    raw = np.arange(65536, dtype=np.int64)
    raw_bytes = np.arange(65536, dtype='<u2').tobytes()
    signed = np.frombuffer(raw_bytes, dtype='<i2').astype(np.int64)
    reference_halfwords = np.array([ref.sx(int(v), 16) for v in raw], dtype=np.int64)
    for byte in range(256):
        independent_byte = int.from_bytes(bytes([byte]), 'little', signed=True)
        assert np.array_equal(independent_byte*signed, ref.sx(byte, 8)*reference_halfwords)
    counts['exhaustive_s8_s16_pairs'] = 256*65536
    result = dict(status='pass', counts=counts, manifest=manifest,
                  scope='ISA and software records only; no neural execution',
                  upstream_record_inherited=R.Machine.__mro__[1].record is ref.Machine.record)
    (Path(__file__).parent / 'isa-result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'manifest'}, indent=2))


if __name__ == '__main__':
    (ROOT / '.cache/neural-accelerator').mkdir(parents=True, exist_ok=True)
    main()
