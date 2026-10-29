"""Compile the guest .insn functions and inspect their exact ELF instruction words."""
import json
from pathlib import Path
import struct
import subprocess
import tempfile
from prepare import ROOT


def main():
    out = Path(tempfile.mkdtemp(prefix='intrinsic-', dir=ROOT / '.cache/neural-accelerator'))
    source = out / 'intrinsic.c'
    source.write_text('#include "ndot.h"\n'
                      'uint32_t four(uint32_t c,const int8_t*w,const int16_t*a){return ndot4(c,w,a);}\n'
                      'uint32_t eight(uint32_t c,const int8_t*w,const int16_t*a){return ndot8(c,w,a);}\n')
    zig = next((ROOT / '.venv/lib').glob('python*/site-packages/ziglang/zig'))
    command = [str(zig), 'cc', '-target', 'riscv32-freestanding', '-mcpu=generic_rv32+m',
               '-O2', '-c', '-I', str(Path(__file__).parent), str(source), '-o', str(out / 'intrinsic.o')]
    subprocess.run(command, check=True, capture_output=True)
    raw = (out / 'intrinsic.o').read_bytes()
    assert raw[:6] == b'\x7fELF\x01\x01'
    shoff = struct.unpack_from('<I', raw, 32)[0]
    shsize, shnum = struct.unpack_from('<HH', raw, 46)
    executable = []
    for i in range(shnum):
        fields = struct.unpack_from('<10I', raw, shoff+i*shsize)
        if fields[2] & 4:
            data = raw[fields[4]:fields[4]+fields[5]]
            executable.extend(struct.unpack('<'+'I'*(len(data)//4), data))
    dots = [word for word in executable if word & 127 == 11]
    assert len(dots) == 2
    assert sorted((w >> 25, w >> 12 & 7) for w in dots) == [(32, 0), (32, 1)]
    for w in dots:
        assert (w >> 7 & 31, w >> 15 & 31, w >> 20 & 31) == (10, 11, 12)
    report = dict(status='pass', words=[f'{w:08x}' for w in dots],
                  decoded=[dict(funct7=w >> 25, funct3=w >> 12 & 7, rd=w >> 7 & 31,
                                rs1=w >> 15 & 31, rs2=w >> 20 & 31) for w in dots],
                  zig=subprocess.check_output([str(zig), 'version'], text=True).strip(), command=command)
    (Path(__file__).parent / 'intrinsic-result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
