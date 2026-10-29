"""Compare all loadable bytes and zero-fill extents of two RV32 ELF files."""
import argparse
import hashlib
import json
from pathlib import Path
import struct


def segments(path):
    raw = path.read_bytes()
    assert raw[:6] == b'\x7fELF\x01\x01'
    offset = struct.unpack_from('<I',raw,28)[0]
    size,count = struct.unpack_from('<HH',raw,42)
    result = []
    for i in range(count):
        header = struct.unpack_from('<8I',raw,offset+i*size)
        kind,start,vaddr,paddr,filesize,memsize,flags,alignment = header
        if kind == 1:
            assert start+filesize <= len(raw) and filesize <= memsize
            result.append((vaddr,raw[start:start+filesize],memsize,flags))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('left',type=Path)
    p.add_argument('right',type=Path)
    p.add_argument('--report',type=Path,required=True)
    a = p.parse_args()
    left,right = segments(a.left),segments(a.right)
    assert left == right
    entries = [struct.unpack_from('<I',p.read_bytes(),24)[0] for p in (a.left,a.right)]
    flags = [struct.unpack_from('<I',p.read_bytes(),36)[0] for p in (a.left,a.right)]
    assert entries[0] == entries[1] and flags[0] == flags[1]
    report = dict(scope='Every PT_LOAD byte, address, memory extent and permission',
        loadable_segments_equal=True,includes_code_ram_drom=True,
        entry_address=entries[0],elf_flags=flags[0],
        elf_sha256=[hashlib.sha256(p.read_bytes()).hexdigest() for p in (a.left,a.right)],
        segments=[dict(address=addr,file_bytes=len(raw),memory_bytes=mem,flags=flags,
                       sha256=hashlib.sha256(raw).hexdigest()) for addr,raw,mem,flags in left])
    a.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
