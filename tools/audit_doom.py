#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess


DOOM_REV = "e257a09527062bfb986b6e5e0780839b6cbd5b71"
QWEN_REV = "ca13310e2f9f788d1dd3feeea8213c1b0c23913c"
ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--qwenputer", type=Path, required=True)
parser.add_argument("--doomgeneric", type=Path, default=ROOT / "doomgeneric")
parser.add_argument("--output", type=Path, default=ROOT)
parser.add_argument("--zig", type=Path)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
source = args.doomgeneric / "doomgeneric"
rv = args.qwenputer / "experimental/riscv/chipformer/rv"
for repo, rev in [(source.parent, DOOM_REV), (args.qwenputer, QWEN_REV)]:
    actual = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    assert actual == rev, (repo, actual, rev)


def define(path, name):
    return int(re.search(r"^#define\s+" + name + r"\s+(\d+)", path.read_text(), re.M)[1])


width = define(source / "i_video.h", "SCREENWIDTH")
height = define(source / "i_video.h", "SCREENHEIGHT")
out_x = define(source / "doomgeneric.h", "DOOMGENERIC_RESX")
out_y = define(source / "doomgeneric.h", "DOOMGENERIC_RESY")
zone_mib = define(source / "i_system.c", "DEFAULT_RAM")
assert "malloc(DOOMGENERIC_RESX * DOOMGENERIC_RESY * 4)" in (source / "doomgeneric.c").read_text()
assert "Z_Malloc (SCREENWIDTH * SCREENHEIGHT, PU_STATIC, NULL)" in (source / "i_video.c").read_text()
assert "*size = default_ram * 1024 * 1024" in (source / "i_system.c").read_text()
assert "RAM_BASE, RAM_WORDS = 0x0, 1 << 13" in (rv / "spec.py").read_text()
assert "__heap_end = 0x7000" in (rv / "env/rom.ld").read_text()
assert "DATA_BASE = 0x100" in (rv / "spec.py").read_text()
result = {
    "doomgeneric_revision": DOOM_REV,
    "qwenputer_revision": QWEN_REV,
    "machine": {"ram_bytes": 32768, "ram_after_null_page_bytes": 32512,
                "maximum_heap_before_static_data_bytes": 0x7000 - 0x100,
                "reserved_stack_bytes": 4096, "drom_bytes": 131072, "rom_bytes": 262144},
    "source_allocations": {
        "indexed_framebuffer_bytes_within_zone": width * height,
        "default_output_framebuffer_bytes_outside_zone": out_x * out_y * 4,
        "default_zone_bytes": zone_mib * 1024 * 1024,
        "default_zone_plus_output_bytes": zone_mib * 1024 * 1024 + out_x * out_y * 4,
        "output_bytes_if_configured_320x200": width * height * 4,
        "zone_plus_output_if_configured_320x200": zone_mib * 1024 * 1024 + width * height * 4,
    },
    "gate": "NO_GO: the 64000-byte engine framebuffer alone exceeds all 32768 bytes of writable RAM",
}
assert result["source_allocations"]["indexed_framebuffer_bytes_within_zone"] > result["machine"]["ram_bytes"]
result["source_sha256"] = {
    str(path.relative_to(source.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in [source / "i_video.h", source / "i_video.c", source / "doomgeneric.h",
                 source / "doomgeneric.c", source / "i_system.c"]
}


def elf_sections(path):
    data = path.read_bytes()
    assert data[:7] == b"\x7fELF\x01\x01\x01"
    hdr = struct.unpack_from("<16sHHIIIIIHHHHHH", data)
    assert hdr[2] == 243
    sections = [struct.unpack_from("<IIIIIIIIII", data, hdr[6] + i * hdr[11]) for i in range(hdr[12])]
    names = sections[hdr[13]]
    names = data[names[4]:names[4] + names[5]]
    return [{"name": names[s[0]:].split(b"\0", 1)[0].decode(), "bytes": s[5],
             "write": bool(s[2] & 1), "alloc": bool(s[2] & 2), "exec": bool(s[2] & 4)} for s in sections]


if args.zig:
    build = args.output / "build"
    build.mkdir(exist_ok=True)
    makefile = (source / "Makefile").read_text()
    object_names = re.search(r"^SRC_DOOM = (.+)$", makefile, re.M)[1].split()
    object_names.remove("doomgeneric_xlib.o")
    cc = [str(args.zig.resolve()), "cc", "-target", "riscv32-linux-musl",
          "-mcpu=baseline_rv32+m+zba", "-mabi=ilp32", "-Os", "-fno-pic",
          "-ffunction-sections", "-fdata-sections", "-DNORMALUNIX", "-DLINUX",
          "-DSNDSERV", "-D_DEFAULT_SOURCE", "-Wno-implicit-function-declaration"]
    commands = []
    objects = []
    with (build / "compile.log").open("w") as log:
        for name in object_names:
            obj = build / name
            command = cc + ["-c", str(source / name.replace(".o", ".c")), "-o", str(obj)]
            commands.append(command)
            subprocess.run(command, stdout=log, stderr=log, check=True)
            objects.append(obj)
    (build / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    sections = [dict(s, object=obj.name) for obj in objects for s in elf_sections(obj)]
    result["crosscompile"] = {
        "kind": "Unpruned relocatable engine objects; not a runnable port or live-size lower bound",
        "target": "RV32IM+Zba, ILP32, Linux musl headers only",
        "engine_translation_units": len(objects),
        "excluded_platform_translation_unit": "doomgeneric_xlib.c",
        "libc_linked": False,
        "linked": False,
        "writable_allocated_bytes": sum(s["bytes"] for s in sections if s["alloc"] and s["write"]),
        "readonly_allocated_bytes": sum(s["bytes"] for s in sections if s["alloc"] and not s["write"] and not s["exec"]),
        "code_bytes": sum(s["bytes"] for s in sections if s["alloc"] and s["exec"]),
    }
    (build / "sections.json").write_text(json.dumps(sections, indent=2) + "\n")
(args.output / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
