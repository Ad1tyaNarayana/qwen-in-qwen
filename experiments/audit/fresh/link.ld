ENTRY(_start)
MEMORY {
 ROM (rx) : ORIGIN = 0x40000, LENGTH = 256K
 RAM (rw) : ORIGIN = 0x01000100, LENGTH = 0xC000
 DROM (r) : ORIGIN = 0x02000000, LENGTH = 4M
}
SECTIONS {
 .text : { *(.text.init) *(.text.engine_main) *(.text.[a-eg-z]*) *(.text) } > ROM
 .far 0x7FF00 : { *(.text.far) } > ROM
 .rodata 0x02000000 : { *(.rodata* .srodata*) } > DROM
 .model : ALIGN(16) { KEEP(*(.model)) } > DROM
 .data : { *(.data* .sdata*) } > RAM
 .bss : { *(.bss* .sbss*) *(COMMON) . = ALIGN(16); } > RAM
 /DISCARD/ : { *(.eh_frame*) *(.comment) *(.riscv.attributes) }
}
