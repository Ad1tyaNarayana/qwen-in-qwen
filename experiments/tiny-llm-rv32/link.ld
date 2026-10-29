ENTRY(_start)
MEMORY {
 ROM (rx) : ORIGIN = 0x40000, LENGTH = 256K
 RAM (rw) : ORIGIN = 0x01000100, LENGTH = 0xff00 - STACK_BYTES - RAM_MARGIN
 DROM (r) : ORIGIN = 0x02000000, LENGTH = 4M
}
SECTIONS {
 .text : { *(.text.init) *(.text*) } > ROM
 .rodata : { *(.rodata* .srodata*) } > DROM
 .model : ALIGN(16) { KEEP(*(.model)) } > DROM
 .data : { *(.data* .sdata*) } > RAM
 .bss : { *(.bss* .sbss*) *(COMMON) . = ALIGN(16); __scratch_end = .; } > RAM
 __stack_top = 0x01010000;
 ASSERT(__scratch_end <= __stack_top - STACK_BYTES - RAM_MARGIN, "Static RAM overlaps stack or margin")
 /DISCARD/ : { *(.eh_frame*) *(.comment) }
}
