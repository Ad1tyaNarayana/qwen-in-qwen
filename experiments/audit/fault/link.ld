ENTRY(_start)
SECTIONS {
 .text 0x40000 : { *(.text.init) }
 .rodata 0x02000000 : { *(.rodata*) }
 .model : { KEEP(*(.model)) }
 . = 0x023FFFFC;
 .last 0x023FFFFC : { LONG(0x8101FF7F) }
 /DISCARD/ : { *(.riscv.attributes) *(.comment) }
}
