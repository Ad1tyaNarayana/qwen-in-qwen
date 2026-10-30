.section .text.init,"ax",@progbits
.global _start
_start:
  .option push
  .option norelax
  li sp, 0x0100F000
  .option pop
  call engine_main
  li a7, 93
  ecall
1: j 1b
