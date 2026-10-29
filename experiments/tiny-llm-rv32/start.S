.section .text.init,"ax",@progbits
.global _start
_start:
  .option push
  .option norelax
  la sp, __stack_top
  .option pop
  call engine_main
  li a7, 93
  ecall
1: j 1b
