# Fault/edge guests. CASE selects one.
.section .text.init,"ax",@progbits
.global _start
_start:
  li a0, 0x02000000
  li a1, 0x01000100
  li a2, 7
  li t0, -2
  sh t0, 0(a1)
  sh t0, 2(a1)
  sh t0, 4(a1)
  sh t0, 6(a1)
#if CASE==1
  li a0, 0x02000001
  .insn r 0x0b, 0, 0x20, a2, a0, a1
#elif CASE==2
  li a1, 0x01000102
  .insn r 0x0b, 0, 0x20, a2, a0, a1
#elif CASE==3
  li a0, 0x01000000
  .insn r 0x0b, 0, 0x20, a2, a0, a1
#elif CASE==4
  li a0, 0x023FFFFC
  li a1, 0x0100FFF8
  li t1, 3
  sh t1, 0(a1)
  li t1, -5
  sh t1, 2(a1)
  li t1, 7
  sh t1, 4(a1)
  li t1, -11
  sh t1, 6(a1)
  .insn r 0x0b, 0, 0x20, a2, a0, a1
  mv a0, a2
  li a7, 1
  ecall
  li a0, 0x02400000
  li a1, 0x01000100
  .insn r 0x0b, 0, 0x20, x0, a0, a1
#elif CASE==5
  .insn r 0x0b, 1, 0x20, a2, a0, a1
#elif CASE==6
  .insn r 0x0b, 2, 0x20, a2, a0, a1
#elif CASE==7
  li a4, 0x02000000
  .insn r 0x0b, 0, 0x20, x0, a4, a1
  li a3, 0x02000000
  .insn r 0x0b, 0, 0x20, a3, a3, a1
  mv a0, a3
  li a7, 1
  ecall
  srli a0, a3, 8
  ecall
  srli a0, a3, 16
  ecall
  srli a0, a3, 24
  ecall
  li a0, 0
#endif
  li a7, 93
  ecall
.section .rodata
d: .byte 1,-2,3,-4,5,-6,7,-8,127,-128,1,0,0,0,0,0,9,-9,100,-100
.section .model,"a",@progbits
.fill 4194000,1,0x5a
