.section .text.init
.globl _start
_start:
    call main
    li a7, 93
    ecall
1:  j 1b
