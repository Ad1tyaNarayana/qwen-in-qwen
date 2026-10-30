.section .model,"a",@progbits
.global blob, blob_end
blob: .incbin "blob.bin"
blob_end:
