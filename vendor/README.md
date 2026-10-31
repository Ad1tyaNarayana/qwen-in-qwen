# Pinned offline compiler source

`qwenputer/` contains the `experimental/riscv/chipformer` source and MIT licence from Qwenputer commit `ca13310e2f9f788d1dd3feeea8213c1b0c23913c`. The release runner copies this snapshot into a temporary `.upstream/qwenputer` directory. The memory-map preparation code verifies every snapshot file against `PIN.json` before applying its existing patches.

No pretrained weights, game ROMs, IWADs, or engine binaries are included. The Python package includes upstream CPU/compiler utilities; release commands use the same fixed source snapshot offline. Do not edit this directory to fix a failing comparison. Update the pin and review source changes explicitly.
