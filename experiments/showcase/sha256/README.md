# SHA-256 guest

`main.c` computes SHA-256 for the empty string and `abc` using integer RV32IM
operations. It implements the one-block case for these short messages.
All padding, schedule expansion, rounds and digest formatting run in the guest.
`verify_local.py` checks the output against Python `hashlib`.

From the repository root, run `python3 experiments/showcase/build.py sha256`,
then `.venv/bin/python experiments/showcase/run_reference.py .cache/showcase/sha256`.
The shared [workflow](../README.md) covers neural and stock acceptance.
[Build and reference evidence](evidence/) and the [gallery manifest](../manifest/sha256.json)
bind the actual artifacts. Full neural execution and the bounded stock prefix pass.
