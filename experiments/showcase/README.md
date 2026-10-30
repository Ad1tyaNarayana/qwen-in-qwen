# Qwenputer general-purpose program pack

The new guests all target the unified W=4 CPU with NDOT4, with 64 KiB RAM,
4 MiB DROM and 256 KiB immutable code. They change program and data state.
They use one network. No guest-specific network construction is permitted.

The common stock BF16 weight SHA-256 is
`0e515bbb3109997a130c903deb40dce3447f8991e15b6637e0071c71f8b6e07d`.
[weight_hash.py](weight_hash.py) hashes contiguous little-endian BF16 tensor
bytes in lexical `state_dict` name order, including the tied output tensor
and zero final padding layer. [weights.json](evidence/weights.json) records
names, shapes, individual hashes and constructor bindings. Emission never
loads an ELF or a snapshot. Stock workers independently hash their actual
loaded model and require this value before generation.

All five native/software-reference and autonomous fused neural runs pass.
Every actual generated token matches the independent reference. All 65,536 RAM
bytes and all registers in each of the three recurrent copies match literally
for every guest. All five stock prefixes pass, with the actual loaded BF16
weights independently hashed on the GPU PC before each stock execution.

| Guest | Instructions | Trace tokens | Mismatches | GPU minutes | Stock prefix | Output |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| [SHA-256](sha256/README.md) | 11,191 | 25,186 | 0 | 0.112 | 4,096 | Empty and `abc` digests |
| [Life](life/README.md) | 346,545 | 403,149 | 0 | 1.798 | 4,096 | Five toroidal frames |
| [Tic-tac-toe](tictactoe/README.md) | 705,245 | 895,755 | 0 | 4.000 | 4,096 | `CPU wins` |
| [MNIST NDOT4](mnist/README.md) | 121,329 | 189,092 | 0 | 0.843 | 16,384 | `7, 2, 1` |
| [MNIST scalar](mnist-scalar/README.md) | 560,569 | 785,128 | 0 | 3.501 | 4,096 | `7, 2, 1` |

GPU minutes are generation time for one complete self-fed execution each.
They include replay, feedback, synchronization and token transfer. They exclude
setup, comparison, IO, checkpoint gaps and the separate stock checks.

MNIST float test accuracy is 96.06%; integer test accuracy is 96.05%.
The same native integer C classifier reproduces every one of the 10,000
integer predictions. NDOT4 reduces this complete guest's instructions by
4.6202× and trace tokens by 4.1521×. These ratios count actual retired work;
they are not GPU wall-time measurements. [Local acceptance](evidence/local-validation.json)
and [instruction scope](evidence/workload-scope.json) retain the counts.
The first NDOT4 begins at trace offset 12,377. The accepted 16,384-token
stock prefix completes 524 NDOT4 instructions; other guests check 4,096 tokens.

The full scalar GPU run takes 210.051 seconds, versus 50.590 seconds
with NDOT4, an observed 4.1520× speedup. This is one complete execution per
build, with identical DROM and output.
The network weight hash is identical. [Summary evidence](evidence/summary.json)
binds the rates, native accuracy and all five completed acceptance reports.

## Reproduce on the Mac

Use the existing toolchain and venv. Commands run from the repository root.
There are no system installations, upstream edits or commits.

```sh
.venv/bin/python experiments/showcase/mnist/train.py
python3 experiments/showcase/build.py sha256
python3 experiments/showcase/build.py life
python3 experiments/showcase/build.py tictactoe
python3 experiments/showcase/build.py mnist
python3 experiments/showcase/build.py mnist --scalar
.venv/bin/python experiments/showcase/run_reference.py .cache/showcase/sha256
.venv/bin/python experiments/showcase/run_reference.py .cache/showcase/life
.venv/bin/python experiments/showcase/run_reference.py .cache/showcase/tictactoe
.venv/bin/python experiments/showcase/run_reference.py .cache/showcase/mnist
.venv/bin/python experiments/showcase/run_reference.py .cache/showcase/mnist-scalar
.venv/bin/python experiments/showcase/verify_local.py
.venv/bin/python experiments/showcase/inspect_workload.py
.venv/bin/python experiments/showcase/weight_hash.py
.venv/bin/python experiments/showcase/weight_hash.py --original \
  --out experiments/showcase/evidence/weights-original.json
python3 experiments/showcase/make_manifests.py
```

Raw traces are created exclusively. Preserve a completed fixture before making
a fresh reference run. Binaries, raw data and machine packages remain under
`.cache/showcase/`. Each demo's `evidence/build.json` binds its actual ELF and
full zero-filled 4 MiB DROM image. Do not confuse the MNIST guest's trained
model weights in DROM with the common CPU network weights.

## GPU acceptance

```sh
.venv/bin/python experiments/showcase/launch.py --label FRESH
.venv/bin/python experiments/showcase/status.py
.venv/bin/python experiments/showcase/collect.py
.venv/bin/python experiments/showcase/verify_gpu.py .cache/showcase/actual-FRESH
.venv/bin/python experiments/showcase/check_verifier_controls.py .cache/showcase/actual-FRESH
python3 experiments/showcase/make_manifests.py
```

The launcher atomically acquires `C:\Users\Admin\gpu-lock` using `ssh gpu`
and writes its owner and time budget. If held, it exits 75 without starting
work. `--wait-seconds 3600` retries atomic acquisition every 30 seconds.
It syncs a named tar payload, checks the existing remote venv and uses
encoded PowerShell plus `Invoke-CimMethod Win32_Process Create` to detach.
Outputs use a fresh `artifacts/showcase-*` directory. A batch deadline and
per-worker watchdog bound its lock hold below four hours. Its `finally` block
releases the lock on success and ordinary worker failure. It never kills
another task's process.

Instruction and trace counts exclude neural prefill and GO.

The fused run reuses the accepted checkpoint worker. It starts from ELF boot,
checks every generated token against the independent reference, checkpoints
actual GPU state at 65,536 and 65,792 tokens and continues to HALT. SHA-256
finishes before either boundary; subsequent invocations restore its complete
actual checkpoint without generating new tokens. No reference checkpoint is
constructed or injected. Each guest must literally match all final RAM bytes
and every register in all three recurrent copies. No Gate 2 dead-state exception
is used. Cross-binary MNIST comparison checks observable outputs; each binary
has its own exact final-state reference.

The bounded stock check uses ordinary autonomous
`Qwen3_5ForCausalLM.generate`, no offload, a 15 GiB allocator cap and both CUDA
and physical-memory free-VRAM checks at 512 MiB. Only initial ELF state enters
the cache. The loaded BF16 weight hash must match the common value. The Mac
verifier compares exported actual token bytes again, decodes BYTE output,
compares full RAM and register files, checks chunk hashes and checks worker
exits. A failed run cannot populate a verified manifest. Single-bit controls reject
corruption of the final fused token, final stock token, RAM and one register copy;
see [the controls](evidence/verifier-controls.json).

Prepared source bytes differ between Mac and Windows because of line endings.
The fused binding compares the exact generated constructor source hash and
all constructor dependency hashes. Stock acceptance additionally compares the
actual full tensor hash. No numerical comparison has a tolerance.

## Existing results

The gallery imports [Doom first frame](manifest/doom-first-frame.json),
[the 38-position NDOT story](manifest/stories260k-38.json) and
[Phase 1](manifest/phase1.json). Their original evidence stays in place.
Doom and Phase 1 use the original circuit, whose retrospective stock BF16 hash
is `3aaf0b6eed10fe0581a5f4f440fc014bcf57d375cf1d1913f891c9fb5f33dab4`.
Those weights differ from the unified circuit. Historical hashes are rebuilt
from the corresponding constructor, not measured from old GPU tensor dumps.
The 38-position story has no matching-ELF stock prefix imported. Its related
five-position ELF passed 32,768 tokens, which is separate evidence.

[decisions.tsv](decisions.tsv) records the implementation choices. Model the
Domain keeps guest fixtures and measured acceptance gates explicit. Benchmark
Checklist keeps deterministic work reductions separate from measured GPU time.
Execution is inline, with no independent agent review.
