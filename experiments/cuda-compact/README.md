# Faithful native Windows circuit acceleration

The fastest tested path is upstream `tritonexec.Executor2`, driven by `fastexec.FreeRun` with its merged-chat-only Q table entry omitted. It runs the circuit's sparse fp32 arithmetic in fused Triton kernels under a CUDA graph. Each output token determines the next embedding on the GPU. It is a compact circuit executor, not the stock Transformers model and not an RV opcode emulator.

Both tested paths generated exactly the same 5,415-token CHIP-8 first-frame trace as the stock-HF CUDA run. Each repeated run reset to the same initial snapshot. Reference tokens were used only for comparison after generation.

| Path | Runs | Median seconds per 5,415 tokens | Range |
| --- | ---: | ---: | --- |
| Triton Executor2 plus CUDA graph | 5 | 0.8414882 | 0.8413926 to 0.8501917 |
| Sparse PyTorch Executor plus CUDA graph, no torch.compile | 5 | 3.6595212 | 3.6594120 to 3.6699488 |

The runs alternated between the two paths. These times include graph replay, device token decoding, self-fed embeddings, trace copies, and final CPU transfer. They exclude construction, graph capture, warm-up, and reference generation. Standalone one-run setup times were 1.13 seconds for Triton and 7.53 seconds for sparse PyTorch; these are not controlled setup comparisons. Standalone peak PyTorch CUDA allocation was 116.2 MB for Triton and 127.1 MB for sparse PyTorch.

A separate 64-token CUDA profiler run confirmed execution of the custom kernels. Head kernels accounted for 3.45 ms across 512 launches, sparse FFN kernels 3.09 ms across 640 launches, and apply kernels 1.24 ms across 1,152 launches. These were the largest captured device costs. The profile was excluded from timings. All repeated runs had zero token mismatches.

Both retained trace files have SHA-256 `38a4a54bfb199040be4dc4a02f449673320733c313663be522ac2b8c5b541d9d`. Independent local validation compared their uint32 contents with the retrieved stock-HF trace. The ELF remained `8969cdb7448eff1c03f44b0044f5b0f379da8c8390081f8e7a74355de5393cf0` and upstream remained `ca13310e2f9f788d1dd3feeea8213c1b0c23913c`.

The Windows fork under the Triton organization documents Windows support, sm120, torch 2.14 paired with Triton 3.8, and bundled CUDA toolchain/TinyCC. Its PyPI project links back to that repository. The compatible cp314 wheel `triton-windows==3.8.0.post29` is 54.1 MB. Installation used `--no-deps --require-hashes` with SHA-256 `1d0fd6a6cf36ba5cced997177068e328d015e42cd4300f6995fff5a6f8707aff`. Environment comparison shows this single added package and no removed packages. An actual Triton JIT arithmetic kernel also produced 1,024 exact CUDA fp32 outputs. No WSL, system CUDA toolkit, C compiler, driver, or global Python change was needed. Sources are the [Windows fork documentation](https://github.com/triton-lang/triton-windows/blob/readme/README.md) and [published package metadata](https://pypi.org/project/triton-windows/3.8.0.post29/).

`standalone-freerun.diff` contains the only source adaptation. Upstream `FreeRun` tries to construct a token for Q even though Q exists only in the merged chat output code and not in standalone `spec.FAMILIES`. The local copied module enumerates standalone families plus controls instead. Q is the final output family, so all standalone family IDs and embedding rows keep their existing indices. Circuit construction, Executor.step, Executor2, and its kernels are unchanged. The initial Q failure is preserved separately. The original upstream Git checkout remains clean.

Start with the native Windows environment in [the stock CUDA instructions](../cuda/README.md), then run the commands in `run-commands.ps1` from the repository root. They copy the archived harnesses and committed firmware into the locations used by the original run. The exact executed probe scripts are under `source/`. They loaded a copied adapted module from the artifact directory rather than changing upstream files. `acceleration_probe.py` has a 240-second process watchdog and `acceleration_compare.py` a 60-second watchdog. No experiment processes remain running.

These rates are for the original 32 KB circuit. The proposed 8 MiB RAM plus 8 MiB DROM circuit grows to hundreds of sequential layers and many more state heads. Its compilation time, memory use, and token rate remain unmeasured here. The base-map rates must not be extrapolated to Doom. Batch throughput also does not accelerate a single dependent game trace. The next useful test is a short exact free-running probe on that actual enlarged graph. The stock-HF run remains separate evidence; these compact measurements do not establish stock-HF Doom execution.
