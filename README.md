# Qwenputer inside Qwen

**A CPU and a dot-product accelerator compiled into one fixed Qwen3.5 network, running tiny language models as ordinary programs.**

```text
Qwen3.5-architecture network       fixed compiled weights, no pretrained weights
└── Qwenputer RV32IM CPU           registers and RAM held in recurrent state
    └── ndot accelerator           custom-0 dot-product instructions, same network
        └── your program           stories260K, tiny Qwen3.5, SHA-256, Life, MNIST
```

[Qwenputer](https://github.com/divagr18/qwenputer) compiles a 32-bit RISC-V CPU into Qwen3.5 weights. This project adds a generic dot-product accelerator to that same network, then runs learned language models inside it. The model weights sit in the CPU's memory, not in the network. Changing the program and data changes what runs; the outer weights stay the same.

## Results

| Program on the neural CPU | Output | CPU trace tokens | Mismatches |
| --- | --- | ---: | ---: |
| stories260K, 38 positions | "Once upon a time, there was a little girl named Lily. She loved to play outside in the park. One day, she saw a big" | 53,296,563 | 0 |
| Tiny Qwen3.5, 1,510,800 parameters, six positions | "Once upon a time, there" | 35,836,005 | 0 |

Mismatches are against each run's independent reference, with literal final RAM and all three register copies checked. Complete SHA-256, Game of Life, tic-tac-toe and MNIST programs run on the same CPU network.

One scalar and one NDOT run on the same network took 4,361.42 and 1,446.63 generation seconds, a measured 3.0149× accelerator gain for that workload. This is one run each, not a distribution.

## Limits

- Stock Hugging Face `generate` covers bounded prefixes only: 32,768 trace tokens for the related five-position NDOT stories fixture and 16,384 for tiny Qwen. These prefixes complete no guest prediction. Complete large workloads use the compact fused executor.
- Tiny Qwen integer inference agreed with floating point in 291/327 predictions on inspected prompts used during quantization investigation, not unseen evaluation.
- Qwen3.5-0.8B is infeasible within this design's memory map.
- Generation is slow: about 6.24 minutes per position for stories260K and 27.82 for tiny Qwen on an RTX 5070 Ti.
- Hashes detect changes; they do not prove execution provenance.

## Read more

- [A tiny language model inside a compiled Qwen network](docs/articles/tiny-llm-inside-qwen.md)
- [Experiment index](experiments/README.md)
- [Independent audit](experiments/audit/REPORT.md) and [blind review](experiments/blind-review/REPORT.md)

The offline Explorer zooms from the compiled network down to the CPU, the program and the inner model. Run `python3 experiments/explorer/server.py`, then open <http://127.0.0.1:8792>.

## Verify a clean clone

Use Python 3.13, Git, and a C compiler named `clang`. On macOS, install the command line developer tools first. Create an environment outside the repository:

```sh
python3 -m venv /tmp/qwenputer-verify-venv
/tmp/qwenputer-verify-venv/bin/python -m pip install -r requirements.txt
/tmp/qwenputer-verify-venv/bin/python tools/verify_all.py --tier 0
```

Tier 0 needs no network or GPU after dependency installation. It checks tracked evidence, checkpoint corruption rejection, and software ISA behavior. It does not regenerate the historical GPU results. Every unchecked claim is printed as `SKIPPED`, separate from `PASS`.

To check the retained run outputs, obtain the evidence bundle and run:

```sh
/tmp/qwenputer-verify-venv/bin/python tools/verify_all.py --tier 1 --bundle /path/to/qwenputer-evidence-v1.tar.gz
```

A local path or HTTPS URL is accepted. The tracked manifest authenticates the archive and each file before any bundled input is used. No download URL is supplied until a bundle is published. [Reproduction instructions](docs/REPRODUCE.md) cover building it, tiers 2 and 3, timings, and the claim-to-command map.

## Repository layout

- `experiments/`: implementations, verifiers and immutable evidence, listed in the [index](experiments/README.md).
- `tools/`: release checks, bundle builder, and early milestone tools.
- `docs/articles/`: result narrative and source map.
- `vendor/qwenputer/`: hash-pinned MIT CPU/compiler source for offline verification.
- `release/`: tracked evidence hashes and bundle manifest.
- `artifacts/`: early CHIP-8 milestone traces and reports.

## Related projects

- [qwenputer-doom](https://github.com/Ad1tyaNarayana/qwenputer-doom): the Doom frame and executor improvements on the same CPU line.
- [qwenputer-games](https://github.com/Ad1tyaNarayana/qwenputer-games): the original CHIP-8 milestone.

Directories named `dmc-*` began as Doom work. Their fused executor and helpers remain here because the language-model runs use them.

[Contributing](CONTRIBUTING.md), [citation](CITATION.cff), and [third-party licences](THIRD_PARTY_NOTICES.md) describe release obligations. Keep repository history when cloning; some checks explicitly resolve historical source bindings. Commit hashes cited in reports, evidence and articles predate a commit-message edit; [the commit map](release/commit-map.json) gives each current hash, with identical file trees.
