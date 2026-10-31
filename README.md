# Qwenputer games and tiny language models

Qwenputer compiles an RV32IM CPU into fixed Qwen3.5-architecture weights. The outer network has no pretrained weights. This research adds a custom-0 `ndot` accelerator and runs learned tiny language models as guest programs inside that CPU.

The older public repository is [Ad1tyaNarayana/qwenputer-games](https://github.com/Ad1tyaNarayana/qwenputer-games). This branch prepares a later public release; no release bundle is hosted yet.

## Results and boundaries

The compact fused executor completed a 38-position stories260K run and a six-position, 1,510,800-parameter tiny Qwen3.5 guest. Their 53,296,563 and 35,836,005 CPU trace tokens matched their own independent references, with literal final RAM and all three register copies checked. Complete SHA-256, Life, tic-tac-toe, and MNIST programs use the same CPU network.

One scalar and one NDOT run on the same network took 4,361.42 and 1,446.63 generation seconds: a measured 3.0149× accelerator gain for that workload. This is one run each, not a distribution. The historical Doom ratio 2.105× is an observed ratio with interrupted-run and executor caveats, not an isolated speedup measurement.

Stock Hugging Face `generate` covers bounded prefixes only: 32,768 trace tokens for the related five-position NDOT stories fixture, 16,384 for tiny Qwen, and 4,096 for unified Doom. These prefixes complete no guest prediction or Doom frame. Complete large workloads use the compact executor. Tiny Qwen integer inference agreed with floating point in 291/327 predictions on inspected prompts used during quantization investigation, not unseen evaluation. Qwen3.5-0.8B is infeasible within this design's memory map. Hashes detect changes; they do not prove execution provenance.

Read the unchanged articles for the evidence and qualifications:

- [A tiny language model inside a compiled Qwen network](docs/articles/tiny-llm-inside-qwen.md)
- [Doom on Qwenputer](docs/articles/doom-on-qwenputer.md)
- [Original CHIP-8 and Doom project guide](docs/CHIP8-AND-DOOM.md)

## Verify a clean clone

Use Python 3.13, Git, and a C compiler named `clang`. On macOS, install the command line developer tools first. Create an environment outside the repository:

```sh
python3 -m venv /tmp/qwenputer-verify-venv
/tmp/qwenputer-verify-venv/bin/python -m pip install -r requirements.txt
/tmp/qwenputer-verify-venv/bin/python tools/verify_all.py --tier 0
```

Tier 0 needs no network or GPU after dependency installation. It checks tracked artifacts, gameplay, checkpoint corruption rejection, software ISA behavior, and evidence integrity. It does not regenerate the historical GPU results. Every unchecked claim is printed as `SKIPPED`, separate from `PASS`.

To check retained non-Doom outputs, obtain the owner-built bundle and run:

```sh
/tmp/qwenputer-verify-venv/bin/python tools/verify_all.py --tier 1 --bundle /path/to/qwenputer-evidence-v1.tar.gz
```

A local path or HTTPS URL is accepted. The tracked manifest authenticates the compressed archive and each file before any bundled input is used. No download URL is supplied until the owner publishes a bundle. [Reproduction instructions](docs/REPRODUCE.md) cover building it, tiers 2 and 3, timings, and the claim-to-command map.

Doom data is excluded from git and the bundle. Doom output verification is always `SKIPPED` in the release command. You need your own IWAD and locally regenerated private outputs to check those results beyond tracked hashes and timing records. No game data or imagery is distributed here.

## Repository layout

- `tools/`: release checks, bundle builder, and original CHIP-8 tools.
- `experiments/`: [experiment index](experiments/README.md), verifiers, implementations, and immutable evidence.
- `docs/articles/`: result narratives and source maps.
- `artifacts/`: tracked CHIP-8 traces and reports.
- `vendor/qwenputer/`: hash-pinned MIT CPU/compiler source for offline verification.
- `release/`: tracked evidence hashes and bundle manifest; bundle bytes remain under ignored `.cache/release/`.

[Contributing](CONTRIBUTING.md), [citation](CITATION.cff), and [third-party licences](THIRD_PARTY_NOTICES.md) describe release obligations. Research reports retain their original source hashes and commit citations. Keep repository history when cloning; some checks explicitly resolve historical source bindings. Commit hashes cited in reports, evidence and articles predate a commit-message edit; [the commit map](release/commit-map.json) gives each current hash, with identical file trees.
