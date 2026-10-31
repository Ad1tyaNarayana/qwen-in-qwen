# Reproduce the retained results

## Set up an isolated environment

The release verification path was tested on macOS 26.7.1, Apple M4 Pro, 24 GiB unified memory, Python 3.13.4, torch 2.15.0, and transformers 5.21.0. `requirements.txt` records that Mac environment. Git and `clang` must be on PATH. No existing repository `.venv` is used or modified.

```sh
git clone --branch oss-release /path/to/source/repository /tmp/qwenputer-clean
cd /tmp/qwenputer-clean
python3 -m venv /tmp/qwenputer-clean-venv
/tmp/qwenputer-clean-venv/bin/python -m pip install -r requirements.txt
/tmp/qwenputer-clean-venv/bin/python tools/verify_all.py --tier 0
```

Replace the clone source with the eventual public URL. No new repository or release is published yet. The older public repository is `Ad1tyaNarayana/qwenputer-games`. Keep full Git history. Story bindings resolve workers at `e7bd077`; changing old hashes to match new code would invalidate the historical check.

Dependency installation uses the network. Tier 0 itself makes no network requests and needs no GPU. `vendor/qwenputer/PIN.json` supplies the pinned MIT CPU/compiler source offline. Checks run in disposable tracked-file copies, with their own empty `.cache` and vendored upstream. Existing verifiers may write reports inside those copies; retained evidence in the clone is untouched. Raw data verifiers keep their original assertions and tolerances.

The Linux CI job uses Python 3.13 and the CPU PyTorch wheel index before installing `requirements.txt`. It has not been executed in this Mac session. Linux CUDA is untested. Windows historical GPU runs used an RTX 5070 Ti with 16 GiB VRAM; `requirements-gpu.txt` records torch 2.14.1+cu130, transformers 5.21.0, and triton-windows 3.8.0.

For a new Windows environment, install the matching CUDA wheel first, then the recorded dependencies:

```sh
python -m venv /path/to/new/gpu-venv
/path/to/new/gpu-venv/Scripts/python.exe -m pip install torch==2.14.1+cu130 --index-url https://download.pytorch.org/whl/cu130
/path/to/new/gpu-venv/Scripts/python.exe -m pip install -r requirements-gpu.txt
```

This records the historical environment; no Windows installation or GPU rerun was performed for this release preparation.

## Choose a verification tier

| Tier | Inputs and command | What it establishes |
| --- | --- | --- |
| 0 | `python tools/verify_all.py --tier 0` | Tracked evidence integrity, CHIP-8 reference replay and retained traces, existing fusion/timing checks, CPU checkpoint corruption rejection, exhaustive software ISA gate, and pytest collection. |
| 1 | `python tools/verify_all.py --tier 1 --bundle /path/to/qwenputer-evidence-v1.tar.gz` | Tier 0 plus authenticated retained non-Doom outputs through the existing story, tiny-Qwen, showcase, explorer, and checkpoint-integrity verifiers. |
| 2 | `python tools/verify_all.py --tier 2 --bundle /path/to/qwenputer-evidence-v1.tar.gz` | Tier 1 plus a bounded stock Hugging Face CPU workload compiled from the pinned source. No pretrained model download. |
| 2, full IR | Add `--full-ir` | Full neural IR gate, with a 3,600-second timeout. It is separate from the short stock regeneration and may exceed the bound on slower CPUs. |
| 3 | Manual full GPU runs, below | New generation and training, rather than validation of retained output bytes. Not run in this release session. |

Every claim prints `PASS`, `FAIL`, or `SKIPPED`, a reason, and elapsed seconds. Any `FAIL` produces a nonzero exit. An unavailable requested bundle is a failure, not a skip. `SKIPPED` never counts as a pass. `--json /tmp/results.json` saves a fresh machine-readable result. Do not use Python `-O`, which disables the existing assertions.

The full IR scripts were written as command-line `main()` functions, so pytest originally collected no tests. They now expose `test_isa` and `test_ir`. The ISA gate runs at tier 0. The complete neural IR gate is long and runs only with the explicit tier-2 option; collection alone does not establish neural IR correctness.

## Build and authenticate a bundle

Only the owner or a researcher with the retained non-Doom outputs can build v1:

```sh
python tools/build_release_bundle.py --cache /path/to/retained/cache
python tools/check_release.py --bundle .cache/release/qwenputer-evidence-v1.tar.gz
python tools/verify_all.py --tier 1 --bundle .cache/release/qwenputer-evidence-v1.tar.gz
```

The builder prints the size of every allowlisted entry. It writes the compressed bundle into ignored `.cache/release/` and updates tracked `release/bundle-manifest.json`. The manifest contains the archive hash, exact member list, sizes, and each file's SHA-256. Review and commit a manifest update deliberately. The verifier checks the archive before extraction, rejects links and unsafe paths, then checks every file before invoking a verifier. HTTPS URLs use the same pinned manifest. The manifest does not establish who produced the original outputs.

The bundle includes complete story and tiny-Qwen actual/reference traces, final RAM/registers, output bytes, chunk journals, restore metadata, packed ELF inputs, the actual pre-shutdown CPU-state checkpoint required by the Qwen recovery verifier, showcase local/reference and GPU/stock outputs, and MNIST test inputs. Frozen independent reference packages are Python source only. Native showcase executables are rebuilt with `clang` on the verifier's platform.

The tiny-Qwen program ELF necessarily embeds its quantized guest model. A separate trained PyTorch guest checkpoint is unnecessary and excluded. The recovery `.pt` is recurrent CPU execution state, not guest training weights. The multi-GB training and quantization sweeps are excluded because the retained-output verifiers do not consume them. The TinyStories corpus and all Doom inputs and derived outputs are excluded. See [licence notes](../THIRD_PARTY_NOTICES.md) before redistributing the bundle.

## Match claims to evidence and commands

Commands below run inside `verify_all`'s disposable copy; direct research commands can require `.cache` and can overwrite historical reports. Use the release command for read-only verification of the clone.

| Claim | Retained evidence | Existing verifier used by release command |
| --- | --- | --- |
| CHIP-8 input-dependent gameplay and retained compact/stock/CUDA traces | `artifacts/compact-game*`, `artifacts/hf-*` | Tier 0: `tools/check_game.py` |
| Exact fused state comparisons and repeated bounded timings | `experiments/tiny-llm-ndot-fused/evidence/` | Tier 0: `check_evidence.py` in that directory |
| Doom historical timing/race record consistency, without game-output verification | `experiments/dmc-rom-fusion/evidence/`, `dmc-feedback-fusion/evidence/` | Tier 0: `check_evidence.py`, `check_diagnosis.py`; the Doom output claim remains SKIPPED |
| Corrupt checkpoint rejection before state copy | `experiments/checkpoint-integrity/evidence/cpu-restore.json` | Tier 0: `experiments/checkpoint-integrity/test_restore.py` with CPU tensors |
| Software NDOT ISA records and arithmetic | `experiments/neural-accelerator/isa-result.json` | Tier 0: pytest `test_isa.py`, including directed, seeded, and exhaustive arithmetic checks |
| 38-position story, literal state, contiguous chunk timing and historical sources | `experiments/tiny-llm-ndot-fused/evidence/story-validation-20261031.json` | Tier 1: `verify_story.py --source-revision e7bd077` with bundle inputs |
| 35,836,005-token tiny-Qwen run, power-off recovery, 16,384-token stock prefix | `experiments/tiny-qwen35/evidence/neural-validation.json` | Tier 1: `experiments/tiny-qwen35/verify_gpu.py` |
| Native/reference SHA-256, Life, tic-tac-toe, MNIST; 10,000-example integer classifier | `experiments/showcase/evidence/local-validation.json`, per-demo evidence | Tier 1: `experiments/showcase/verify_local.py` after native compilation |
| All showcase GPU tokens, literal RAM/registers, bounded stock prefixes | Per-demo `experiments/showcase/*/evidence/` | Tier 1: `experiments/showcase/verify_gpu.py` |
| Explorer graph exports, observed state, application arrays, corruption control | `experiments/explorer/data/`, `data-validation.json` | Tier 1: `verify_data.py` and `verify_showcase.py` |
| Controlled same-network 3.0149× scalar/NDOT ratio and restore reports | `experiments/checkpoint-integrity/evidence/controlled-measurement.json` | Tier 1: `experiments/checkpoint-integrity/verify.py` |
| Story export negative controls | `experiments/checkpoint-integrity/evidence/verifier-controls-20261031.json` | Tier 1: `test_verifier_exports.py --story-only`, nine checks; Doom controls excluded |
| New stock CPU execution of a small program | `artifacts/hf-smoke/report.json` is historical; new result stays temporary | Tier 2: `tools/hf_smoke.py` |
| Frozen evidence bytes | `release/tracked-evidence.json` | Tier 0: SHA-256 comparison, explicitly not provenance |

Integer quality 291/327 remains a result on six inspected prompts used during packing investigation. Tier 1 validates integer program execution; it does not independently rerun floating-point quality evaluation. Full training, unseen-prompt generalization, universal circuit correctness, and full stock completion are not established by these tiers.

## Run new GPU work explicitly

Workers take paths such as `--repo`, `--upstream`, `--inputs`, and `--out`. Read each worker's `--help` before running it. The following sends a worker command to a host you configure; it does not assume a private SSH alias or directory:

```sh
python tools/gpu_remote.py --host YOUR_SSH_HOST \
  --python D:/research/venv/Scripts/python.exe \
  --worker D:/research/repo/experiments/showcase/gpu_worker.py -- \
  --repo D:/research/repo --upstream D:/research/qwenputer \
  --inputs D:/research/inputs --out D:/research/new-run --max-seconds 9800
```

The retained launch/collect/status scripts accept shared explicit options `--ssh-host`, `--remote-root`, `--remote-lock`, and, where needed, `--training-root`. These configure paths; fixture labels and orchestration remain specific to the historical research. Audit and blind-review scripts remain unmodified historical records. No remote script was executed during release preparation.

Budget full generation from the measured evidence, not projected rates. Historical retained generation took 1.2115 hours for scalar five-position stories, 0.4018 hours for NDOT five-position stories, 3.9516 hours for the longer story, and 2.7817 hours for tiny Qwen. Setup, validation, IO, checkpoint/restart gaps, power-off time, and discarded replay are excluded. These are one run each. Their timings do not predict your machine's wall time.

The unified Doom frame retained 0.4703 generation hours. The earlier DMC run retained 0.8457 hours and was interrupted. The 2.105× historical ratio is observed, not a controlled executor-only speedup. Full GPU reproduction also needs independent software references, memory guards, exact own-reference comparisons, and fresh output directories. Use the experiment READMEs and unchanged articles for the precise workload and worker sequence.

## Keep Doom private

The automatic release tiers always report Doom output verification as SKIPPED. A stranger can check tracked hashes, source bindings, and timing-record arithmetic. Without locally regenerated private inputs, those records do not establish actual frame pixels, full Doom token traces, final RAM, or their provenance.

Supply your own local IWAD and engine source in a private directory. No game data download is provided. The existing [bare-metal port instructions](../experiments/doom-baremetal/README.md#build-and-reproduce) accept `--wad`, `--doomgeneric`, `--zig`, and `--out`. The [DMC port instructions](../experiments/dmc-rv32/README.md) describe its separate engine and conversion path. Keep all WAD-derived ELF/NOR files, traces, RAM, frames, and PNGs outside git and the bundle.

For the unchanged retained Doom verifier, build the required local reference trace, strip observer, and actual run exports first. `experiments/doom-showcase/verify_runs.py` names its `TRACE`, `OBSERVER`, and `CACHE` inputs; these refer to private outputs. Its full and unified runs are separate checks. `test_verifier_exports.py --out /tmp/private-controls.json` preserves the original story-plus-Doom acceptance and rejection suite when those private inputs exist. The release's `--story-only` scope is printed explicitly and does not claim those Doom controls ran.
