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

Replace the clone source with the eventual public URL. No new repository or release is published yet. Earlier milestones are public in `Ad1tyaNarayana/qwenputer-games` and `Ad1tyaNarayana/qwenputer-doom`. Keep full Git history. Story bindings resolve workers at `e7bd077`; changing old hashes to match new code would invalidate the historical check.

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

Release preparation measured these end-to-end commands on the Mac described above, using a fresh clone and fresh external virtual environment. The original working cache was renamed and its path pointed at a nonexistent directory for all three runs. Bundle bytes were supplied separately from a local file. Installation time is not included. These are observed verification timings, not performance benchmarks.

| Tier | PASS claims | FAIL claims | SKIPPED claims | Measured seconds |
| --- | ---: | ---: | ---: | ---: |
| 0 | 12 | 0 | 13 | 13.052 |
| 1 | 22 | 0 | 5 | 70.921 |
| 2 | 23 | 0 | 4 | 85.359 |

The tier-2 result and full per-claim timings are recorded in [release validation](../release/VALIDATION.md). Counts describe claim rows, not individual assertions. The release attack-control row includes nine test cases. Full IR, float quality sweeps, Doom output checks, and GPU regeneration were not run. No timing is claimed for them.

The tree policy limits individual files to 100 MiB. Bundle members are limited to 256 MiB, with at most 2 GiB expanded data. The builder also enforces a 2 GB compressed archive budget. `check_release` rejects absolute or escaping links, game data, Doom-derived executables/state/imagery, and common credential patterns. This is a release gate, not a claim that arbitrary secret formats can be detected.

The full IR scripts were written as command-line `main()` functions, so pytest originally collected no tests. They now expose `test_isa` and `test_ir`. The ISA gate runs at tier 0. The complete neural IR gate is long and runs only with the explicit tier-2 option; collection alone does not establish neural IR correctness.

## Published bundle

Release [`evidence-v1`](https://github.com/Ad1tyaNarayana/qwen-in-qwen/releases/tag/evidence-v1) hosts `qwenputer-evidence-v1.tar.gz` (238,005,501 bytes, SHA-256 `638764c250a16661db37afa284baf7afee50f2df67bcf49970f6d6b3c4b87b9c`). Pass its URL or a downloaded copy to `--bundle`:

```sh
python tools/verify_all.py --tier 1 --bundle https://github.com/Ad1tyaNarayana/qwen-in-qwen/releases/download/evidence-v1/qwenputer-evidence-v1.tar.gz
```

The tracked manifest pins the archive, so a changed or substituted download fails before use.

## Build and authenticate a bundle

Only the owner or a researcher with the retained non-Doom outputs can build v1:

```sh
python tools/build_release_bundle.py --cache /path/to/retained/cache
python tools/check_release.py --bundle .cache/release/qwenputer-evidence-v1.tar.gz
python tools/verify_all.py --tier 1 --bundle .cache/release/qwenputer-evidence-v1.tar.gz
```

The builder prints the size of every allowlisted entry. It writes the compressed bundle into ignored `.cache/release/` and updates tracked `release/bundle-manifest.json`. The manifest contains the archive hash, exact member list, sizes, and each file's SHA-256. Review and commit a manifest update deliberately. The verifier checks the archive before extraction, rejects links and unsafe paths, then checks every file before invoking a verifier. HTTPS URLs use the same pinned manifest. The manifest does not establish who produced the original outputs.

The bundle includes complete story and tiny-Qwen actual/reference traces, final RAM/registers, output bytes, chunk journals, restore metadata, packed ELF inputs, the actual pre-shutdown CPU-state checkpoint required by the Qwen recovery verifier, showcase local/reference and GPU/stock outputs, and MNIST test inputs. Frozen independent reference packages are Python source only. Six bundle-only JSON metadata copies redact local home-directory paths. Their manifest entries record both the original source SHA-256 and the distributed SHA-256, with a redaction count. The builder never modifies the retained inputs. No numerical, output, timing, script-binding, or verifier assertion changes. Native showcase executables are rebuilt with `clang` on the verifier's platform.

The tiny-Qwen program ELF necessarily embeds its quantized guest model. The trained float checkpoint is not needed by the verifiers and is excluded from the bundle; it is published separately on Hugging Face as [`Ayt1da/qwen-in-qwen-tiny`](https://huggingface.co/Ayt1da/qwen-in-qwen-tiny), with the same DROM image and the training log. The recovery `.pt` is recurrent CPU execution state, not guest training weights. The multi-GB training and quantization sweeps are excluded because the retained-output verifiers do not consume them. The TinyStories corpus and all Doom inputs and derived outputs are excluded. See [licence notes](../THIRD_PARTY_NOTICES.md) before redistributing the bundle.

## Match claims to evidence and commands

Commands below run inside `verify_all`'s disposable copy; direct research commands can require `.cache` and can overwrite historical reports. Use the release command for read-only verification of the clone.

| Claim | Retained evidence | Existing verifier used by release command |
| --- | --- | --- |
| CHIP-8 input-dependent gameplay and retained compact/stock/CUDA traces | `artifacts/compact-game*`, `artifacts/hf-*` | Tier 0: `tools/check_game.py` |
| Exact fused state comparisons and repeated bounded timings | `experiments/tiny-llm-ndot-fused/evidence/` | Tier 0: `check_evidence.py` in that directory |
| Fused executor timing and race record consistency, without game-output verification | `experiments/dmc-rom-fusion/evidence/`, `dmc-feedback-fusion/evidence/` | Tier 0: `check_evidence.py`, `check_diagnosis.py`; the Doom output claim remains SKIPPED |
| Corrupt checkpoint rejection before state copy | `experiments/checkpoint-integrity/evidence/cpu-restore.json` | Tier 0: `experiments/checkpoint-integrity/test_restore.py` with CPU tensors |
| Software NDOT ISA records and arithmetic | `experiments/neural-accelerator/isa-result.json` | Tier 0: pytest `test_isa.py`, including directed, seeded, and exhaustive arithmetic checks |
| 38-position story, literal state, contiguous chunk timing and historical sources | `experiments/tiny-llm-ndot-fused/evidence/story-validation-20261031.json` | Tier 1: `verify_story.py --source-revision 017c39c` with bundle inputs |
| 35,836,005-token tiny-Qwen run, power-off recovery, 16,384-token stock prefix | `experiments/tiny-qwen35/evidence/neural-validation.json` | Tier 1: `experiments/tiny-qwen35/verify_gpu.py --source-revision 017c39c` |
| Native/reference SHA-256, Life, tic-tac-toe, MNIST; 10,000-example integer classifier | `experiments/showcase/evidence/local-validation.json`, per-demo evidence | Tier 1: `experiments/showcase/verify_local.py` after native compilation |
| All showcase GPU tokens, literal RAM/registers, bounded stock prefixes | Per-demo `experiments/showcase/*/evidence/` | Tier 1: `experiments/showcase/verify_gpu.py --source-revision 017c39c` |
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

## Doom

Doom now lives in [qwenputer-doom](https://github.com/Ad1tyaNarayana/qwenputer-doom), including its article, verifiers and reproduction notes. No game data is distributed here. `dmc-rom-fusion`, `dmc-feedback-fusion`, `dmc-cuda` and `dmc-rv32` stay in this repository because the LLM runs use their executor and helper code. Their names record where the work began.
