# Release validation, 2026-10-31

Code and bundle were tested at `296acb26615a19ea36df6288da2dc482b54783c6`. Subsequent changes record documentation and these results. No push, repository creation, release publication, or GPU access occurred. The branch is `oss-release`.

## Fresh clone and environment

A fresh full-history local Git clone and a fresh Python 3.13.4 venv under `/tmp` installed `requirements.txt`. The host was macOS 26.7.1 on Apple M4 Pro with 24 GiB unified memory. For all three positive runs, the original working `.cache` was renamed and its old path pointed at a nonexistent directory. The bundle was copied outside it first. The original cache was restored afterward. The clone stayed clean.

Commands were `python tools/verify_all.py --tier N --json /tmp/result.json`, with `--bundle /tmp/qwenputer-evidence-v1.tar.gz` for tiers 1 and 2. Times include verification setup and temporary copies, and exclude dependency installation. These are observed verification times, not performance benchmarks. Counts refer to claim rows; individual checks can contain many assertions.

| Tier | PASS | FAIL | SKIPPED | Seconds | Exit |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 12 | 0 | 13 | 13.052 | 0 |
| 1 | 22 | 0 | 5 | 70.921 | 0 |
| 2 | 23 | 0 | 4 | 85.359 | 0 |

The tier-0 release-boundary tests include nine passing cases. ISA and IR files both collect under pytest. The full ISA gate runs; the long full neural IR gate does not run without `--full-ir`. See [machine-readable per-claim results](validation.json) and [reproduction commands](../docs/REPRODUCE.md).

## Bundle

`qwenputer-evidence-v1.tar.gz` is 238,005,501 compressed bytes, with 506 entries and 1,093,024,684 expanded bytes. It is ignored under `.cache/release/`; [the manifest](bundle-manifest.json) is tracked. It contains story and tiny-Qwen actual/reference traces, final state, chunk journals, the required actual recovery checkpoint, showcase reference/native inputs and GPU/stock exports, MNIST inputs, frozen reference source, packed guest ELFs, and licence notices. Explorer assets are already tracked; their private dependencies are supplied by these fixtures.

Six distribution-only JSON metadata copies redact home-directory paths. Both original and released SHA-256 values are recorded. Original cache metadata still matches all six source hashes. A text scan found no home paths, email addresses, or tested private LAN address patterns in distributed text. A byte-level scan afterward found that all nine bundled `engine.elf` files embed compiler debug paths under the author's home directory. They are kept byte for byte because the ELF hashes identify the executed programs. They contain no credentials. Git history also retains the same path, a machine name and one private LAN address in evidence bound by hash or commit. The privacy transformation changes no checked numerical field, trace, output, timing, or source binding. Both full tiers pass afterward.

Doom data, traces, state, pixels, frames, and images are absent. Large training/quantization sweeps and separate trained PyTorch guest checkpoints are omitted. The tiny-Qwen ELF embeds the quantized guest weights needed to identify the retained program. The recovery `.pt` is saved compiled CPU execution state.

## Negative controls

All controls ran in a separate disposable clone, and all produced the required nonzero failure. The bundled-output control changes one byte of the actual story output and repacks the archive. A separate control authenticates the original bundle, changes the same output byte, then invokes the unchanged story data assertions directly. The tracked-evidence control changes one byte in a retained tiny-Qwen evidence JSON. The WAD and absolute-link controls stage fake entries only in that disposable clone. All tracked files are restored afterward.

| Control | Observed | Exit | Seconds | Failure reason |
| --- | --- | ---: | ---: | --- |
| wad | FAIL | 1 | 0.939 | game data or pixel dump |
| symlink | FAIL | 1 | 0.914 | absolute symlink |
| evidence | FAIL | 1 | 13.010 | Tracked evidence SHA-256 mismatch |
| bundle | FAIL | 1 | 13.052 | Bundle SHA-256 differs from tracked manifest |
| real-verifier | FAIL | 1 | 0.185 | AssertionError |

## Unchecked work and owner decisions

Tier 0 skips eight bundle-dependent claim rows plus stock CPU regeneration, full IR, Doom outputs, float quality, and full GPU/training work. Tier 1 removes the eight bundle skips and adds two authentication/policy passes; stock CPU regeneration remains skipped. Tier 2 also runs the bounded stock CPU workload. Its remaining four SKIPPED rows are full IR, Doom outputs, float quality, and full GPU/training work. SKIPPED rows are not passes.

No full GPU execution or training was rerun. The inspected-prompt 291/327 quality result remains historical evidence. Doom output validation requires the user's own IWAD and locally regenerated private outputs; public readers can inspect hashes and timing arithmetic only. Hashes do not establish provenance. Historical full-run hours and bounded stock coverage are documented in the reproduction guide.

The GitHub workflow later passed tier 0 on a standard Linux CPU runner: 12 PASS, 0 FAIL, 12 SKIPPED. Tiers 1 and 2 have not run on hosted CI. Linux CUDA is untested. Before public hosting, the owner must choose the bundle location and review model/dataset redistribution notices, including TinyStories and MNIST attribution. No public download URL is claimed.

`git diff --stat tiny-llm -- '*/evidence/*' '*/data/*' 'docs/articles/*' '*/REPORT.md'` is empty. Historical reports, articles, evidence bytes, source hashes, and cited Git history are unchanged. The 24 absolute control symlinks were removed; runtime controls still construct their temporary inputs.

## Commit hashes

Commit messages were edited after this validation to remove a co-author trailer, which changed 28 commit hashes, including the hash named above. File trees, authors and dates are unchanged, checked commit by commit. [The commit map](commit-map.json) lists every old and new hash. Evidence files and reports keep citing the old hashes. `experiments/blind-review/local_checks.py` and `verifier_controls.py` are historical records and still name the old hash, so run them against the map.

## Doom split

Doom-only experiments, the Doom article and its figures moved to the public [qwenputer-doom](https://github.com/Ad1tyaNarayana/qwenputer-doom) repository. Their entries left `tracked-evidence.json`. The tier tables above predate this. The Doom output row is gone, the three executor rows were relabelled, and Doom worker cases left the restore and export controls. A fresh clone afterward gave tier 0: 12 PASS, 0 FAIL, 12 SKIPPED; tier 1: 22 PASS, 0 FAIL, 4 SKIPPED; tier 2: 23 PASS, 0 FAIL, 3 SKIPPED.

## Publication

The bundle pinned by `release/bundle-manifest.json` (SHA-256 `638764c2…c4b87b9c`) is hosted as a GitHub release asset: [evidence-v1](https://github.com/Ad1tyaNarayana/qwen-in-qwen/releases/tag/evidence-v1). The trained tiny-Qwen float checkpoint, tokenizer, DROM image and training log are published on Hugging Face as [`Ayt1da/qwen-in-qwen-tiny`](https://huggingface.co/Ayt1da/qwen-in-qwen-tiny). Its `model.safetensors` SHA-256 is `b7929b5d…f87e626c`, matching `experiments/tiny-qwen35/evidence/calibration.json`. Licence notes for TinyStories, stories260K and MNIST are in `THIRD_PARTY_NOTICES.md`. The evidence numbers above are unchanged by publication.
