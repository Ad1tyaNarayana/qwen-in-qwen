# Checkpoint integrity and controlled stories measurement

This follow-up acts on the audit without editing `experiments/audit/`. No executor arithmetic, circuit construction, or model weights change.

## Checkpoint format correction, 2026-10-30

`../tiny-llm-ndot/checkpoint_integrity.py` adds versioned SHA-256 content integrity. Save describes every nested tensor by its location, shape, dtype and raw-byte SHA-256, then hashes the complete description and saved scalar/container metadata. Restore checks the root before copying any saved state into the GPU. This protects recurrent S tables, lag histories, next embedding, token, offsets, timers, bindings, and any additional saved tensor or metadata field. Cleared scratch that is reconstructed as zero retains its pre-save assertion. Saved Doom observation buffers and scratch are included too.

The check is used by:

- `experiments/tiny-llm-ndot/worker.py`, also reused by the fused story, tiny Qwen3.5 and showcase GPU workers.
- `experiments/tiny-llm-neural/worker.py`.
- `experiments/doom-showcase/showcase_worker.py`, also reused by `unified_worker.py`.
- `experiments/dmc-cuda/source/runtime.py`, used by the historical full workers.

Old checkpoints without integrity metadata intentionally fail with an explicit legacy-checkpoint error. Do not add a checksum to an unverified historical file to make it load. The checksum detects corruption; a hostile producer able to rewrite both content and checksums can create a consistent file. This is not execution attestation.

`test_restore.py` exercises these real save/restore entry points with CPU tensors. Its retained [report](evidence/cpu-restore.json) checks tensor and metadata mutations and verifies that checksum failures leave the destination state unchanged.

`gpu_ckpt.py` is copied from the audit's original scenarios, outside the immutable audit directory. Only rejection reporting, an exact final-register comparison, and acceptance assertions are added. [GPU results](evidence/gpu-mutations.json) reject the changed binding, live recurrent register, next embedding, token, lag history, unused DROM row and unused RAM row at restore. The unchanged checkpoint continues to the end with zero token mismatches, literal final RAM and all register copies exact. [Source bindings](evidence/mutation-bindings.json) identify the audit original and the copy.

```sh
.venv/bin/python experiments/checkpoint-integrity/test_restore.py
```

## Controlled measurement

The scalar fixture is `.cache/tiny-llm/neural-stories`; NDOT4 is `.cache/tiny-llm/ndot5-20261030`. Both have prompt `Once`, IDs `[1,403]`, five inference positions including prefill, identical packed model bytes, and predictions `[403,407,261,378,432]`. Each binary has its own independent reference trace, final RAM and registers. No cross-binary dead-state exception applies to neural validation.

Both full runs use the unified W=4 compiled Qwen3.5-architecture network and the same compact fused executor. `showcase/gpu_worker.py` binds the generic constructor to the common stock-emitted BF16 identity `0e515bbb3109997a130c903deb40dce3447f8991e15b6637e0071c71f8b6e07d`. This is a stock-weight identity, not a hash of compact fp32 buffers. Stock generation covered bounded prefixes only; it is not used for these complete runs.

The detached batch runs the GPU corruption test, then scalar prefix/restore/full, then NDOT4 prefix/restore/full, sequentially. Each run begins at boot and retains actual GPU checkpoints at 65,536 and 65,792 tokens, then at most 262,144 transitions apart. The worker compares every generated token online, but reference tokens never enter feedback or recurrent state. A watchdog bounds each process. The launcher checks that no other Python process is running before each stage. The usual venv Python shim and its child are one worker, not two jobs.

Generation time sums successful generation calls, including graph replay, feedback, synchronization, copies and CPU transfer. It excludes setup, comparison, IO, checkpoint writes and restart gaps. A ratio will be published only after both complete raw traces pass `verify.py` on the Mac. This is one complete run per binary, not a repeated benchmark. It measures the accelerator's effect on this guest using one executor, not an executor-only speedup.

Current run commands and remote path are in [launch.json](evidence/launch.json). Large traces and checkpoints stay private under the new `artifacts/checkpoint-integrity-20261030a` directory and `.cache/checkpoint-integrity/`.

Before a fresh launch, require `ssh -O check gpu` to pass. Acquire the lock atomically with `ssh gpu "mkdir C:\Users\Admin\gpu-lock"`, write this task's owner to `owner.txt` through encoded PowerShell, then run `launch.py --label FRESH`. The launcher requires that owner name. It tar-syncs named files and detaches with `Invoke-CimMethod Win32_Process Create`. The detached parent's `finally` releases the lock using `rmdir /s /q`, including ordinary worker failure. A local launch failure releases it too. Use the existing `C:/Users/Admin/qwenputer-games-work/.venv`; install nothing.

If a reboot interrupts the batch, preserve its files. When SSH is available again, inspect ownership and processes before acquiring a new lock. `launch.py --label SAME --resume` skips completed stages and restores this run's own saved GPU tensors for the incomplete stage. Remove its generated scratch package before rebuilding, as the launcher does. The trace is truncated to the saved checkpoint offset. If a crash left chunk rows beyond that offset, preserve the original journal and trim only those rows before resuming. Never change the checkpoint's offset/timer/binding, and never synthesize it from a reference. No wall-time claim may include discarded post-checkpoint work as retained computation.

While the batch runs, `status.py` is read-only. A failed `ssh -O check gpu` stops GPU work. Collect only the small mutation reports with `collect.py`. After completion:

```sh
.venv/bin/python experiments/checkpoint-integrity/collect.py --complete
.venv/bin/python experiments/checkpoint-integrity/verify.py
.venv/bin/python experiments/checkpoint-integrity/publish.py
.venv/bin/python docs/articles/figures-llm/make_figures.py
```

## Power-off records

[Windows event records](evidence/windows-reboots.json) queried System event IDs 41, 1074 and 6008 over the preceding four days. The two recent interruptions have ID 1074 records at `2026-10-30T06:24:34Z` and `2026-10-30T14:48:59Z`. Both record `StartMenuExperienceHost.exe` initiating power off, under Home and Admin respectively, with reason “Other (Unplanned)”. No event 41 or 6008 was returned by that query. This supports a Start-menu-requested shutdown, not a demonstrated GPU crash. It does not identify who operated the account or why. Raw messages and XML are retained in `evidence/windows-reboots.txt`. PC settings were not changed.

## 2026-10-31 reference export repair

The scalar worker generated all 16,213,774 tokens exactly, then stopped at `meta['final_registers']`: the older scalar reference writer had never exported that field. This was a harness error, not a failed numerical comparison. A fresh local software replay in `.cache/checkpoint-integrity/scalar-reference` reproduced the complete trace and final RAM literally. The writer now exports final registers, PC, anchor and counter, and the launcher preflights required reference fields before GPU work.

Only the missing comparison fields were added to the retained scalar fixture metadata. No reference state entered the GPU. The complete actual GPU checkpoint was restored at offset 16,213,774 to finish validation without generating another scalar token. The original failed report, exit and logs are preserved with `initial-failure` names. Generation time remains the sum of the original generation chunks; the metadata-repair gap and validation-only restart are excluded. [Repair evidence](evidence/scalar-reference-repair.json) binds the new reference writer and exact replay.

## Accepted controlled measurement, 2026-10-31

[Independent Mac validation](evidence/controlled-measurement.json) passes for both complete executions. The scalar run retained 4,361.4247863 generation seconds for 16,213,774 trace tokens, at 3,717.5407 tokens/s and 14.5381 minutes per inference position. NDOT4 retained 1,446.6262497 seconds for 5,423,298 tokens, at 3,748.9282 tokens/s and 4.8221 minutes per position. Scalar/NDOT4 generation time is 3.0149×. This is one complete run each, on the same unified W=4 compiled Qwen3.5-architecture network through the compact fused executor. Stock generation remains bounded-prefix evidence only.

Both actual ID sequences are `[403, 407, 261, 378, 432]`. Every token, final RAM byte, and register in all three copies matches each binary's own reference. All timed chunks and source bindings pass. Scalar's terminal validation-only restore generates no new tokens and does not change its retained generation sum. [Corrected files](evidence/corrected-files.json) identify the dated README, article and manifest updates. The comparison is not an executor-only gain or a ratio applicable to the separate longer story.

## Historical source bindings

This change edited `experiments/dmc-cuda/source/runtime.py`, `experiments/doom-showcase/showcase_worker.py`, `experiments/tiny-llm-ndot/worker.py`, `experiments/tiny-llm-neural/worker.py` and `experiments/tiny-llm-neural/reference.py`. Runs recorded before this change bind the SHA-256 of those files as they were at the time, including `experiments/dmc-cuda/SHA256SUMS.json`, `experiments/tiny-llm-neural/SHA256SUMS.json` and the reports of the Doom, stories260K, tiny Qwen3.5 and showcase runs. Those records are unchanged and remain correct for the source that produced them, but a verifier that compares them with the current working tree reports a mismatch.

To reproduce a pre-change verification, check out commit `e7bd077`, which holds the bound versions. Do not rewrite historical hashes to match the current files. Runs made under this directory bind the updated files.
