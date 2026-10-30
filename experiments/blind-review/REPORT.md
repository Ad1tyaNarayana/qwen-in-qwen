# Blind adversarial review

Reviewed branch `tiny-llm`, HEAD **ee2dd8149731c73052611063adc90eaf2751556d**, on 2026-10-31. Pinned upstream: `ca13310e2f9f788d1dd3feeea8213c1b0c23913c`.

**The execution claims survive this review within their stated bounds. I found two reproducible verifier omissions and three overstrong Doom sentences, not a demonstrated false neural result.** Full language-model and Doom results remain compact-executor results; this review does not establish full stock language-model inference or full stock Doom.

I did not open `experiments/audit/`, either article SOURCES file, outside notes, or memories. These verdicts were written without the earlier audit. I have elected not to read it afterwards. No project fixes, commits or pushes were made. All local writes are under this directory; remote test outputs are under the authorized new `artifacts/blind-review-ee2dd814` directory.

## Verdicts

| Claim | Verdict | Basis and boundary |
|---|---|---|
| A. Fixed compiled Qwen3.5-architecture weights implement RV32IM plus dot product; programs change state | **Holds with caveats** | Constructor/emitter separate circuit weights from program/data tables. Fresh independently designed guest runs under the published loaded stock weight hash. This is a restricted machine with host-predecoded immutable code, fixed memory/alignment/I/O contracts, not pretrained Qwen or arbitrary privileged/self-modifying RV software. Exhaustive ISA correctness is not established. |
| B. Self-fed generation, no host guest results, actual-state checkpoints | **Holds with caveats** | Reviewed full-run loops feed network tokens back on GPU. References supply comparisons, stopping/observation offsets and initial program state, not intermediate execution. Fresh test disables reference execution methods. Checkpoints copy actual executor tensors. Artifact equality cannot prove remote provenance against an adversary who replaces all evidence. |
| C. Fused executor computes the stock network | **Holds with caveats** | Supported for this compiled circuit's valid-state contract and tested trajectories. Compact uses fp32 sparse/specialized operations; stock uses BF16 dense layers and ordinary `generate`. Existing bounded state/weight checks plus fresh identical 383-token stock/fused run support correspondence. No universal proof, arbitrary-state equivalence, or complete stock large-workload run was supplied or produced. |
| D. Language models and Doom ran with stated outputs/counts | **Holds with caveats** | Independently rechecked complete retained raw scalar, NDOT, 38-position story, tiny-Qwen and two Doom traces; outputs, RAM/register exports where available, and journals agree. Fresh GPU test supports execution mechanism, not independent regeneration of those long historical workloads. Some verifier omissions need correction; retained files themselves passed stronger checks. |
| E. Controlled scalar versus NDOT4 measurement is sound | **Holds with caveats** | Complete own-reference traces, same model bytes, constructor/executor sources, prompt and five-position budget verified. Raw retained timers give **3.0148939902096714×**. One retained complete trajectory per binary across checkpoint stages, not repeated independent trials or uninterrupted wall measurements. No variance estimate or general speedup claim follows. |
| F. Checkpoint restore rejects altered content | **Holds** | For altered saved content with unchanged integrity metadata, all reviewed executor restore paths verify before copying. Re-executed 35 corruption/legacy cases across four real entrypoints, with unchanged controls and destination-preservation checks. Content is not authenticated against a writer who also replaces the checksum, as explicitly disclosed. |

## Findings, ranked by severity

### 1. Medium — story verifier accepts an entirely absent timing journal

**Location:** `experiments/tiny-llm-ndot-fused/verify_story.py:54`–57, 63–73.

The verifier checks hashes of rows that happen to be present, but never requires a nonempty, contiguous, full-length journal. It also does not connect the journal's terminal accumulated time to the reported time. An empty `chunks.jsonl` passes the loop vacuously. Report time is checked against checkpoint *metadata*, not independently against the journal.

**Reproduced:** `verifier-controls.json` records `story_unmodified.accepted=true` and `story_empty_journal.accepted=true`. The negative control retains all trace/state files unchanged and replaces only the journal with an empty file. Historical source-hash reads are resolved from the documented `e7bd077` commit, rather than falsely comparing the old report against new source. No data assertion is removed.

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/blind-review/verifier_controls.py
```

**Impact:** This verifier cannot substantiate its journal/timing coverage if an export is incomplete. It does **not** invalidate the existing story: my separate `raw_checks.py` required contiguous positive chunk sizes, full trace coverage, every hash, monotonic cumulative timing and terminal equality, and passed the actual journal. The newer controlled-measurement verifier already performs stronger continuity/timer checks and is unaffected.

**Before publishing:** Require those checks in the story verifier and add an empty/deleted/reordered-row rejection control.

### 2. Medium — Doom verifier silently accepts missing final-state exports

**Location:** `experiments/doom-showcase/verify_runs.py:53`–55; the `verify()` function never reads `actual-registers.json`.

Final RAM comparison is conditional on the file existing. The same verifier handles unified Doom, whose retained result includes and claims a literal final-RAM comparison. Removing that file and the register export still returns successfully. Its returned result omits the literal-RAM flag, so it is not fabricating that flag; the problem is that required acceptance coverage silently shrinks.

**Reproduced:** `verifier-controls.json`, `doom_missing_final_state.accepted=true`. All remaining files are untouched links to retained artifacts. Reproduction command is the same as finding 1.

**Impact:** A deficient unified export can be accepted without the full intended state evidence. Current unified RAM bytes passed comparison to the Mac reference in my unmodified run. The GPU worker also asserts all register copies against its reference; this finding is about independent export validation, not proof that the current worker computed wrong state. Run 1 historically lacks a final RAM export, so requirements must be specific to run type rather than invented retroactively.

**Before publishing:** Make unified final RAM/register files mandatory, validate register contents, and reject deleted-file controls. Keep historical run 1's distinct coverage explicit.

### 3. Low — historical Doom ratio is described too causally

**Location:** `docs/articles/doom-on-qwenputer.md:129` (“clean before/after”; “The executor change accounts for that performance comparison”), also introductory framing at line 7.

The ratio is numerically correct, but the historical run was interrupted and had the disclosed warm-up-history problem. New versus historical runs do not hold all timing conditions constant. Same ELF, circuit and output trace do not isolate executor change as the sole cause of every difference in measured time. The alternating repeated window experiment is substantially better evidence of a fusion effect.

```sh
sed -n '43,55p;119,137p' docs/articles/doom-on-qwenputer.md
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/blind-review/raw_checks.py
```

**Before publishing:** Call 2.105× an observed historical generation-time ratio consistent with the controlled window results. Avoid implying a uniquely isolated causal effect from the two complete runs. The revised LLM 3.0149× paragraph already handles its scope much more carefully.

### 4. Low — narrow chunk timings cannot establish absence of GPU contention

**Location:** `docs/articles/doom-on-qwenputer.md:137`.

A steady competing workload can produce a narrow timing distribution. The raw 762-chunk ranges check out, but they establish stable within-run throughput, not “no GPU contention.” Process and telemetry observations would be separate evidence and would still have sampling limits.

```sh
sed -n '137p' docs/articles/doom-on-qwenputer.md
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/blind-review/raw_checks.py
```

**Before publishing:** Say no substantial within-run throughput variation was observed; describe any contention observations separately.

### 5. Low — Doom output checks are given more provenance power than they have

**Location:** `docs/articles/doom-on-qwenputer.md:139`, final sentence.

The output checks alone cannot distinguish a genuine execution from a byte-identical copied reference trace/frame. The reviewed data flow and fresh reference-disabled GPU execution strengthen the execution claim; ordinary hashes are not attestations. The LLM article explicitly acknowledges this at lines 140–142, but the corresponding Doom sentence lacks that qualification.

```sh
sed -n '139p' docs/articles/doom-on-qwenputer.md
sed -n '140,142p' docs/articles/tiny-llm-inside-qwen.md
```

**Before publishing:** Attribute the conclusion jointly to inspected execution data flow and output validation, and state the provenance limitation locally.

## What holds up in the data flow

- `tiny-llm-ndot/worker.py` builds a circuit from the ELF's initial state, resets full histories after capture, then calls `fr.run(amount)` before consulting the corresponding reference slice. The reference neither selects next tokens nor supplies guest arithmetic results. Trace length controls the budget; actual HALT is checked at completion.
- Fused `story.py` installs graph-derived ROM batching and feedback before recapture/reset. `runtime.fixtures()` does execute reference code for explicitly named reference-seeded *benchmark windows*. That path is not the complete story worker's boot-to-HALT loop. Treating those benchmark seeds as full-run contamination would be incorrect.
- `fusion.py` selects immutable eligible DROM heads using dependencies and lag checks. Its ordered apply retains per-sublayer history and uses explicit barriers. It does not implement a host RV interpreter. The upstream sparse kernels are specialized circuit evaluation, not general DeltaNet kernels; that distinction limits claim C.
- `neural-accelerator/stock.py:25` instantiates the real HF class on meta storage, streams emitted BF16 tensors into it, fills the padding layer, ties weights and checks CUDA placement. `boot_cache()` injects initial machine tables. The subsequent prefix worker uses ordinary stock `generate`; its streamer compares outputs and its stop criterion guards memory. It supplies no reference token as a generated input.
- `showcase/gpu_worker.py:20` copies a stock weight hash from evidence **after** checking the actual constructor manifest. This is constructor binding, not a measurement of compact buffers. In contrast, `showcase/gpu_stock.py` hashes the actual loaded BF16 tensors. The article now explains the distinction honestly; my fresh stock run independently reproduced that hash.
- Doom workers stop at reference-known observation boundaries but assert the generated BYTE marker, read descriptor/palette/pixels from actual recurrent RAM, and perform host palette conversion/placement. I found no reference-pixel injection or native scene renderer in that path.
- Flags such as `reference_tokens_used_as_inputs=False` and report hashes can be hand-written. They are descriptive metadata, not independent proofs. I traced the corresponding code paths rather than accepting those flags as evidence.

## Checkpoint coverage and limits

The recursive digest covers tensor bytes, dtype, shape, list position/dictionary key and saved scalar/container metadata. The integrity envelope itself is excluded, intentionally; it is not restored machine state. “Location” means path in the saved structure, not CUDA device identity or allocation address. Recurrent tables, histories, next input, token, bindings, offsets and timers are covered. Doom surface/coverage and saved scratch are covered too. Unstored scratch is asserted zero and reconstructed as zero.

The four restored entrypoints checked were:

| Entry point | Rejected mutation/legacy cases rerun |
|---|---:|
| `tiny-llm-ndot/worker.py` (also fused story/showcase/tiny-Qwen) | 9 |
| `tiny-llm-neural/worker.py` | 9 |
| `doom-showcase/showcase_worker.py` (also unified Doom) | 10 |
| `dmc-cuda/source/runtime.py` | 7 |

Unchanged controls restore exactly; rejected checksum cases leave destination tensors unchanged. Therefore these tests are not “everything throws” controls. The retained GPU report's seven changed checkpoints all identify checksum rejection, including unused DROM/RAM words; its unchanged control reaches exact final state. I read that GPU report and its test, but did not rerun those seven GPU cases.

I found no uncovered saved field in the reviewed execution restore paths. Training/calibration loads and read-only checkpoint exporters are separate from restoring the neural CPU. The old Doom restore has weaker schema/source binding than newer workers, and Doom checks shapes while copying, so a separately *resealed incompatible* checkpoint is not guaranteed atomic failure. That is outside the documented unchanged-checksum corruption guarantee and does not defeat F. Checksums do not bind checkpoint offsets to an external trusted execution transcript, stop an authorized writer recomputing them, or attest execution.

## Measurement and numeric recheck

`local_checks.py` ran the current controlled-measurement verifier with output paths redirected here. `raw_checks.py` independently checked every token and all saved output/state for scalar, NDOT4, long story and tiny-Qwen, and imposed complete contiguous journal coverage. The Doom verifier was rerun against both complete unmodified exports; raw chunk arithmetic was recomputed separately.

Scalar: 16,213,774 tokens / 4,361.424786300027 retained seconds. NDOT4: 5,423,298 / 1,446.6262496999807 seconds (see JSON for stored precision). Division yields **3.0148939902096714**. Both predictions are `[403,407,261,378,432]`, same packed model bytes and prompt, same constructor and fused source bindings. Chunk coverage is contiguous: 1,981 scalar chunks and 663 NDOT chunks. Prefix/restore stages make one complete retained trajectory each, not multiple benchmark replicates. Scalar's last restart is a terminal validation-only restore after missing reference-register metadata was repaired; no extra scalar generation enters the journal.

This does not measure setup, validation, checkpoint writes, restart gaps or end-to-end wall time. It does not establish variance, significance, thermal/order independence, performance on longer context, or accelerator speed on other models. A repeated counterbalanced experiment would be needed for those stronger statements; it is not required to report this accurately scoped single comparison.

**57 numerical checks passed**, documented individually in [NUMBERS.md](NUMBERS.md), with machine-readable values in `number-checks.json`. They include raw token counts, mismatches, final-state dimensions, inference positions, chunk times, rates, ratio, Doom pixel counts/ranges and the Qwen stock prefix. The 291/327 quality figure was recomputed from individual float and integer prediction records, not the aggregate summary. I did not rerun model training or float inference. The quality prompts are correctly described as inspected during quantization, not held-out quality evidence.

Every causal/scope paragraph and caption was assessed in [SCOPE_REVIEW.md](SCOPE_REVIEW.md). The important stock, quality, memory, first-frame, interruption and projection limits generally appear next to their results. Literal arbitrary-state/network equivalence and universal model compatibility remain unsupported and are not claimed in the revised text.

## Historical bindings are documented honestly

The checkpoint-integrity README explicitly names changed workers, says old reports will fail current-tree source comparisons, and gives **e7bd077** as the pre-change source revision. I checked **78 bound script entries** from historical story, tiny-Qwen, showcase and Doom reports against that commit: all matched; eight differ from current HEAD, as expected. The unmodified story verifier also passed with source-hash reads resolved at that documented revision and all data checks intact.

I did not rewrite old evidence hashes. Later source edits do not invalidate source-bound historical observations. Reproduction still requires the retained private raw assets/checkpoints and the stated software environment; a public checkout alone does not contain every large trace or Doom asset. I verified source reproducibility and available local raw evidence, not every historical launcher on a newly provisioned machine.

## New GPU experiment

See `gpu_test.py`, `gpu-summary.json`, `gpu-source-bindings.json`, and `gpu-results/`.

I hand-encoded a new 31-word RV32 guest: NDOT4 on signed weights `[-128,127,-7,19]` and signed activations `[-32768,32767,-12345,23456]` with accumulator −17, followed by a branched 32-iteration reflected CRC recurrence, a RAM store and four byte outputs. This exercises signed extrema, multiple result tokens, a data-dependent recurrence, a backward branch, memory and output. Expected values come from independent Python integer arithmetic: dot **8,887,775**, final CRC **4,282,289,376**, bytes **`e0 8c 3e ff`**.

The project loader creates initial program state only. `ref.Machine.record/run` and `reference.Machine.record/run` are replaced with raising stubs before circuit execution. The test reads no reference trace or expected-output fixture. The oracle is used only after generation; no guest result is injected.

| Result | Fused | Ordinary stock generate |
|---|---|---|
| Status | Pass | Pass on retry |
| Generated tokens through HALT | 383 | 383 |
| Output | `e08c3eff` | `e08c3eff` |
| Trace SHA-256 | `9770436ed89a4f712f1397ecceb4abfef67771989cf82dc4b7709e704a1be04b` | Same, bytes compared literally |
| Independent final RAM/register result | Exact RAM word and all three x3 copies | Output checked; stock cache not independently decoded |

The bounded stock invocation completed the whole small guest; this does not extend stock coverage to full LLM inference or Doom. Loaded stock weights hash to **0e515bbb3109997a130c903deb40dce3447f8991e15b6637e0071c71f8b6e07d**. All parameters stayed BF16 on CUDA; no offload. Allocator cap was 15 GiB, peak allocation 14,422,660,608 bytes, peak reservation 14,430,502,912 bytes, minimum guard-observed CUDA free memory 1,205,047,296 bytes. Guard threshold was 512 MiB; a separate nvidia-smi sample also remained above it. The stock environment used HF's fallback PyTorch recurrent/convolution implementations; no packages were installed.

Initial stock attempt ended with an incomplete `running` report, no exit JSON and no Python traceback; it is retained and **not counted as a pass**. No process remained and the lock had been released. I reacquired the atomic lock and retried in a fresh directory with cmd-level log redirection inside the detached PowerShell wrapper. Native stderr becoming a PowerShell terminating error is a plausible harness explanation, not a proven diagnosis. Retry exited zero. GPU lock release was checked. All runs stayed within the approximately 90-minute budget. Eleven relevant remote source hashes match the reviewed local HEAD files.

To repeat, check `ssh -O check gpu`, atomically acquire `C:\Users\Admin\gpu-lock` and set its owner; tar-sync `gpu_test.py` into a **fresh** `artifacts\blind-review-*` directory. Use the existing remote `.venv` to run the script with `--mode fused` and `--mode stock`, `--repo`, `--upstream`, and fresh distinct `--out` paths. `runner.ps1` and `retry-runner.ps1` preserve the exact detached commands used; change only the output label for repetition. Release the lock in `finally`. The scripts here preserve failures as well as successes.

## Executed versus inspected

**Executed:** fresh GPU fused and stock guest; complete retained raw trace/output/state comparisons for four LLM workloads; both Doom export verifiers; independent journal/hash/numeric checks; 35 CPU checkpoint negative cases; story empty-journal and Doom missing-state negative controls; 78 historical script bindings; remote/local source checks.

**Read, not rerun:** long historical GPU workloads, full program-pack GPU runs, counter diagnosis, full per-token fusion-state/race suite, GPU checkpoint mutation suite, model training/float quality generation, native 64-position software gate, and continuing Doom reference/profiler runs. Read all requested workers/verifiers, stock construction, fused adapter/planner, pinned execution kernels/feedback, relevant guest runtime/packing and design sources.

**Not established:** universal CPU or fused/stock correctness, stock completion of large guests, unseen-prompt model quality, continuing neural gameplay, repeated full-run performance distributions, or trusted remote execution attestation. No evidence discovered requires retracting the complete compact-executor outputs or the scoped 3.0149× measurement.

Before publication, close findings 1–2 and narrow the three Doom statements in findings 3–5. Preserve the existing adjacent limitations and the historical source records. Do not describe this bounded review as proof of all possible programs or states.
