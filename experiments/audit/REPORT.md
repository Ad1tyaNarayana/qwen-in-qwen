# Qwenputer audit at 4df34c6

Status: complete within the stated scope. All eight attacks have results, including the full-frame unfused state comparison, three new randomized guests, and a corrected sort/CRC guest. The corrected guest passes its complete fused execution and a 4,096-token stock prefix. The GPU reboot interrupted only its first build; that incomplete output was preserved. The shared GPU lock is released and no Python worker remains.

The original reviewer used Opus 5.5. The continuation uses GPT-6.1 Sol at medium effort, inline. This report preserves the original reviewer's scripts and records corrections to the audit harness below. Neither reviewer implemented the project changes under review. The continuation has read the handoff and prior results, so it is not a blind review.

## Verdicts within the tested scope

| Claim | Verdict | Evidence and limit |
| --- | --- | --- |
| C1: one fixed unified network | Holds with caveats | One fused executor instance ran 11 guests. Its non-state tensor digest stayed `e8ce2254…`. An independent hash of the materialized stock model is `0e515bbb…` over 15,292,471,808 BF16 bytes. This is evidence for the checked constructor and programs, not every possible backend or program. |
| C2: self-fed execution without host guest results | Holds with caveats | Worker inspection found generated output feeding the next transition. The audit GPU worker has no reference trace input. Full retained outputs agree with an independent interpreter. Boot setup and explicitly labelled reference-seeded diagnostic windows do use host-created initial state. Artifact comparisons cannot attest that an untrusted producer ran the reported computation. |
| C3: stock Qwen execution and compact equivalence | Holds with caveats | The audit's loaded model has the exact stock class, no instance overrides, and zero hooks. Installed model, generation, cache, and wrapper source files match their wheel RECORD hashes; assertions are enabled. Its weights hash matches the shared network. The corrected normal-exit guest also passes a 4,096-token autonomous stock prefix. The first fresh guest's 32,768-token autonomous stock prefix matches the project reference and the independent interpreter's observable records. It does not reach that guest's later fault. Existing project stock checks cover bounded prefixes; the full Doom and LLM runs use the compact executor. Universal equivalence is unproven. |
| C4: exact executor fusions | Holds with caveats | Existing isolated-window evidence checks recurrent state, not just outputs. The audit independently reproduced observable full-run results. Five audit guests now also have identical full recurrent-state digests in fused and unfused execution. The complete Doom frame now also matches bit for bit: 20,430,848 S words, 259,899 history words, next embedding, and token. Its complete trace hash equals the accepted fused trace. This is final-state equality for this run, not equality asserted at every intermediate transition or a universal proof. No claim about arbitrary graphs follows. |
| C5: published results and timings | Holds with caveats | 55 numerical checks recompute. Independent full-program checks match Doom, stories260K, tiny Qwen3.5, and five showcase traces after division-digit removal, plus literal final RAM and registers. Timing ratios retain the published measurement and projection scopes. There is no 24 fps result. |
| C6: model-independent guest runtime | Holds for the supported format | Six llama-style and Qwen3.5 fixtures use the same 11,732 executable bytes, hash prefix `51c2c9e1f39e657c`. The older 6,012-byte stories runtime is a different executable. The supported operators and memory limits do not support an unrestricted “any LLM” claim. |
| C7: changes to verification gates | Holds with caveats | Independent scalar/NDOT execution reproduces the output equality and the dead-register and dead-stack differences behind Gate 2. Integer counters change the circuit to address cached/chunked arithmetic differences. The retained memory-guard retry reclaims unused allocator blocks and rechecks the same 512 MiB threshold. Quantization used inspected prompts; 291/327 is not an unseen benchmark. |

## Findings

### Medium: stock and compact coverage must remain explicit

`experiments/tiny-qwen35/README.md:60` correctly limits the stock result to 16,384 CPU trace tokens and states that it completes no guest prediction. `experiments/doom-showcase/unified_worker.py:164` performs the full frame through `fr.run`, rather than through stock Hugging Face layers. The stock prefixes are evidence that the compiled network can execute in those layers. They are not evidence that an entire LLM output or Doom frame completed there.

Use “compiled Qwen3.5-architecture network,” identify the compact executor for full runs, and keep stock coverage adjacent to result claims. “Inside pretrained Qwen” is unsupported. The current Doom article already distinguishes these paths. The OpenAI continuation adds the complete-run executor and bounded stock-prefix scope to the tiny Qwen Explorer subtitle. This UI copy change does not alter neural execution or its evidence bindings.

Reproduction: inspect `experiments/neural-accelerator/stock.py`, `experiments/tiny-llm-ndot/stock_prefix.py`, and `experiments/doom-showcase/unified_worker.py`. The audit's `scripts/gpu_stock.py` independently records class identity, hook count, installed modeling source, and loaded weight hash.

### Medium: matching artifacts are not execution attestation

`experiments/showcase/verify_gpu.py:26` rejects a changed actual token, and its later checks bind RAM, registers, outputs, chunk hashes, source hashes, and stock output. All 13 corruption controls in `results/verifier-mutations.json` were rejected. Copying an unchanged reference trace over the identical actual trace was accepted.

That acceptance is expected for equality-based verification: identical bytes have no intrinsic execution provenance. A report flag or local hash is not independent proof of how bytes were produced. The code review and new reference-free GPU worker strengthen the execution evidence; they do not create remote attestation. No accepted run was found to substitute reference tokens.

Reproduction: `.venv/bin/python experiments/audit/scripts/verifier_mut.py`. Private retained artifacts are required.

### Medium: checkpoint restore does not checksum recurrent contents

`experiments/tiny-llm-ndot/worker.py:129` checks the checkpoint binding and tensor shapes, then copies saved recurrent tensors. It does not authenticate the tensor contents against a trusted checksum. The audit checkpoint experiment rejected a changed ELF binding, but accepted small bit changes in recurrent state, the next embedding, the output token, and unused memory. Every tested small perturbation left this guest's later trace and final RAM unchanged. These results do not establish that larger or later-used damage is harmless.

`results/ckpt.json` records the raw outcomes. The selected bit flip changes a low floating-point mantissa bit, so this is a restore-integrity test rather than a claim of changed guest semantics. Changing `fr.tok` alone need not change generation because the next embedding is retained separately. Unused or subthreshold state damage can escape output-based checks.

The tiny Qwen shutdown recovery verifier separately compares a retained checkpoint file's SHA-256 with its recorded recovery digest. Preserve that distinction: a file hash check, restore identity checks, and eventual output comparisons establish different facts. Do not claim every altered checkpoint is rejected by `restore`.

Reproduction: run `scripts/gpu_ckpt.py` against the retained SHA fixture with the GPU lock. The script records mutations separately and preserves the unmodified checkpoint as its control.

### Low: distinguish projections, timing comparisons, and complete runs

`results/claims.json` records 55 checked values and how they were derived. The approximately 3.4× figure divides an original-backend projection by a shorter measured workload. The approximately 5.3× figure compares projections for the earlier workload. The 1.86× comparison changes both network and backend. These are not interchangeable with a controlled measurement of executor-only speedup.

The Doom 2.105× ratio compares the same original circuit and 6,342,933-token trace. The unified Doom timing uses a different network. Its approximately 1.798× historical ratio is a cross-network comparison. The 38-position story's 6.2394 minutes per position is a separate complete-run measurement with a different denominator. Do not apply the shorter-workload ratios to it.

Reproduction: `.venv/bin/python experiments/audit/scripts/claims.py`. The script uses retained raw traces and chunk records where available. It checks arithmetic and evidence consistency, not independent wall-clock truth.

## Executed checks

- Independent RV32IM, NDOT, and Zba interpreter in `scripts/rvi.py`. It imports no project interpreter. `scripts/cmp_ref.py` imports the project interpreter only on the comparison side.
- Doom: 3,885,544 instructions; all non-division-digit trace records agree; final RAM and register files agree. Frame extraction gives RGB565 SHA-256 `6d60d53e30603c046bc61d6fdc1818cc9318f227064ec9b77285b1bda15d0db2`.
- Stories260K: 36,563,797 instructions and 2,463,616 NDOT operations; non-division-digit records, final RAM, and registers agree.
- Tiny Qwen3.5: 25,121,038 instructions and 2,257,920 NDOT operations; non-division-digit records, final RAM, and registers agree.
- SHA-256, Life, tic-tac-toe, MNIST NDOT, and MNIST scalar: independent full-program comparisons agree. Detailed counts are in `results/full-*.json`.
- One fused network instance: 11 original and mutated guests, including seven fault or boundary cases. Every compared observable record, RAM byte, and register agrees after correcting the audit interpreter's initial SP. A first fresh guest matches through an unintended fault; it is not counted as a successful normal-exit test.
- ELF scans did not find entire generated outputs or SHA digests baked into the showcase images. Text labels and tokenizer fragments are expected inputs. DROM mutations change SHA output, LLM text, and Doom rendering under independent execution. Selected fresh and SHA mutations also ran on the network.
- Gate 2: independently executed scalar and NDOT guests emit the same output and exit zero. Differences remain in x28–x31 and three dead stack words. This explains why literal equality between different implementations was the wrong gate. Literal final-state equality against each implementation's own reference remains required.
- Explorer: all 37 checked periodic Life and MNIST snapshots match independent RAM at their exact offsets. This audit result does not claim coverage of all 53 published periodic captures. Neural samples and trace reconstructions use distinct UI labels.
- Full-frame unfused check: all 6,342,933 Doom trace tokens have the same SHA-256 as the accepted fused trace. All 20,690,747 S and history words compare literally, with zero differing words. The final embedding and token also match; scratch output is zero. `results/deep-doom.json` retains the raw comparison.
- Corrected sort/CRC guest: 310,968 complete fused trace tokens, 195,341 guest instructions, 170 NDOT operations, normal exit, and `bad=0`. The complete project trace, independent RAM, and registers agree. Its 4,096-token stock prefix passes with loaded weight hash `0e515bbb…`. `results/safe-fused-vs-rvi.json` and `results/safe-stock-validation.json` retain the checks.
- Three fuzz guests: seeds 1101, 2202, and 3303 agree between the independent and project interpreters, including final registers and RAM. All three fused GPU runs now agree with the independent interpreter, including RAM and registers, and exit normally. Their complete trace hashes also match the project reference, including division-digit tokens. The 8,192-token autonomous stock fuzz prefix also passes with the shared loaded weight hash.

## Audit harness corrections

1. The first uploaded unfused worker lacked the import path for `worker`. It exited before computation. The current script includes that path. An empty output directory left by this failure can be reused; nonempty output directories remain rejected.
2. The first independent comparison compared a Python list with an `array.array`, incorrectly reporting unequal tokens. The corrected comparison checks the sequence contents. Raw final registers and RAM already agreed.
3. The independent interpreter initially set every register to zero. The boot contract sets SP to RAM end (`spec.SP_INIT`). Correcting that initial state resolves the otherwise unchanged x2 discrepancies in the seven assembly fault guests.
4. The original fresh guest's random RAM vectors could overwrite its active stack. It returned through corrupted x1 (`0xe1078000`) and faulted, with identical behavior in both interpreters and the network. `fresh/main_safe.c` reserves the stack and uses explicit ISA division operations for overflow and zero-divisor cases, avoiding C undefined behavior. Both interpreters now agree through normal exit: 195,341 instructions, 170 NDOT operations, 310,968 project trace tokens, and diagnostic `bad=0`. After reconnection, a fresh fused execution of this corrected guest passes all 310,968 project trace tokens, including division digits, and the independent interpreter's RAM and registers. Its 4,096-token autonomous stock prefix also passes, with the shared loaded weight hash. The interrupted build was preserved separately.

## Attack outcomes

| Attack | Outcome |
| --- | --- |
| 1: inspect worker and verifier data flow | Completed source inspection; no reference feedback or token replacement found. Mutation controls exercise the verifier. |
| 2: search images for answers | Completed scans and data mutations; expected labels and tokenizer data distinguished from computed results. |
| 3: independent interpreter | Completed full retained-program rechecks; internal QS/QD digit protocol excluded. |
| 4: never-run guest on fixed weights | Three new randomized guests complete normally in fused execution, with exact full project traces and independent RAM and register checks. One also passes an 8,192-token stock prefix on the shared weights. The original sort/CRC guest agrees through a guest fault and passes a 32,768-token stock prefix. Its corrected variant now completes normally in fused execution with exact full trace, RAM, and registers; its 4,096-token autonomous stock prefix also passes on the same loaded network weights. |
| 5: mutations | 13 saved-artifact controls rejected; data mutations produce predicted changed outputs. Eight recurrent-checkpoint controls completed; only the changed binding was rejected at restore. See the integrity finding. |
| 6: stock implementation and deep comparison | Loaded class, weights, and zero hooks inspected. The original fresh 32,768-token stock prefix passes. The 2,048-token default-precision check and a 4,096-token reference-seeded Qwen window at offset 20,000,000 also pass. The 4,096-token reference-seeded Doom window at offset 5,000,000 passes as well. |
| 7: at least 25 published numbers | 55 checks completed. Ratios and timing scopes are distinguished above. |
| 8: explorer provenance | Source inspection and 37 exact periodic RAM checks completed. Remaining recordings not independently checked here. |

## Scope and reproduction

Run from the repository root with the retained `.venv` and private `.cache` artifacts. GPU scripts accept `--repo`, `--upstream`, and explicit output directories. They require the existing Windows CUDA environment and the shared GPU lock. No packages were installed for this continuation.

`scripts/verify_full.py NAME ELF TRACE RAM REGISTERS` regenerates one complete guest and writes `results/full-NAME.json`. `scripts/cmp_gpu.py GPU_DIR RESULT_JSON` compares the audit worker's generated trace, RAM, and registers with independent execution. `scripts/doom_frame.py` independently reconstructs the frame. `scripts/code_identity.py` checks executable identity. `scripts/claims.py` recomputes selected published numbers.

The independent interpreter does not regenerate QS and QD internal division-digit tokens. These are stripped from GPU traces before comparison. Guest division results remain checked. Project verifiers separately compare every generated token, including QS and QD, against the project reference.

Not established: universal circuit equivalence, full-program execution through stock layers, unseen-model quality, arbitrary LLM compatibility, steady-state interactive Doom, or 24 displayed fps. Training was not rerun. Private raw artifact checks require the retained local files and cannot be reproduced from published JSON summaries alone.

The actual explorer manifest endpoint returns ten entries, including exactly one unified Doom manifest with the shared weight hash. `gallery()` renders one card per returned manifest, so the card was already included. `results/doom-gallery.json` records this endpoint check. No new browser rendering is claimed.

## Connection loss after the stock fuzz check

The 8,192-token stock fuzz output was copied and independently accepted before SSH became unreachable. The last collected remote status records `stock-fuzz exit 0` and `job3 lock released`. The control socket was then absent and a new connection returned “No route to host.” The PC later reconnected after a confirmed reboot at 20:23:01 local time. No audit worker survived. The interrupted `safe-fused` directory was preserved as `safe-fused-interrupted-20261030-1455`. After checking ownership, the stale audit lock was cleared and only the corrected-guest checks restarted from fresh boot state. Both completed and passed; `job4 lock released` was collected, the lock directory is absent, and process inventory shows no Python worker. The interruption is separate from numerical results. See `results/connection-interruption.json`.

`results/final-validation.json` collects all eleven completed GPU stages. `results/run-commands.json` records their exact argument vectors. `results/source-bindings.json` hashes the saved audit files and the UI clarification. Raw trace and state exports remain under `.cache/audit/remote`.
