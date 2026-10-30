# Blind review working notes
Target HEAD ee2dd8149731c73052611063adc90eaf2751556d, branch tiny-llm.
No audit directory or SOURCES article notes opened.
- Read both articles, main worker, fused runtime/story adapter, stock model emitter/prefix worker, showcase worker/verifier, tiny-Qwen verifier, Doom workers/verifier, checkpoint implementation and tests.
- No reference-result input found in full-generation loops. Reference-seeded benchmark windows are explicit.
- Integrity checksum recursively covers saved tensors/metadata and verifies before copying. Shape checks in Doom are interleaved with copying, relevant only to independently sealed incompatible files, not unchanged-checksum corruption.
- Historical source staleness explicitly documented with e7bd077 reproduction commit; must check the hashes at that commit.
- Potential article overclaims: narrow chunk ranges cannot establish absence of GPU contention; historical vs fresh Doom timing cannot isolate executor as the only cause.
- GPU lock acquired successfully; no Python workers listed. New dot/CRC guest prepared with independent arithmetic oracle, reference execution disabled.
- Controlled measurement revalidated with full retained raw traces (16,213,774 and 5,423,298 tokens), own references, RAM, registers, journal hashes, timers, source bindings. Ratio 3.0148939902096714. No gaps/overlap in journals.
- Independent raw pass checked long story and Qwen traces, outputs, RAM, register copies, and contiguous monotonic hashed journals. All pass.
- Negative control: story verifier accepts completely empty chunks.jsonl; source lookup resolved at documented e7bd077, with all data checks intact.
- Negative control: unified Doom verifier accepts missing final-gpu-ram.bin and actual-registers.json. It does not claim literal RAM comparison in this deficient output, but returns successfully without enforcing the required evidence.
- CPU corruption tests across all four real restore entrypoints pass; checksum errors precede copies.
- Initial stock attempt ended without an exit report or Python traceback. Lock was released and no process remained. Retained as an incomplete harness attempt; retry uses cmd redirection inside detached PowerShell to avoid native stderr becoming a terminating PowerShell error. Cause not established.
