# Phase 1 workflow

All implementation runs inline. No commits, upstream edits, installs, circuit changes, or reference-derived continuation state.

1. Build the five-position stories260K workload with the Phase 0 engine and unchanged memory map.
2. Stream the independent Mac reference, record output and position boundaries, and require native byte equality and clean HALT.
3. Verify a bounded fused GPU prefix and an actual-state checkpoint continuation. Launch the resumable full worker detached with logs and heartbeat.
4. While GPU generation runs, compare custom-model int16 and int8 KV quality and memory budgets. Measure stack use before reducing its reservation. Select context from measured fit and quality.
5. Require RV32/native byte equality and count instructions and trace tokens for the custom engine.
6. Run a bounded stock Hugging Face prefix sequentially on the GPU with a 30-minute budget.
7. Independently check the retained GPU trace, output bytes, checkpoint provenance, final RAM, reports, and private-artifact boundaries.

Done requires zero trace mismatches through HALT, GPU-derived output, a scoped stock-HF prefix report, and a measured custom model fit and quality report. A failed or unrun gate remains explicit. Timing is one end-to-end run, not a five-run optimization benchmark. Reference comparisons and IO are outside generation timers. The circuit remains the known limiter; no speedup claim is planned.

Throughput checkpoint: five stories positions should require about 16 million trace tokens and roughly one hour at the prior measured 4,380 tokens/s. Use 8,192-token validation chunks and 262,144-token checkpoints. The approved outer watchdog is four hours. Multi-hour execution is already authorized.
