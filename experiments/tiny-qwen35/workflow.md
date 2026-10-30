# Execution gates

Work runs inline. No commits or edits to the concurrent experiments.

1. Capture the installed Transformers 5.21 architecture and old integer outputs.
2. Define a versioned DROM graph and deterministic RAM allocator. Reject overflow.
3. Build one fixed runtime binary and swap only DROM images and prompt data.
4. Compare legacy IDs, all logit hashes, clipping counters, and output bytes.
5. Check Qwen recurrent operators against installed HF and native against RV32.
6. Train an actual HF Qwen3_5ForCausalLM in the existing training venv only after
   tasklist has no python.exe jobs. Stop training within 60 GPU minutes.
7. Measure float and integer greedy agreement, clipping, linked RAM, and traces.
8. Only after A through C pass, run autonomous neural continuation and stock HF
   prefix. Check every actual trace token and exported final state on the Mac.

Done means all eight gates have retained evidence. A scaffold, a random model,
a quality projection, or another program's neural run cannot pass later gates.
The throughput checkpoint uses actual reference trace counts divided by the
supplied 3735 trace tokens/s. It is a projection until this guest is measured.

The orchestrator added a shared GPU lease during training. Every later GPU
stage atomically acquires `C:/Users/Admin/gpu-lock`, writes `owner.txt`, and
releases its own lease on success or failure. Holds stay below four hours.
The already-running training lease has a detached cleanup monitor. The
accepted fused executor is from external commit `9365186`.
