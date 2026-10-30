# Explorer contract

The caller launches `python experiments/explorer/server.py` and opens localhost.
The server serves fixed assets and an allowlisted local recordings directory.
No network requests or GPU work happen while browsing.

Two designs were considered inline. A live Python model session would join all
views directly, but needs GPU access and risks obscuring state provenance.
The selected design uses immutable graph and recording bundles. Every value
has its source and exact token offset. Missing samples remain missing.

`graph.v1` contains layers, named IR heads, FFN terms, physical slot ownership
intervals, and exact stock row-emitter statistics. Layer detail files load on
demand. A batched Canvas projection draws all unit points. It preserves unit
identity when orbiting, zooming and opening the searchable list.

`recording.v1` contains interleaved generated token IDs and decoded PC values,
ELF symbols and code, timeline bins, and zero or more neural state snapshots.
Snapshots carry the number of generated tokens at observation time. No nearest
sample is silently presented as current. Reference comparisons are separate.
Application arrays use ELF symbol addresses and the guest packing metadata.

The recorder observes compact recurrent S through the existing RAM/register
observers. Fused GPU capture starts from boot or an actual bound checkpoint.
Reference tokens may validate the generated stream but never feed it. An
existing validated run can also be imported on CPU without running a GPU job.

Throughput checkpoint. Blocking work is schema and compiler grounding.
Graph export, recording import and UI implementation execute inline.
Shared writes stay in this explorer directory. The smallest safe decomposition
is one owner because this task is already the implementation sub-agent.
No nested delegation, upstream edits or commits are authorized.
