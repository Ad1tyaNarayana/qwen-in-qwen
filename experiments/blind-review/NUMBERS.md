# Article number checks

Computed from retained raw files where present. Wall time is a retained clock report, not independently measurable after the run. Quality records were recomputed, not generation rerun.

| Claim | Recomputed value | Article value | Evidence |
|---|---:|---:|---|
| scalar trace tokens | 16213774 | 16213774 | `.cache/checkpoint-integrity/actual-20261030a/scalar/actual.tokens.u32` |
| scalar mismatches | 0 | 0 | `.cache/tiny-llm/neural-stories/reference.tokens.u32` |
| scalar final RAM bytes | 65536 | 65536 | `.cache/checkpoint-integrity/actual-20261030a/scalar/actual-ram.bin` |
| scalar register copies | 3 | 3 | `.cache/checkpoint-integrity/actual-20261030a/scalar/actual-registers.json` |
| scalar positions | 5 | 5 | `.cache/checkpoint-integrity/actual-20261030a/scalar/actual.tokens.u32` |
| scalar generation seconds | 4361.424786300027 | 4361.42 | `.cache/checkpoint-integrity/actual-20261030a/scalar/chunks.jsonl` |
| scalar min/position | 14.538082621000092 | 14.5381 | `.cache/checkpoint-integrity/actual-20261030a/scalar/chunks.jsonl` |
| ndot trace tokens | 5423298 | 5423298 | `.cache/checkpoint-integrity/actual-20261030a/ndot/actual.tokens.u32` |
| ndot mismatches | 0 | 0 | `.cache/tiny-llm/ndot5-20261030/reference.tokens.u32` |
| ndot final RAM bytes | 65536 | 65536 | `.cache/checkpoint-integrity/actual-20261030a/ndot/actual-ram.bin` |
| ndot register copies | 3 | 3 | `.cache/checkpoint-integrity/actual-20261030a/ndot/actual-registers.json` |
| ndot positions | 5 | 5 | `.cache/checkpoint-integrity/actual-20261030a/ndot/actual.tokens.u32` |
| ndot generation seconds | 1446.6262496999807 | 1446.63 | `.cache/checkpoint-integrity/actual-20261030a/ndot/chunks.jsonl` |
| ndot min/position | 4.822087498999935 | 4.8221 | `.cache/checkpoint-integrity/actual-20261030a/ndot/chunks.jsonl` |
| story trace tokens | 53296563 | 53296563 | `.cache/tiny-llm/ndot-story38-20261030-actual/main/actual.tokens.u32` |
| story mismatches | 0 | 0 | `.cache/tiny-llm/ndot-story38-20261030/reference.tokens.u32` |
| story final RAM bytes | 65536 | 65536 | `.cache/tiny-llm/ndot-story38-20261030-actual/main/actual-ram.bin` |
| story register copies | 3 | 3 | `.cache/tiny-llm/ndot-story38-20261030-actual/main/actual-registers.json` |
| story positions | 38 | 38 | `.cache/tiny-llm/ndot-story38-20261030-actual/main/actual.tokens.u32` |
| story generation seconds | 14225.90152749946 | 14225.9 | `.cache/tiny-llm/ndot-story38-20261030-actual/main/chunks.jsonl` |
| story min/position | 6.239430494517307 | 6.2394 | `.cache/tiny-llm/ndot-story38-20261030-actual/main/chunks.jsonl` |
| qwen trace tokens | 35836005 | 35836005 | `.cache/tiny-qwen35/actual-covariance-20261030/main/actual.tokens.u32` |
| qwen mismatches | 0 | 0 | `.cache/tiny-qwen35/trained-program/reference.tokens.u32` |
| qwen final RAM bytes | 65536 | 65536 | `.cache/tiny-qwen35/actual-covariance-20261030/main/actual-ram.bin` |
| qwen register copies | 3 | 3 | `.cache/tiny-qwen35/actual-covariance-20261030/main/actual-registers.json` |
| qwen positions | 6 | 6 | `.cache/tiny-qwen35/actual-covariance-20261030/main/actual.tokens.u32` |
| qwen generation seconds | 10014.168106500378 | 10014.17 | `.cache/tiny-qwen35/actual-covariance-20261030/main/chunks.jsonl` |
| qwen min/position | 27.817133629167714 | 27.82 | `.cache/tiny-qwen35/actual-covariance-20261030/main/chunks.jsonl` |
| scalar trace tokens/s | 3717.5406649061574 | 3717.54 | `scalar raw count/chunk timer` |
| NDOT trace tokens/s | 3748.928239844086 | 3748.93 | `NDOT raw count/chunk timer` |
| scalar/NDOT ratio | 3.0148939902096714 | 3.0149 | `both complete raw chunk journals` |
| Qwen min/new token including prefill | 33.38056035500126 | 33.38 | `qwen raw chunk journal` |
| ff-full tokens | 6342933 | 6342933 | `.cache/doom-showcase/export-ff-full/actual.tokens.u32` |
| ff-full generation seconds | 1446.2763813000174 | 1446.28 | `.cache/doom-showcase/export-ff-full/chunks.jsonl` |
| ff-full tokens/s | 4385.699083530988 | 4385.7 | `.cache/doom-showcase/export-ff-full/chunks.jsonl` |
| ff-full us/token | 228.01381967932144 | 228.01 | `.cache/doom-showcase/export-ff-full/chunks.jsonl` |
| ff-full wall seconds | 1469.4132961000005 | 1469.41 | `.cache/doom-showcase/export-ff-full/report.json` |
| ff-full full chunks | 762 | 762 | `.cache/doom-showcase/export-ff-full/chunks.jsonl` |
| ff-full frame pixels | 76800 | 76800 | `.cache/doom-showcase/export-ff-full/frame-0001.rgb565` |
| ff-full min us/token | 227.80885009754394 | 227.809 | `.cache/doom-showcase/export-ff-full/chunks.jsonl` |
| ff-full median us/token | 227.91465454113924 | 227.915 | `.cache/doom-showcase/export-ff-full/chunks.jsonl` |
| ff-full max us/token | 229.3341552734862 | 229.334 | `.cache/doom-showcase/export-ff-full/chunks.jsonl` |
| unified-full tokens | 6342933 | 6342933 | `.cache/doom-showcase/export-unified-full/actual.tokens.u32` |
| unified-full generation seconds | 1693.106171399988 | 1693.11 | `.cache/doom-showcase/export-unified-full/chunks.jsonl` |
| unified-full tokens/s | 3746.3291476607073 | 3746.33 | `.cache/doom-showcase/export-unified-full/chunks.jsonl` |
| unified-full us/token | 266.92796083452055 | 266.93 | `.cache/doom-showcase/export-unified-full/chunks.jsonl` |
| unified-full wall seconds | 1715.4851406999987 | 1715.49 | `.cache/doom-showcase/export-unified-full/report.json` |
| unified-full full chunks | 762 | 762 | `.cache/doom-showcase/export-unified-full/chunks.jsonl` |
| unified-full frame pixels | 76800 | 76800 | `.cache/doom-showcase/export-unified-full/frame-0001.rgb565` |
| unified-full min us/token | 266.7166503906859 | 266.717 | `.cache/doom-showcase/export-unified-full/chunks.jsonl` |
| unified-full median us/token | 266.8337341308336 | 266.834 | `.cache/doom-showcase/export-unified-full/chunks.jsonl` |
| unified-full max us/token | 268.2981201171941 | 268.298 | `.cache/doom-showcase/export-unified-full/chunks.jsonl` |
| quality matched predictions | 291 | 291 | `covariance-quality.json per-prompt integer_records vs float_continuation` |
| quality denominator | 327 | 327 | `same` |
| quality prompts | 6 | 6 | `same` |
| Qwen stock prefix tokens | 16384 | 16384 | `.cache/tiny-qwen35/actual-covariance-20261030/stock/actual.tokens.u32` |
| Qwen stock prefix mismatches | 0 | 0 | `same, against reference trace` |
