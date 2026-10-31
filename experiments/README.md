# Experiment index

Use [REPRODUCE.md](../docs/REPRODUCE.md) for supported release commands. Per-experiment commands describe research runs and can require private caches or GPU hardware. Old `launch`, `collect`, `monitor`, and report-generation scripts are historical orchestration records, including outside the audit directories. Run current workers directly with explicit paths for new runs.

Doom-only experiments moved to [qwenputer-doom](https://github.com/Ad1tyaNarayana/qwenputer-doom). Historical records here may still name those paths, which remain in this repository's history.

## The neural CPU and accelerator

| Directory | Scope |
| --- | --- |
| [neural-accelerator](neural-accelerator/README.md) | Research implementation and retained results |
| [neural-accelerator-design](neural-accelerator-design/README.md) | Historical design record |
| [memory-map](memory-map/README.md) | Research implementation and retained results |
| [memory-map-cuda](memory-map-cuda/README.md) | Research implementation and retained results |
| [ram64](ram64/README.md) | Research implementation and retained results |
| [cuda](cuda/README.md) | Research implementation and retained results |
| [cuda-compact](cuda-compact/README.md) | Research implementation and retained results |
| [one-instr-design](one-instr-design/README.md) | Historical design record |

## Tiny language models

| Directory | Scope |
| --- | --- |
| [tiny-llm-rv32](tiny-llm-rv32/README.md) | Research implementation and retained results |
| [tiny-llm-train](tiny-llm-train/README.md) | Research implementation and retained results |
| [tiny-llm-neural](tiny-llm-neural/README.md) | Research implementation and retained results |
| [tiny-llm-ndot](tiny-llm-ndot/README.md) | Research implementation and retained results |
| [tiny-llm-ndot-fused](tiny-llm-ndot-fused/README.md) | Research implementation and retained results |
| [tiny-qwen35](tiny-qwen35/README.md) | Research implementation and retained results |

## Other guest programs and the Explorer

| Directory | Scope |
| --- | --- |
| [showcase](showcase/README.md) | Research implementation and retained results |
| [explorer](explorer/README.md) | Research implementation and retained results |

## Fused executor and shared helpers

| Directory | Scope |
| --- | --- |
| [dmc-rom-fusion](dmc-rom-fusion/README.md) | Batched ROM heads, used by the LLM runs |
| [dmc-feedback-fusion](dmc-feedback-fusion/README.md) | Single-kernel token feedback, used by the LLM runs |
| [dmc-cuda](dmc-cuda/README.md) | Runtime and state helpers, used by the LLM runs |
| [dmc-rv32](dmc-rv32/README.md) | ELF and build helpers, used by the LLM builds |

## Reviews and integrity

| Directory | Scope |
| --- | --- |
| [checkpoint-integrity](checkpoint-integrity/README.md) | Research implementation and retained results |
| [audit](audit/README.md) | Historical record |
| [blind-review](blind-review/README.md) | Historical record |
