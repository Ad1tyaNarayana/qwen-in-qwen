# TinyStories model for a small RISC-V target

This run trains a standard llama2.c Transformer for later int8 deployment. The exported files load in the original native C runner. Integer-only inference on the target remains untested.

## Data and tokenizer

The corpus is [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), using the `TinyStories_all_data.tar.gz` URL embedded in upstream `tinystories.py`. `prepare.py` downloads that same archive with HTTP error handling, a disk reserve, safe extraction of JSON members, and hashes. It replaces the upstream download and shell extraction stage. The archive contains 50 JSON shards totaling 6,973,633,727 bytes. Its compressed size is 1,608,001,638 bytes and SHA-256 is `26cf7605aca15bc4ea6fa637256400d9d01317b28ed296172b2d1dd160cd7699`. `dataset.json` records every shard hash.

The patched `tinystories.train_vocab(1024)` trains SentencePiece BPE with byte fallback and identity normalization on shards 1 through 10. Upstream includes shard 0 in vocab training. This run excludes it because shard 0 is validation. Shards 1 through 49 train the language model. The patched `tinystories.pretokenize(1024)` uses four workers and consumes executor results so worker errors propagate. Stories are stripped, prefixed with BOS, and have no EOS, matching upstream. Training uses uint16 token files.

Tokenizer SHA-256 values are `74dd6b293cb95be06b604276613be9b76120fe86fa6bc5d7e3298d79ed95aff2` for `.model` and `c41549f4ca412eab26549ee8e22d8e8527802850a2ff1b13b8d1d23c75d1f3e1` for the native `.bin`. The exported tokenizer occupies 12,401 bytes. The binary tokenizer comes from upstream `Tokenizer.export()`.

## Architecture and memory

The model has dim 192, 6 layers, 12 query heads, 1 KV head, head dim 16, KV dim 16, hidden dim 512, vocab 1024, and training context 256. It shares the embedding and classifier. It uses upstream RMSNorm, RoPE, SwiGLU, and causal attention with no added layers.

The proposed 8-layer, KV-dim-32 model needs `8 * 2 * 32 * 128 = 65,536` KV bytes, which exceeds the limit. Six layers and one 16-wide KV head give `6 * 2 * 16 * 128 = 24,576` bytes at the requested inference context. The wider dim 192 uses the remaining weight budget while staying within the activation width limit. The quality cost of 12-to-1 query-to-KV sharing has not been isolated in an ablation.

Matrix weights total `1024 * 192 + 6 * (2 * 192 * 192 + 2 * 192 * 16 + 3 * 192 * 512) = 2,445,312` parameters. Norms add `(2 * 6 + 1) * 192 = 2,496`, giving **2,447,808 parameters**.

For group-64 symmetric int8 quantization, matrices occupy 2,445,312 bytes and float32 per-group scales occupy `2,445,312 / 64 * 4 = 152,832` bytes. Float32 norms add `2,496 * 4 = 9,984` bytes. The v2 header adds 256 bytes. The measured Q8 file is **2,608,384 bytes**, below `3 * 1024 * 1024 = 3,145,728` bytes. It leaves 1,585,920 bytes in a 4 MiB ROM before vocabulary and tables.

The requested legacy v0 file holds float32 weights and RoPE tables. Its measured size is 9,807,644 bytes. It is an interchange and native sanity-check artifact, not the deployment-sized image. The additional v2 file verifies the int8 storage calculation. Its scales and norms still use floating point. A target integer format may retain these 32-bit slots with fixed-point values, but that conversion is not implemented here.

The KV calculation covers int8 K and V payload only. Scale metadata, working activations, attention scratch, accumulators, stack, tokenizer structures, and alignment need a complete target RAM plan. The upstream runner uses float KV, which is four times larger. A 256-token int8 KV cache would also exceed the requested 24 KiB limit, so target inference must cap context at 128.

## Training configuration and execution

Source is llama2.c commit `350e04fe35433e6d2941dce5a1f53308f87058eb`. `llama2-windows.patch` records modifications to copies under `.cache/tiny-llm/custom/source`; the downloaded checkout remains unchanged. All seven PC source-file hashes match the inspected Mac copies, recorded in `source-hashes.json` and `pc-source-hashes.json`. The patch excludes validation text from vocab training, limits tokenizer workers, exposes preprocessing errors, supports current PyTorch checkpoint loading, limits CPU threads, logs losses and peak allocation, stops on a wall-clock limit, and evaluates and saves at the final update without an extra optimizer step.

The PC uses NVIDIA GeForce RTX 5070 Ti, Python 3.14.2, PyTorch 2.14.1+cu130, and CUDA 13.0. BF16 support was checked at runtime. The new venv is `C:\Users\Admin\tiny-llm-train\.venv`. The verified neural-CPU venv was not modified. Packages came from the official CUDA 13.0 PyTorch wheel index and PyPI. [PyTorch installation guidance](https://pytorch.org/get-started/locally/) describes the supported native Windows setup. `requirements-lock.txt` records exact installed versions.

`train_config.py` contains overrides and `training-config.json` contains the resolved upstream configuration. The run uses batch 32, context 256, accumulation 1, BF16 autocast, FP32 master weights, fused AdamW, peak learning rate 0.001, 500 warmup updates, cosine decay to zero at 60,000 updates, weight decay 0.1, betas 0.9 and 0.95, grad clipping 1, dropout 0, seed 1337, and `compile=False`. It evaluates 40 batches per split every 2,500 updates and at the end. Every evaluation saves the current checkpoint. The delivered checkpoint is the final one.

The 50-update pilot succeeded with peak PyTorch allocation 718,949,888 bytes. Its logged steady updates took about 12 to 13 ms in one run. That estimate only sized the training schedule; it is not a hardware performance benchmark. The main training process took **875.1 seconds (14.59 minutes)** including startup, evaluation, checkpoints, and exports triggered by `train.py`. Its last logged training-loop elapsed time is 870.9 seconds. It finished update 60,000, presenting `60,000 * 32 * 256 = 491,520,000` training tokens. The pilot took 9.6 loop seconds. Peak main-run PyTorch allocation was 718,949,888 bytes. The process had a 4,200-second hard training-loop cutoff, leaving margin under 90 GPU minutes.

A `Start-Process` attempt died after the SSH session closed. The successful launch uses `Win32_Process.Create` through WMI to run `cmd.exe /c .venv\Scripts\python.exe -u finish.py > finish.log 2>&1` with the workspace as its working directory. Subsequent SSH calls checked actual processes and the advancing heartbeat. `pipeline.py` prepares the tokenizer and runs the pilot. `finish.py` runs training, exports, evaluates, and stages final files. `heartbeat.json`, `train.log`, and `out/loss.jsonl` expose progress. Only this workspace received package installs.

## Loss comparison

The final upstream-style fixed-batch estimates are **train 1.242594, validation 1.236323 nats per token** at 256-token context. These use 327,680 targets per split. `loss.jsonl` retains the full evaluation history.

For the comparison below, `evaluate.py` evaluates the same first 256 stories from validation shard 0. It starts each story with BOS and predicts every text token with no EOS. It splits sequences into nonoverlapping 128-target chunks, carrying the previous token into each chunk. It ignores padded targets. All models use BF16 autocast in the same PyTorch implementation and the same context cap. The sample contains 197,220 UTF-8 bytes. Its exact JSON SHA-256 is `71e5703fa7012d05f59f1b0948e9da25c3fd56b2b8dd03f9ace30305fe0ea0d4`.

| Model | Nats per token | Bits per UTF-8 byte | Predicted tokens |
| --- | ---: | ---: | ---: |
| custom | 1.282628 | 0.606080 | 64,596 |
| custom_q8_dequantized | 1.282908 | 0.606212 | 64,596 |
| stories260K | 1.377393 | 0.952147 | 94,498 |

The same exported legacy files also receive teacher-forced evaluation through unmodified `run.c` tokenizer and forward functions on the Mac, using `evaluate_native.c`. Both native token counts match SentencePiece exactly on this sample. The evaluator resets positions at each 128-target chunk and sums stable float32-forward log likelihoods in double precision.

| Native run.c model | Nats per token | Bits per UTF-8 byte |
| --- | ---: | ---: |
| custom | 1.282614 | 0.606073 |
| stories260K | 1.377205 | 0.952017 |

Token losses use different 1024- and 512-token vocabularies and must not be ranked directly. Bits per byte normalizes the summed text negative log likelihood to the same bytes. On this text sample, the custom model has lower bits per byte than stories260K. This comparison includes differences in model capacity, tokenizer, and training history. It does not attribute the result to the architecture alone. Q8 matrix rounding increases custom token loss by 0.000280 nats in this test. The quantized row applies upstream symmetric group-64 Q8 rounding and dequantizes the matrices before floating-point evaluation. It tests weight quantization error, not integer-only arithmetic, activation quantization, or int8 KV error. Norms remain unchanged. The baseline's original training data overlap with these stories is unknown. Its checkpoint has dim 64, 5 layers, 8 query heads, 4 KV heads, hidden dim 172, vocab 512, and 260,032 parameters. Its KV dimension is 32, so its int8 KV payload at 128 tokens would be `5 * 2 * 32 * 128 = 40,960` bytes. It is a quality reference and does not meet this task's KV limit.

## Native greedy samples

The custom bird sample follows the broken-wing premise initially, then shifts to eating an apple without resolving it. The custom toy sample repeats a sentence. The baseline bird sample repeats a phrase earlier and loses the initial premise. Both models produce grammatical short passages with weak plot continuity. Three prompts do not establish a general quality ranking.

Both models use the same unmodified `run.c`, compiled on the Mac with `clang -O3 run.c -lm`, temperature 0, seed 1, and 128 total positions including the prompt. Samples can end mid-word or mid-sentence at that limit. Different tokenizers produce different text lengths at the same token limit. `verify_artifacts.py` regenerates these samples from the copied artifacts and verifies headers, sizes, and SHA-256 hashes. The runner source SHA-256 is `9c4f2d5c6ae01b71726d1cc37530d71e60bff0ec7cc012565f16a43c1ca658bd`.

Prompt 1. `Once upon a time, there was a little`

Custom model.

> Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, red ball in the sky. She wanted to play with it, but it was too high up in the sky.
> Lily asked her mom, "Can you help me get the ball?" Her mom said, "Sure, let's go get it." They went to the store and found the ball. Lily was so happy! She said, "Thank you, Mom!"
> Later, Lily and her mom went to the park. They saw

stories260K.

> Once upon a time, there was a little girl named Lily. She loved to play outside in the park. One day, she saw a big, red ball. She wanted to play with it, but it was too high.
> Lily's mom said, "Lily, let's go to the park." Lily was sad and didn't know what to do. She said, "I want to play with your ball, but I can't find

Prompt 2. `Lily found a small bird in the garden.`

Custom model.

> Lily found a small bird in the garden. It was a small bird with a broken wing. Lily wanted to help the bird, so she took it to her mom.
> "Mom, can we help the bird?" Lily asked.
> "Of course, Lily. Let's go to the garden and see what we can find," her mom said.
> They went to the garden and found a big, red apple. Lily picked it up and took a bite. It was sweet and juicy. She smiled and

stories260K.

> Lily found a small bird in the garden. She was very happy and wanted to see what was inside. She wanted to see what was inside.
> "Hello, little bird!" Lily said. "I am sorry, Lily. I didn't know what to do."
> "I'm sorry," Lily said.
> "I don't know," Lily said.
> "

Prompt 3. `Tom was sad because he lost his toy.`

Custom model.

> Tom was sad because he lost his toy. He looked everywhere, but he could not find it. He asked his mom, "Mom, where is my toy?"
> His mom said, "I don't know, Tom. Maybe it is in the box. Maybe it is in the box."
> Tom wanted to help his mom. He said, "Mom, can you help me look for my toy? I can't find it."
> His mom said, "Okay, Tom. Let's look for it together."
> T

stories260K.

> Tom was sad because he lost his toy. He wanted to play with his toys and show his friends. He wanted to play with his toys, but he was too shy.
> Tom's friend, a little bird named Bob, saw a big, red ball. Bob was scared and did not know what to do. He wanted to play with the ball. Bob was sad.
> The bird said,

## Reproduction

The checked-in scripts are intended to sit beside the patched upstream Python files in the isolated PC workspace. Copy the pinned upstream files, apply `llama2-windows.patch`, and copy `prepare.py`, `pipeline.py`, `finish.py`, `evaluate.py`, and `train_config.py`. The phase order is `prepare.py`, then `pipeline.py`, then `finish.py`, each using that workspace's venv. `pipeline.py` finishes after its pilot. `finish.py` starts a fresh main model, not a pilot resume. Supply the existing baseline checkpoint and tokenizer under `baseline/` before the final phase. Launch long PC phases through WMI as described above, with redirected logs.

Export alone uses `python export.py model.bin --version 0 --checkpoint out/ckpt.pt`. The extra Q8 export uses version 2. `python tokenizer.py -t data/tok1024.model` exports the tokenizer binary. After copying artifacts back to the Mac, `python3 experiments/tiny-llm-train/verify_artifacts.py` compiles the original C runner and regenerates samples and native losses. `write_report.py` regenerates this report from evidence.

## Files and limits of the evidence

Final local artifacts are under `.cache/tiny-llm/custom/`. `model.bin` is the requested legacy v0 model. `model-q8.bin` is the additional upstream v2 Q8 export. `tok1024.bin`, `tok1024.model`, and `tok1024.vocab` hold the tokenizer. The directory also contains resolved configuration, model arguments, dataset hashes, environment, package versions, loss logs, training logs, evaluation JSON, and samples. Large artifacts remain outside version control. This experiment directory contains the report, exact overrides, support scripts, source patch, decision trail, small evidence JSON, and copies of the loss and evaluation logs. No commits were made.

The source PC retains the final PyTorch checkpoint at `C:\Users\Admin\tiny-llm-train\out\ckpt.pt`. It is not needed by the native runner. The full corpus stays on the PC, leaving more than 10 GiB free at the final disk check.

This establishes artifact integrity, storage bounds, held-out floating-point loss, a weight-only Q8 sensitivity check, and three short native generations. It does not establish an integer-only kernel, target speed, target numerical accuracy, fixed-point softmax or RMSNorm, activation and KV quantization quality, whole-program RAM fit, long-context quality, story completion, or broad language ability. No architecture or hyperparameter search was run. The three greedy samples are qualitative checks, not a controlled quality evaluation.

The Prove It Works principle led to native generation and post-transfer hash checks. Model the Domain kept the evidence organized around architecture, artifacts, losses, and samples. The decision trail is `decisions.tsv` and the workflow is `workflow.md`.
