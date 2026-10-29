import hashlib
import json
import math
from pathlib import Path
import torch
from model import ModelArgs, Transformer
from tokenizer import Tokenizer
from export import quantize_q80

def load(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    args = dict(checkpoint["model_args"])
    args["max_seq_len"] = max(128, args.get("max_seq_len", 128))
    model = Transformer(ModelArgs(**args))
    state = {k.removeprefix("_orig_mod."): v for k, v in checkpoint["model"].items()}
    model.load_state_dict(state, strict=True)
    return model.eval().cuda()

@torch.no_grad()
def evaluate(model, tokenizer, stories):
    chunks = []
    tokens_total = 0
    for text in stories:
        ids = tokenizer.encode(text, bos=True, eos=False)
        tokens_total += len(ids) - 1
        for offset in range(0, len(ids) - 1, 128):
            part = ids[offset:offset + 129]
            chunks.append((part[:-1], part[1:]))
    nll = 0.0
    for offset in range(0, len(chunks), 32):
        batch = chunks[offset:offset + 32]
        x = torch.zeros((len(batch), 128), dtype=torch.long, device="cuda")
        y = torch.full_like(x, -1)
        for row, (a, b) in enumerate(batch):
            x[row, :len(a)] = torch.tensor(a, device="cuda")
            y[row, :len(b)] = torch.tensor(b, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(x, y)
        nll += model.last_loss.item() * int((y != -1).sum())
    byte_count = sum(len(s.encode("utf-8")) for s in stories)
    return {"nats_per_token": nll / tokens_total, "bits_per_byte": nll / math.log(2) / byte_count, "tokens": tokens_total, "utf8_bytes": byte_count, "stories": len(stories), "context": 128, "total_nll": nll}

if __name__ == "__main__":
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    stories = [e["story"].strip() for e in json.loads(sorted(Path("data/TinyStories_all_data").glob("*.json"))[0].read_text(encoding="utf-8"))[:256]]
    Path("heldout_stories.json").write_text(json.dumps(stories, ensure_ascii=False), encoding="utf-8")
    results = {"protocol": "First 256 stories in held-out shard 0. Each story starts with BOS, no EOS. Nonoverlapping chunks predict every text token, with previous token as input at each 128-token boundary. Padded targets are ignored. bf16 autocast.", "heldout_sha256": hashlib.sha256(Path("heldout_stories.json").read_bytes()).hexdigest()}
    for label, checkpoint, tok in [("custom", "out/ckpt.pt", "data/tok1024.model"), ("stories260K", "baseline/stories260K.pt", "baseline/tok512.model")]:
        model = load(checkpoint)
        tokenizer = Tokenizer(tok)
        results[label] = evaluate(model, tokenizer, stories)
        if label == "custom":
            for parameter in model.parameters():
                if parameter.ndim == 2:
                    q, scale, _ = quantize_q80(parameter, 64)
                    parameter.copy_((q.float() * scale[:, None]).reshape(parameter.shape))
            results["custom_q8_dequantized"] = evaluate(model, tokenizer, stories)
        del model
        torch.cuda.empty_cache()
    Path("evaluation.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)
