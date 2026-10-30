"""Time-bounded HF Qwen3.5 training in the existing, isolated training venv."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
import transformers


def save_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n')
    temp.replace(path)


def train(root, config_path, output):
    start = time.monotonic()
    cfg = json.loads(config_path.read_text())
    settings = cfg['training']
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    random.seed(cfg['seed'])
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this training run')
    torch.cuda.set_per_process_memory_fraction(0.45)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats()
    model = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**cfg['model'])).cuda()
    model.config.use_cache = False
    optimizer = torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],
                                 betas=(0.9,0.95),weight_decay=0.1)
    shards = sorted((root/'data/tok1024').glob('*.bin'))
    assert len(shards) >= 2
    maps = [np.memmap(path,dtype=np.uint16,mode='r') for path in shards]
    length = settings['context']+1

    def batch(validation=False):
        shard = maps[0] if validation else random.choice(maps[1:])
        starts = np.random.randint(0,len(shard)-length,size=settings['batch_size'])
        data = np.stack([shard[i:i+length] for i in starts]).astype(np.int64)
        return torch.from_numpy(data).cuda()

    env = dict(python=sys.version,torch=torch.__version__,transformers=transformers.__version__,
               cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),config=cfg,
               parameters=sum(p.numel() for p in model.parameters()),
               model_class=type(model).__name__,layer_types=model.config.layer_types,
               dataset_manifest_sha256=hashlib.sha256((root/'dataset.json').read_bytes()).hexdigest(),
               tokenizer_sha256=hashlib.sha256((root/'data/tok1024.model').read_bytes()).hexdigest(),
               training_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               validation_shard=shards[0].name,train_shards=[p.name for p in shards[1:]])
    save_json(output/'environment.json',env)
    (output/'requirements-lock.txt').write_bytes(subprocess.check_output([sys.executable,'-m','pip','freeze']))
    step = 0
    with (output/'loss.jsonl').open('w',buffering=1) as log:
        while step < settings['max_steps'] and time.monotonic()-start < settings['max_seconds']:
            step += 1
            fraction = min(1,step/settings['warmup_steps'])
            progress = min(1,(time.monotonic()-start)/settings['max_seconds'])
            lr = settings['learning_rate']*fraction*(0.1+0.9*0.5*(1+math.cos(math.pi*progress)))
            for group in optimizer.param_groups:
                group['lr'] = lr
            model.train()
            inputs = batch()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                loss = model(input_ids=inputs,labels=inputs,use_cache=False).loss
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
            optimizer.step()
            if step % 20 == 0 or step == 1:
                torch.cuda.synchronize()
                row = dict(step=step,loss=float(loss.detach()),lr=lr,
                           elapsed_seconds=time.monotonic()-start,
                           peak_allocated=torch.cuda.max_memory_allocated())
                log.write(json.dumps(row)+'\n')
                save_json(output/'heartbeat.json',row)
                print(json.dumps(row),flush=True)
            if step % settings['eval_interval'] == 0 and time.monotonic()-start < settings['max_seconds']-30:
                model.eval()
                losses = []
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    for _ in range(settings['eval_batches']):
                        inputs = batch(True)
                        losses.append(float(model(input_ids=inputs,labels=inputs,use_cache=False).loss))
                row = dict(step=step,validation_loss=sum(losses)/len(losses),elapsed_seconds=time.monotonic()-start)
                log.write(json.dumps(row)+'\n')
                print(json.dumps(row),flush=True)
    train_seconds = time.monotonic()-start
    model.eval().float()
    model.config.use_cache = True
    model.save_pretrained(output/'hf')
    import sentencepiece as spm
    tok = spm.SentencePieceProcessor(model_file=str(root/'data/tok1024.model'))
    samples = []
    torch.backends.cuda.matmul.allow_tf32 = False
    for prompt in ['Once upon a time','Lily found a small bird in the garden.',
                   'Tom was sad because he lost his toy.']:
        ids = [1]+tok.encode(prompt)
        with torch.inference_mode():
            generated = model.generate(input_ids=torch.tensor([ids],device='cuda'),max_new_tokens=64,
                                       do_sample=False,use_cache=True)
        raw = generated[0].cpu().tolist()
        samples.append(dict(prompt=prompt,prompt_tokens=ids,tokens=raw,text=tok.decode(raw)))
    save_json(output/'samples.json',samples)
    save_json(output/'done.json',dict(status='complete',steps=step,training_seconds=train_seconds,
              wall_seconds=time.monotonic()-start,peak_allocated=torch.cuda.max_memory_allocated(),
              hf_files={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                        for p in (output/'hf').iterdir() if p.is_file()}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a = p.parse_args()
    train(a.root,a.config,a.out)
