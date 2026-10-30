"""Free-running greedy comparison against the unmodified installed HF model."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess

from pack import qwen, ROOT


def quality(checkpoint, runtime, out, steps=64, quantization='midpoint', prompts=None, calibration=None):
    import torch
    from transformers import Qwen3_5ForCausalLM
    torch.set_num_threads(1)
    model = Qwen3_5ForCausalLM.from_pretrained(checkpoint,torch_dtype=torch.float32,
                                             attn_implementation='eager').eval()
    cases = []
    prompts = prompts or ['Once upon a time','Lily found a small bird in the garden.',
                          'Tom was sad because he lost his toy.']
    for i,prompt in enumerate(prompts):
        directory = out/f'case-{i}'
        packed = qwen(checkpoint,ROOT/'.cache/tiny-llm/custom/tok1024.bin',directory,64,prompt,steps,quantization,calibration)
        output = subprocess.check_output([str(runtime.resolve()),str(directory.resolve()/'model.bin')])
        records = [list(map(int,r)) for r in re.findall(rb'^@(\d+) (\d+) (\d+) (\d+) (\d+)\n',output,re.M)]
        assert len(records) == steps
        tokens = packed['prompt_tokens']
        begin = len(tokens)-1
        generation = copy.deepcopy(model.generation_config)
        generation.eos_token_id = None
        generation.pad_token_id = 0
        generation.do_sample = False
        generation.max_new_tokens = steps-begin
        with torch.inference_mode():
            result = model.generate(input_ids=torch.tensor([tokens]),generation_config=generation,use_cache=True)
        generated = result[0,len(tokens):].tolist()
        assert len(generated) == steps-begin
        float_records = [[begin+i,tokens[-1] if i == 0 else generated[i-1],token]
                         for i,token in enumerate(generated)]
        matches = sum(a[2] == b[2] for a,b in zip(records[begin:],float_records))
        tok = json.loads((directory/'tokenizer.json').read_text())
        def decode(ids):
            pieces = []
            for token in ids:
                piece = bytes(tok['pieces'][token])
                if re.fullmatch(rb'<0x[0-9a-fA-F]{2}>',piece):
                    piece = bytes([int(piece[3:5],16)])
                pieces.append(piece)
            return b''.join(pieces).decode(errors='replace')
        cases.append(dict(prompt=prompt,prompt_tokens=tokens,generated_predictions=steps-begin,
                          free_running_matches=matches,free_running_agreement=matches/(steps-begin),
                          integer_records=records,float_records=float_records,
                          float_continuation=decode(generated),integer_output=output.decode(errors='replace'),
                          kv_clips=records[-1][4],numeric_clips=re.findall(rb'^!(\d+)$',output,re.M),
                          rom_sha256=packed['rom_sha256']))
        cases[-1]['numeric_clips'] = [int(n) for n in cases[-1]['numeric_clips']]
        (directory/'integer-output.bin').write_bytes(output)
    result = dict(status='measured',kind='free-running greedy float HF vs integer native',
                  float_method='unmodified model.generate, full prompt prefill, fixed length with EOS stopping disabled',
                  checkpoint=str(checkpoint),cases=cases,
                  quantization=quantization,
                  calibration_sha256=hashlib.sha256(calibration.read_bytes()).hexdigest() if calibration else None,
                  calibration_file=str(calibration) if calibration else None,
                  free_running_agreement=sum(c['free_running_matches'] for c in cases)/
                  sum(c['generated_predictions'] for c in cases),
                  checkpoint_files={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in checkpoint.iterdir() if p.is_file()},
                  runtime_sha256=hashlib.sha256(runtime.read_bytes()).hexdigest())
    (out/'quality.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'cases'},indent=2))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--steps',type=int,default=64)
    p.add_argument('--quantization',default='midpoint')
    p.add_argument('--calibration',type=Path)
    a = p.parse_args()
    quality(a.checkpoint,a.runtime,a.out,a.steps,a.quantization,calibration=a.calibration)
