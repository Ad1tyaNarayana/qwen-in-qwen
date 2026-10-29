"""Compare custom-model KV formats against unmodified float llama2.c."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from build import build, ROOT, HERE
from verify import records, text


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=ROOT / '.cache/tiny-llm/custom-quality')
    p.add_argument('--steps', type=int, default=129)
    a = p.parse_args()
    a.out = a.out.resolve()
    a.out.mkdir(parents=True, exist_ok=True)
    model = ROOT / '.cache/tiny-llm/custom/model.bin'
    tokenizer = model.with_name('tok1024.bin')
    source = ROOT / '.cache/tiny-llm/downloads/llama2.c/run.c'
    shutil.copyfile(source, a.out / 'run.c')
    subprocess.run(['clang','-O2','-Wno-unknown-pragmas','-I'+str(a.out), str(HERE / 'float_probe.c'),
                    '-lm','-o',str(a.out / 'float-probe')], check=True)
    subprocess.run(['clang','-O2','-Wno-unknown-pragmas',str(a.out / 'run.c'),'-lm',
                    '-o',str(a.out / 'float-original')], check=True)
    cases = []
    formats = [(16,10,129),(8,4,255),(8,3,255),(8,2,255)]
    for i, prompt in enumerate(['Once upon a time', 'Lily found a small bird in the garden.',
                                'Tom was sad because he lost his toy.']):
        args = [str(model),str(tokenizer),prompt,str(a.steps)]
        floating = subprocess.check_output([str(a.out / 'float-probe'), *args])
        fr = records(floating)
        original = subprocess.check_output([str(a.out / 'float-original'),str(model),'-z',str(tokenizer),
                                           '-i',prompt,'-t','0','-n',str(a.steps)])
        assert original == text(floating).encode()+b'\n'
        for bits, frac, context in formats:
            out = a.out / f'case-{i}-kv{bits}-q{frac}'
            meta = build(out, prompt, a.steps, context=context, model=model, tokenizer=tokenizer,
                         kv_bits=bits, kv_frac=frac, stack_bytes=1024)
            integer = subprocess.check_output([str(out / 'native'),str(out / 'model.bin')])
            ir = records(integer)
            history = out / 'history.txt'
            history.write_text('\n'.join(str(r[1]) for r in ir))
            teacher = subprocess.check_output([str(a.out / 'float-probe'), *args, str(history)])
            tr = records(teacher)
            assert len(ir) == len(fr) == len(tr) == a.steps
            n = len(meta['prompt_tokens'])-1
            assert [r[1] for r in ir[:n+1]] == [r[1] for r in fr[:n+1]]
            assert [r[1] for r in ir] == [r[1] for r in tr]
            same = sum(x[2] == y[2] for x,y in zip(ir[n:],fr[n:]))
            common = sum(x[2] == y[2] for x,y in zip(ir[n:],tr[n:]))
            for name, raw in [('integer',integer),('float',floating),('teacher',teacher)]:
                (out / (name+'.txt')).write_bytes(raw)
            case = dict(prompt=prompt,prompt_tokens=meta['prompt_tokens'],kv_bits=bits,kv_frac=frac,
                context=context,steps=a.steps,generated_predictions=a.steps-n,
                free_running_matches=same,common_history_matches=common,kv_clips=ir[-1][4],
                integer_text=text(integer),float_text=text(floating),integer_records=ir,
                float_records=fr,teacher_records=tr,pack=meta['pack'],elf=meta['elf'],ram=meta['ram'])
            cases.append(case)
            print(json.dumps({k:case[k] for k in ['prompt','kv_bits','kv_frac','free_running_matches',
                                                 'common_history_matches','generated_predictions','kv_clips']}), flush=True)
            (a.out / 'quality.json').write_text(json.dumps(dict(cases=cases,
                model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                tokenizer_sha256=hashlib.sha256(tokenizer.read_bytes()).hexdigest(),
                float_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()),indent=2)+'\n')


if __name__ == '__main__':
    main()
