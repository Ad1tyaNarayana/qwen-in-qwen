"""Build the original integer engine with width-four matrix-dot loops."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = ROOT / 'experiments/tiny-llm-rv32'
sys.path.insert(0, str(ORIGINAL))
spec = importlib.util.spec_from_file_location('rv32_build', ORIGINAL / 'build.py')
rv32 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rv32)

LOOP = '''        uint32_t acc = 0;
        const int8_t *row = w + i * n;
        int j = 0;
        for (; j + 4 <= n && ((uintptr_t)row & 3) == 0; j += 4) {
            __asm__ volatile(".global dot_loop_begin\\ndot_loop_begin:" ::: "memory");
#ifdef NATIVE
            for (int lane = 0; lane < 4; lane++)
                acc += (uint32_t)((int32_t)row[j+lane] * quant[j+lane]);
#else
            acc = ndot4(acc, row + j, quant + j);
#endif
            __asm__ volatile(".global dot_loop_end\\ndot_loop_end:" ::: "memory");
        }
        for (; j < n; j++) acc += (uint32_t)((int32_t)row[j] * quant[j]);
        int32_t sum = (int32_t)acc;
'''


def build(out, prompt='Once', steps=5):
    out = out.resolve()
    source_dir = out / 'source'
    source_dir.mkdir(parents=True, exist_ok=False)
    source = (ORIGINAL / 'engine.c').read_text()
    begin = source.index('        int32_t sum = 0;\n')
    end = source.index('        o[i] =', begin)
    source = source[:begin] + LOOP + source[end:]
    source = source.replace('#include "model.h"', '#include "model.h"\n#ifndef NATIVE\n#include "ndot.h"\n#endif')
    source = source.replace('static int16_t quant[HIDDEN > DIM ? HIDDEN : DIM];',
                            'static _Alignas(8) int16_t quant[HIDDEN > DIM ? HIDDEN : DIM];')
    (source_dir / 'engine.c').write_text(source)
    for name in ('start.S', 'link.ld'):
        shutil.copyfile(ORIGINAL / name, source_dir / name)
    shutil.copyfile(ROOT / 'experiments/neural-accelerator/ndot.h', source_dir / 'ndot.h')
    rv32.HERE = source_dir
    meta = rv32.build(out, prompt, steps, context=64, stack_bytes=8192)
    header = (out / 'model.h').read_text()
    # Every stories matrix row is divisible by four. Scalar cleanup handles loop tails.
    for line in header.splitlines():
        if line.startswith('#define O_') and any(line.split()[1] == 'O_'+name for name in ('wq','wk','wv','wo','w1','w2','w3','cls')):
            assert int(line.split()[2]) % 4 == 0, line
    meta['ndot'] = dict(width=4, matrix_only=True, intrinsic_sha256=hashlib.sha256((source_dir / 'ndot.h').read_bytes()).hexdigest(),
                        original_engine_sha256=hashlib.sha256((ORIGINAL / 'engine.c').read_bytes()).hexdigest())
    (out / 'build.json').write_text(json.dumps(meta, indent=2)+'\n')
    return meta


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--prompt', default='Once')
    p.add_argument('--steps', type=int, default=5)
    a = p.parse_args()
    build(a.out, a.prompt, a.steps)
