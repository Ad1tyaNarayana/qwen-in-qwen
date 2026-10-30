"""Exercise load-before-clear across Triton lane layouts in ordered apply."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import struct
import threading


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--tokens', type=int, default=8192)
    a = p.parse_args()
    watchdog = threading.Timer(120, lambda: os._exit(124))
    watchdog.daemon = True
    watchdog.start()
    import runtime
    work = a.out.parent / 'apply-scratch'
    work.mkdir(parents=True, exist_ok=True)
    fr, _, _, _ = runtime.build(work, a.inputs)
    import torch
    import triton
    import triton.language as tl
    from fusion import _ordered_apply
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'dmc-rom-fusion'))
    from bisect_fusion import _unsafe_apply
    from fusion import install_rom
    info = install_rom(fr.ex)
    D = fr.ex.D
    groups = max(len(b['indices']) for b in info['batches'].values())
    blocks = triton.cdiv(D,1024)
    x = torch.zeros(D, device='cuda')
    y = torch.ones(groups, D, device='cuda')
    hist = torch.zeros(groups, 3, D, device='cuda')
    initial_hist = torch.empty_like(hist)
    initial_hist[:,0].fill_(7.0)
    initial_hist[:,1].fill_(11.0)
    initial_hist[:,2].fill_(13.0)
    records = torch.zeros(a.tokens, blocks, 3, device='cuda', dtype=torch.int32)
    pos = torch.zeros(1, device='cuda', dtype=torch.int32)

    @triton.jit
    def record(x, y, hist, records, pos, D: tl.constexpr, groups: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        d = pid * BLOCK + tl.arange(0, BLOCK)
        val = tl.load(x+d, d < D, other=float(groups))
        bad = (d < D) & (val != float(groups))
        for g in range(groups):
            h = hist + g * 3 * D + d
            bad |= (d < D) & (tl.load(h, d < D, other=0.0) != g)
            bad |= (d < D) & (tl.load(h + D, d < D, other=0.0) != 7.0)
            bad |= (d < D) & (tl.load(h + 2 * D, d < D, other=0.0) != 11.0)
            bad |= (d < D) & (tl.load(y + g * D + d, d < D, other=0.0) != 0.0)
        first = tl.min(tl.where(bad, d, D), 0)
        bits = tl.load(x+first, first < D, other=float(groups)).to(tl.int32, bitcast=True)
        t = tl.load(pos)
        tl.store(records+(t*tl.num_programs(0)+pid)*3, tl.sum(bad.to(tl.int32), 0))
        tl.store(records+(t*tl.num_programs(0)+pid)*3+1, first)
        tl.store(records+(t*tl.num_programs(0)+pid)*3+2, bits)

    @triton.jit
    def advance(pos):
        tl.store(pos, tl.load(pos)+1)

    report = {'tokens': a.tokens, 'kernels': {}, 'scripts': {name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in ('fusion.py', 'apply_race.py')}}
    for name, kernel in [('original', _unsafe_apply), ('barrier', _ordered_apply)]:
        def step():
            x.zero_()
            y.fill_(1.0)
            hist.copy_(initial_hist)
            kernel[(blocks,)](x, y, hist, D, groups, BLOCK=1024)
            record[(blocks,)](x, y, hist, records, pos, D, groups, BLOCK=1024)
            advance[(1,)](pos)
        step()
        torch.cuda.synchronize()
        pos.zero_()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
        pos.zero_()
        records.zero_()
        for _ in range(a.tokens):
            graph.replay()
        torch.cuda.synchronize()
        raw = records.cpu().numpy()
        import numpy as np
        bad = np.argwhere(raw[:, :, 0] != 0)
        first = None
        if len(bad):
            t, block = map(int, bad[0])
            first = {'token': t+1, 'word': int(raw[t, block, 1]), 'actual_bits': int(raw[t, block, 2]) & 0xffffffff, 'expected_bits': struct.unpack('<I',struct.pack('<f',float(groups)))[0]}
        report['kernels'][name] = {'bad_token_blocks': len(bad), 'mismatched_words': int(raw[:, :, 0].sum()), 'first': first}
        print(json.dumps({name: report['kernels'][name]}), flush=True)
    a.out.write_text(json.dumps(report, indent=2)+'\n')
    assert report['kernels']['barrier']['mismatched_words'] == 0
    watchdog.cancel()


if __name__ == '__main__':
    main()
