"""Check integer-counter channels under the installed stock DeltaNet kernels."""
import argparse
import json
from pathlib import Path

import torch
from transformers.models.qwen3_5 import modeling_qwen3_5 as stock


def check(events, initial_count, device):
    n = len(events)
    q = torch.zeros(1, n, 1, 128, dtype=torch.bfloat16, device=device)
    q[..., 0] = 128
    k = q.clone()
    v = torch.zeros_like(q)
    v[..., :2] = 181 * 2**17
    b = torch.tensor([128 if x == 'reset' else -30.5 if x == 'step' else -128
                      for x in events], dtype=torch.bfloat16, device=device).reshape(1, n, 1)
    beta = b.sigmoid()
    assert torch.all(beta[b == -30.5] == 2.**-44)
    for i, event in enumerate(events):
        if event != 'reset':
            v[0, i, 0, 2] = 181 * 2.**44
    g = torch.zeros(1, n, 1, device=device)
    initial = torch.zeros(1, 1, 128, 128, device=device)
    initial[0, 0, 0, :2] = 181 * 2**17
    initial[0, 0, 0, 2] = 181*initial_count
    count = initial_count
    states = []
    for i, event in enumerate(events):
        count = 0 if event == 'reset' else count + (event == 'step')
        _, state = stock.torch_recurrent_gated_delta_rule(q[:, i:i+1], k[:, i:i+1], v[:, i:i+1],
            g[:, i:i+1], beta[:, i:i+1], initial_state=initial if i == 0 else states[-1],
            output_final_state=True, use_qk_l2norm_in_kernel=True)
        assert state[0, 0, 0, 2].item() == 181*count, (i, event, state[0, 0, 0, 2].item(), count)
        states.append(state)
    results = []
    for size in (1, 2, 4, 8, 16, 64):
        state = initial
        for offset in range(0, n, size):
            end = min(offset+size, n)
            _, state = stock.torch_chunk_gated_delta_rule(q[:, offset:end], k[:, offset:end], v[:, offset:end],
                g[:, offset:end], beta[:, offset:end], initial_state=state,
                output_final_state=True, use_qk_l2norm_in_kernel=True)
            assert torch.equal(state, states[end-1]), (size, end, state[0, 0, 0, 2].item(), states[end-1][0, 0, 0, 2].item())
        results.append(dict(chunk_size=size, exact=True, final_integer_count=count))
    return dict(initial_count=initial_count, events=events, results=results)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    import random
    rng = random.Random(20261030)
    cases = [('step',)*64, ('reset',)+('step',)*63,
             tuple(['step','idle','reset','step','step','reset','idle','step']*8)]
    cases += [tuple(rng.choices(['step','idle','reset'], weights=[8,2,1], k=64)) for _ in range(16)]
    report = dict(status='running', device=a.device, torch=torch.__version__,
                  transformers=__import__('transformers').__version__, runs=[])
    a.out.write_text(json.dumps(report, indent=2)+'\n')
    try:
        norm = stock.Qwen3_5RMSNormGated(128).to(dtype=torch.bfloat16, device=a.device)
        rows = torch.zeros(128, 128, device=a.device)
        rows[:, :2] = 181 * 2**17
        rows[:, 2] = torch.arange(128, device=a.device) * 181
        read = (rows / (128**0.5)).to(torch.bfloat16)
        z = torch.zeros_like(read)
        z[:, 2] = 128
        values = norm(read, z)[:, 2].float() * 128
        assert torch.equal(values, torch.arange(128, device=a.device).float())
        report['stock_normalized_reads_exact_0_through_127'] = True
        for initial in (0, 63):
            for events in cases:
                report['runs'].append(check(events, initial, a.device))
                a.out.write_text(json.dumps(report, indent=2)+'\n')
        report['status'] = 'pass'
    except Exception as error:
        report.update(status='fail', error=repr(error))
        raise
    finally:
        a.out.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
