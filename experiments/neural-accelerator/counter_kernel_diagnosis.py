"""Isolate inherited counter decay and reject a beta-halving replacement."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path

import torch
from transformers.models.qwen3_5 import modeling_qwen3_5 as stock


def scalar(value):
    value = value.detach().float().cpu().contiguous()
    return dict(value=value.item(), bits=f'0x{value.view(torch.int32).item() & 0xffffffff:08x}')


def run(length, beta_halving, device, reset_clear=True):
    q = torch.zeros(1, length, 1, 128, device=device)
    k = torch.zeros_like(q)
    v = torch.zeros_like(q)
    beta = torch.ones(1, length, 1, device=device)
    state = torch.zeros(1, 1, 128, 128, device=device)
    state[0, 0, 0, 2] = 128
    if beta_halving:
        q[..., 0] = 128
        k[..., 0] = 128
        beta.fill_(0.5)
        g = torch.zeros_like(beta)
    else:
        q[:, 1:, :, :2] = 128
        k[:, 0, :, 0] = 128
        k[:, 1:, :, 1] = 128
        v[:, 0, :, 2] = 128
        v[:, 1:, :, :2] = 2**24
        a = torch.full_like(beta, 110)
        a[:, 0] = 238 if reset_clear else 0
        g = -torch.nn.functional.softplus(a - 110)
        state[0, 0, 1, :2] = 2**24
    args = dict(query=q, key=k, value=v, g=g, beta=beta, initial_state=state,
                output_final_state=True, use_qk_l2norm_in_kernel=True)
    _, recurrent = stock.torch_recurrent_gated_delta_rule(**args)
    _, chunked = stock.torch_chunk_gated_delta_rule(**args)
    return dict(length=length, beta_halving=beta_halving, reset_clear=reset_clear,
                recurrent=scalar(recurrent[0, 0, 0, 2]), chunked=scalar(chunked[0, 0, 0, 2]),
                g=[scalar(x) for x in g[0, :min(length, 2), 0]],
                cumulative_g=[scalar(x) for x in g.cumsum(1)[0, :min(length, 2), 0]],
                pairwise_exp=scalar((g.cumsum(1)[0, 1, 0] - g.cumsum(1)[0, 0, 0]).exp()) if length > 1 else None,
                equal=torch.equal(recurrent, chunked))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    a = p.parse_args()
    report = dict(device=a.device, torch=torch.__version__, transformers=__import__('transformers').__version__,
                  scope='stock kernel isolation; supplied synthetic counter channels, not autonomous guest evidence',
                  source_sha256={name: hashlib.sha256(inspect.getsource(getattr(stock, name)).encode()).hexdigest()
                                 for name in ('torch_chunk_gated_delta_rule', 'torch_recurrent_gated_delta_rule')},
                  runs=[run(n, beta, a.device) for beta in (False, True) for n in (1, 2, 4, 8, 16, 32, 64)]
                       + [run(n, False, a.device, False) for n in (16, 32, 64)])
    a.out.write_text(json.dumps(report, indent=2) + '\n')
    a.out.with_suffix('.source.py').write_text('\n\n'.join(inspect.getsource(getattr(stock, name))
        for name in ('torch_chunk_gated_delta_rule', 'torch_recurrent_gated_delta_rule')), encoding='utf-8')


if __name__ == '__main__':
    main()
