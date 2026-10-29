"""CPU-only primitive checks, not an implemented dot instruction or CPU loop."""
import json
import hashlib
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '.upstream/qwenputer/experimental/riscv'))
import numpy as np
import torch
from chipformer.rv import hf
from chipformer.q35.ir import Target, evaluate
from chipformer.q35.prims import Circ, from_bits


def product_graph():
    c = Circ(Target(key_heads=16, value_heads=32, intermediate=9216))
    a = [c.token_feature(f'a{i}', lambda t, i=i: (t >> i) & 1) for i in range(4)]
    b = [c.token_feature(f'b{i}', lambda t, i=i: (t >> (4+i)) & 1) for i in range(4)]
    p = c.node([term for v in range(1, 16)
                for term in c.t_and([bit if v >> i & 1 else 1-bit
                                     for i, bit in enumerate(a)], v, a=from_bits(b))], 'p')
    return c, p


def digit_graph():
    c = Circ(Target(key_heads=16, value_heads=32, intermediate=9216))
    ab = [c.token_feature(f'b{i}', lambda t, i=i: (t >> i) & 1) for i in range(9)]
    z = from_bits(ab[:4]) + from_bits(ab[4:8]) + ab[8]
    carry = c.node(c.t_ge(z, 16, 0, 31), 'carry')
    digit = c.node([(1, z - 16 * carry, c.one)], 'digit')
    return c, digit, carry


def scalar(e, matrix):
    return sum(float(w)*matrix[:, d] for d, w in e.terms.items())


def limb_product(a, b, width=64):
    """Unsigned magnitude nibble convolution, normalized, then signed as bits."""
    aa, bb = abs(a), abs(b)
    na = max(1, (aa.bit_length()+3)//4)
    nb = max(1, (bb.bit_length()+3)//4)
    columns = [0] * (na+nb+1)
    for i in range(na):
        for j in range(nb):
            p = ((aa >> (4*i)) & 15) * ((bb >> (4*j)) & 15)
            columns[i+j] += p & 15
            columns[i+j+1] += p >> 4
    carry = 0
    digits = []
    for z in columns:
        z += carry
        digits.append(z & 15)
        carry = z >> 4
    assert carry == 0
    magnitude = sum(d << (4*i) for i, d in enumerate(digits))
    return (-magnitude if (a < 0) != (b < 0) else magnitude) % (1 << width)


def main():
    torch.set_num_threads(4)
    c, p = product_graph()
    m = evaluate(c, list(range(256)), check=True, kernel=True)
    assert np.array_equal(scalar(p, m), np.array([(t & 15)*(t >> 4) for t in range(256)]))
    d, digit, carry = digit_graph()
    m = evaluate(d, list(range(512)), check=True, kernel=True)
    expected = np.array([(t & 15)+((t >> 4) & 15)+(t >> 8) for t in range(512)])
    assert np.array_equal(scalar(digit, m), expected % 16)
    assert np.array_equal(scalar(carry, m), expected // 16)
    stock = []
    for graph, toks in ((c, [0,1,15,16,17,127,128,239,240,254,255]),
                        (d, [0,15,16,255,256,271,496,510,511])):
        model, layout, ids = hf.build_model(graph, toks)
        stock.append({'decode': hf.check(graph, model, layout, ids, toks, mode='decode'),
                      'prefill': hf.check(graph, model, layout, ids, toks, mode='prefill'),
                      'hidden': layout.hidden})
        del model
    rng = random.Random(271828)
    cases = []
    for ba, bb in ((8, 16), (16, 16), (32, 16)):
        edges_a = [-(1 << (ba-1)), -1, 0, 1, (1 << (ba-1))-1]
        edges_b = [-(1 << (bb-1)), -1, 0, 1, (1 << (bb-1))-1]
        pairs = [(a,b) for a in edges_a for b in edges_b]
        pairs += [(rng.randrange(-(1 << (ba-1)), 1 << (ba-1)),
                   rng.randrange(-(1 << (bb-1)), 1 << (bb-1))) for _ in range(10000)]
        for a,b in pairs:
            assert limb_product(a,b) == (a*b) % (1 << 64)
        for n in (4,8,16):
            for _ in range(1000):
                batch = rng.choices(pairs, k=n)
                assert sum(limb_product(a,b) for a,b in batch) % (1 << 64) == sum(a*b for a,b in batch) % (1 << 64)
        cases.append({'operand_bits': [ba,bb], 'products': len(pairs), 'dot_vectors': 3000})
    result = {'scope': 'IR bf16-safe nibble product and digit adder; Python limb arithmetic only for full dots',
              'nibble_pairs': 256, 'digit_additions': 512,
              'product_graph': c.stats(), 'digit_graph': d.stats(), 'integer_checks': cases,
              'stock_HF_supplied_inputs': stock, 'full_dot_IR_checked': False,
              'autonomous_generation_checked': False, 'torch': torch.__version__,
              'transformers': __import__('transformers').__version__,
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    Path(__file__).with_name('arithmetic-result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
