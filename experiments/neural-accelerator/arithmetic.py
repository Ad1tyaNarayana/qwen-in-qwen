"""Bounded nibble arithmetic for the neural constructor, without host dot evaluation."""
from chipformer.dsl.graph import Expr
from chipformer.q35.prims import from_bits
from chipformer.rv.core import add_nibbles, stage1_digits, adder_stage2


def negate_if(c, bits, sign, name):
    flipped = [c.node(c.t_and([bit, 1-sign])+c.t_and([1-bit, sign]), f'{name}.xor{i}')
               for i, bit in enumerate(bits)]
    return add_nibbles(c, flipped, [sign]+[Expr()]*(len(bits)-1), len(bits)//4, [], name)


def lane_product(c, weights, activations, name):
    a = negate_if(c, weights, weights[-1], name+'.absw')
    b = negate_if(c, activations, activations[-1], name+'.absa')
    sign = c.node(c.t_and([weights[-1], 1-activations[-1]])
                  + c.t_and([1-weights[-1], activations[-1]]), name+'.sign')
    products = {}
    for i in range(2):
        for j in range(4):
            products[i, j] = c.node([term for v in range(1, 16)
                for term in c.t_and([bit if v >> k & 1 else 1-bit
                                     for k, bit in enumerate(a[4*i:4*i+4])], v,
                                    a=from_bits(b[4*j:4*j+4]))], f'{name}.p{i}{j}')
    high = {key: c.split(p, 0, 225, f'{name}.hi{key}') for key, p in products.items()}
    columns = [Expr() for _ in range(8)]
    bounds = [0]*8
    for (i, j), product in products.items():
        columns[i+j] += product-16*high[i, j]
        columns[i+j+1] += high[i, j]
        bounds[i+j] += 15
        bounds[i+j+1] += 14
    # Split each column, then add its bounded carry to the next digit in parallel.
    carries = [c.split(z, 0, limit, f'{name}.col{k}') for k, (z, limit) in enumerate(zip(columns, bounds))]
    sums = [[(columns[k]-16*carries[k]+(carries[k-1] if k else 0), [])] for k in range(8)]
    g, p, d = stage1_digits(c, sums, name+'.normalize', 30)
    magnitude, _ = adder_stage2(c, g, p, d, 8, name+'.normalize')
    return negate_if(c, magnitude, sign, name+'.signed')


def dot(c, weights, activations, accumulator, dot4, dot8, width):
    lanes = []
    for i in range(width):
        product = lane_product(c, weights[8*i:8*i+8], activations[16*i:16*i+16], f'ndot.lane{i}')
        enabled = dot4+dot8 if i < 4 else dot8
        lanes.append([c.node(c.t_and([bit, enabled]), f'ndot.active{i}.{k}')
                      for k, bit in enumerate(product)])
    level = 0
    while len(lanes) > 1:
        lanes = [add_nibbles(c, lanes[i], lanes[i+1], 8, [], f'ndot.tree{level}.{i//2}')
                 for i in range(0, len(lanes), 2)]
        level += 1
    return add_nibbles(c, lanes[0], accumulator, 8, [], 'ndot.accumulate')
