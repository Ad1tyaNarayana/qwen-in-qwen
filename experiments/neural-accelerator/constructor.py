"""New generic CPU constructor assembled from a hash-bound frozen scalar source.

Only this module's checked insertion sites change the isolated constructor.
The pinned upstream files and the prepared package files remain untouched.
"""
import hashlib
from pathlib import Path
import types


RAM = '''    ndot4, ndot8 = uops["NDOT4"], uops["NDOT8"]
    ndot_any = ndot4 + ndot8
    ra = c.select_bits([([ndot_any], B), ([1-ndot_any], EA)], 32, "ndot.raddr")
    da = c.select_bits([([ndot_any], A), ([1-ndot_any], EA)], 32, "ndot.daddr")
    stripes = [c.zeros(32, f"ndot.rstripe{k}") for k in range(NDOT_WIDTH//2)]
    n_ram_heads = S.RAM_WORDS >> 8
    for h in range(n_ram_heads):
        hl = lits_of(EA[2:6], h & 15) + lits_of(EA[14:14 + (n_ram_heads >> 4).bit_length() - 1], h >> 4)
        scalar_hl = lits_of(ra[2:6], h & 15) + lits_of(ra[14:14 + (n_ram_heads >> 4).bit_length() - 1], h >> 4)
        dot_hl4 = lits_of(ra[3:6], (h & 15) >> 1) + lits_of(ra[14:14 + (n_ram_heads >> 4).bit_length() - 1], h >> 4)
        dot_hl8 = lits_of(ra[4:6], (h & 15) >> 2) + lits_of(ra[14:14 + (n_ram_heads >> 4).bit_length() - 1], h >> 4)
        bank = c.node(c.t_and([1-ndot_any]+scalar_hl)+c.t_and([ndot4]+dot_hl4)+c.t_and([ndot8]+dot_hl8), f"ndot.rbank{h}")
        keys, q = {}, {}
        for row in range(128):
            rl = lits_of(EA[7:14], row)
            keys[row] = Chan(andpre(rl + hl + [is_store]), (0, 1, 0, 0))
            rr = lits_of(ra[7:14], row)
            q[row] = andpre(rr+[bank])
        vhs = []
        for x in range(2):
            v, z, outs, into = {}, {}, [], []
            for i in range(32):
                v[2+i] = chan(Chan(sd[i], (0, 1, 0, 0)), A_BIT)
            for i in range(32):
                lag = 3-i//15 if i < 30 else 1
                taps = [0, 0, 0, 0]
                taps[0], taps[lag] = 1, -1
                v[34+i] = chan(Chan(-P[i%15 if i < 30 else i-30]+isSS+isV, tuple(taps)), A_BIT)
            for rc in REF_COLS:
                v[rc] = chan(one, A_REF)
            for col in range(2, 66):
                z[col] = K*(ra[6] if x else 1-ra[6])
                outs.append((col, OUT_W))
                into.append(stripes[h % (NDOT_WIDTH//2)][(col-2)%32])
            vhs.append({"v": v, "z": z, "outs": outs, "into": into, "beta": fam[f"ST{x}"]+fam[f"SS{x}"]})
        c.table(f"ram{h}", q, keys, vhs)
    mem = [c.node(c.t_and([1-ndot_any], a=sum((s[i] for s in stripes), Expr())), f"mem.b{i}") for i in range(32)]
    rbits = []
    for lane in range(NDOT_WIDTH):
        word, half = lane//2, lane%2
        for bit in range(16):
            terms = c.t_and([ndot8], a=stripes[word][16*half+bit])
            if lane < 4:
                if NDOT_WIDTH == 4:
                    terms += c.t_and([ndot4], a=stripes[word][16*half+bit])
                else:
                    terms += c.t_and([ndot4, 1-B[3]], a=stripes[word][16*half+bit])
                    terms += c.t_and([ndot4, B[3]], a=stripes[word+2][16*half+bit])
            rbits.append(c.node(terms, f"ndot.rbit{16*lane+bit}"))
'''

DROM = '''    wide = c.zeros(NDOT_WIDTH, "ndot.dbytes")
    dgates = {}
    for x in range(2):
        for column in range(64):
            common = lits_of([da[12]], x)
            dgates[x, column] = K*c.node(c.t_and([1-ndot_any]+lits_of(da[2:6], column >> 2)+common)+c.t_and([ndot4]+lits_of(da[2:6], column >> 2)+common)+c.t_and([ndot8]+lits_of(da[3:6], column >> 3)+common), f"ndot.dgate{x}.{column}")
    for h in range(S.DROM_WORDS >> 11):
        hl = lits_of(da[13:(4*S.DROM_WORDS).bit_length()-1], h)
        q = {}
        for p in range(64):
            e = andpre(lits_of(da[6:12], p)+hl)
            q[p], q[p+64] = e, e
        vhs = []
        for x in range(2):
            z, outs, into = {}, [], []
            for column in range(64):
                col = 2+column
                z[col] = dgates[x, column]
                outs.append((col, OUT_W))
                into.append(wide[column % NDOT_WIDTH])
            vhs.append({"v": {}, "z": z, "outs": outs, "into": into})
        c.table(f"drom{h}", q, {}, vhs, levels="byte")
    ddims = [c.node([term for value in wide[k::4] for term in c.t_and([1-ndot_any], a=value)], f"drom.b{k}") for k in range(4)]
    dbits = [b for k in range(4) for b in c.int_bits(256*ddims[k], 255, f"drom.b{k}", 8)]
    raw_wbits = [c.int_bits(256*wide[k], 255, f"ndot.wbyte{k}", 8) for k in range(NDOT_WIDTH)]
    wbits = []
    for lane in range(NDOT_WIDTH):
        for bit in range(8):
            terms = c.t_and([ndot8, raw_wbits[lane][bit]])
            if lane < 4:
                if NDOT_WIDTH == 4:
                    terms += c.t_and([ndot4, raw_wbits[lane][bit]])
                else:
                    terms += c.t_and([ndot4, 1-A[2], raw_wbits[lane][bit]])
                    terms += c.t_and([ndot4, A[2], raw_wbits[lane+4][bit]])
            wbits.append(c.node(terms, f"ndot.wbit{8*lane+bit}"))
    ndot_result = neural_dot(c, wbits, rbits, C, ndot4, ndot8, NDOT_WIDTH)
    dot_aligned4 = c.and_([1-b for b in A[:2]+B[:3]], "ndot.align4")
    dot_aligned8 = c.and_([1-b for b in A[:3]+B[:4]], "ndot.align8")
    dot_mis = c.node(c.t_and([ndot4, 1-dot_aligned4])+c.t_and([ndot8, 1-dot_aligned8]), "ndot.misaligned")
    dot_regions = c.and_(region_literals(A, S.DROM_BASE, 4*S.DROM_WORDS)+region_literals(B, S.RAM_BASE, 4*S.RAM_WORDS), "ndot.regions")
    dot_oor = c.and_([ndot_any, 1-dot_mis, 1-dot_regions], "ndot.range")
'''


def build(width=4, target=None, reduced=True, stock_integer_counters=False):
    if width not in (4, 8):
        raise ValueError(width)
    from chipformer.rv import core
    from arithmetic import dot
    import reference
    source_path = Path(core.__file__)
    source = source_path.read_text()
    if stock_integer_counters:
        from integer_counters import rewrite
        source = rewrite(source, core)
    start = source.index('    mem = c.zeros(32, "mem")')
    end = source.index('    # ---------------------------------------------------------------- P3: the text buffer', start)
    source = source[:start] + RAM + DROM + '\n' + source[end:]
    replacements = [
        ('from chipformer.rv import ref as RF', 'import reference as RF'),
        ('2: mis_ld, 3: mis_st, 4: oor_ld, 5: oor_st,', '2: mis_ld + dot_mis, 3: mis_st, 4: oor_ld + dot_oor, 5: oor_st,'),
        ('    n_plain = len(srcs)', '    srcs.append(([ndot_any], ndot_result))\n    n_plain = len(srcs)'),
        ('+ uops["MUL"] + mulhs + any_of(*RF.ZBA) + sltish + sltuish)', '+ uops["MUL"] + mulhs + any_of(*RF.ZBA) + sltish + sltuish + ndot_any)'),
        ('"imm": imm, "r": r, "val": val,', '"ndot": ndot_result, "ndot_w": wbits, "ndot_a": rbits, "ndot_raw_w": wide, "ndot_stripes": stripes, "imm": imm, "r": r, "val": val,'),
    ]
    for before, after in replacements:
        if source.count(before) != 1:
            raise ValueError(f'Constructor insertion site changed: {before}')
        source = source.replace(before, after)
    module = types.ModuleType('neural_accelerator_core')
    module.__dict__.update(NDOT_WIDTH=width, neural_dot=lambda *args: dot(*args, compact=reduced))
    exec(compile(source, str(Path(__file__).resolve())+'::<frozen-core>', 'exec'), module.__dict__)
    # W=4 keeps the same uop names for a stable constructor, but never emits NDOT8 code.
    if width == 4:
        adapter = types.SimpleNamespace(**{name: getattr(reference, name)
            for name in dir(reference.ref) if not name.startswith('__')})
        for name in ('CUSTOM', 'U', 'UOPS', 'WRITES_RD'):
            setattr(adapter, name, getattr(reference, name))
        adapter.rom_word = lambda word, pc: reference.rom_word(word, pc, 4)
        module.RF = adapter
    circuit = module.build(target)
    circuit.unreused_stats = circuit.stats()
    if reduced:
        from reuse import reuse
        reuse(circuit)
    circuit.accelerator_module = module
    circuit.accelerator_width = width
    circuit.stock_integer_counters = stock_integer_counters
    circuit.source_manifest = dict(base_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
                                   constructor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                   generated_sha256=hashlib.sha256(source.encode()).hexdigest(),
                                   arithmetic_sha256=hashlib.sha256(Path(__file__).with_name('arithmetic.py').read_bytes()).hexdigest(),
                                   reuse_sha256=hashlib.sha256(Path(__file__).with_name('reuse.py').read_bytes()).hexdigest(),
                                   reduced=reduced, stock_integer_counters=stock_integer_counters,
                                   integer_counters_sha256=hashlib.sha256(Path(__file__).with_name('integer_counters.py').read_bytes()).hexdigest() if stock_integer_counters else None,
                                   replacement_sites=[before for before, _ in replacements])
    return circuit


def snapshot(circuit, machine):
    circuit.accelerator_module.snapshot(circuit, machine)
    if getattr(circuit, 'stock_integer_counters', False):
        from integer_counters import snapshot as integer_snapshot
        integer_snapshot(circuit, machine)
