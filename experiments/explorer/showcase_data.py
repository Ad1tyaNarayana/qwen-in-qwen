"""Demo layouts and application events decoded from actual generated commit tokens."""
import hashlib
import struct
from common import ROOT

TRACE = 'decoded from generated trace'
NEURAL = 'read from neural state'
DEMOS = ('sha256', 'life', 'tictactoe', 'mnist', 'mnist-scalar')
WEIGHTS = '0e515bbb3109997a130c903deb40dce3447f8991e15b6637e0071c71f8b6e07d'
SHA_ELF = 'c7b0dbe9da21ada944059ad7a69e6f99f71847306427cff1a7d43906f71c2869'


def layout(inputs, program):
    """Only recognized, hash-bound guests receive an application decoder."""
    import json
    for demo in DEMOS:
        path = ROOT/f'experiments/showcase/manifest/{demo}.json'
        if not path.exists(): continue
        manifest = json.loads(path.read_text())
        if manifest['elf_sha256'] == program['elf_sha256']:
            return dict(kind=demo, symbols={s['name']:s for s in program['symbols']},
                        elf_sha256=program['elf_sha256'], network_weight_sha256=WEIGHTS)
    return None


def arrays_from_ram(ram, config):
    result = {}
    for name, fmt in (('board','B'), ('next','B'), ('input','h'), ('hidden','h')):
        sym = config['symbols'].get(name)
        if not sym: continue
        off = sym['address']-0x1000000
        size = sym['size']
        assert 0 <= off < off+size <= len(ram)
        result[name] = dict(address=sym['address'], values=list(struct.unpack_from('<'+str(size//struct.calcsize(fmt))+fmt,ram,off)))
    if config['elf_sha256'] == SHA_ELF:
        # This ELF's digest prologue allocates 320 bytes below engine_main's
        # 16-byte frame. w = fp-288; out = fp-320. Retained after return.
        frame = 0x1010000-16
        for name, off, count in (('schedule',frame-288,64), ('digest_words',frame-320,8)):
            result[name] = dict(address=off, values=list(struct.unpack_from('<'+str(count)+'I',ram,off-0x1000000)))
    return result


def snapshot_app(snapshot, config):
    return dict(kind=config['kind'], provenance=NEURAL, offset=snapshot['offset'],
                arrays=arrays_from_ram(bytes.fromhex(snapshot['ram_hex']),config),
                note='Exact recurrent RAM sample. SHA stack buffers retain the last invocation after return. MNIST logits are register temporaries, not a RAM array.')


def trace_application(pairs, program, inputs, config):
    """Replay committed values and stores, without computing guest instructions."""
    from chipformer.rv import elf, spec
    from bisect import bisect_right
    prog = elf.program(inputs/'engine.elf')
    ram = bytearray(65536)
    off = prog.data_base-spec.RAM_BASE
    ram[off:off+len(prog.data)] = prog.data
    regs = [0]*32
    regs[2] = spec.SP_INIT
    code = {i[0]:i for i in program['instructions']}
    events = []
    output = ''
    payloads = []
    logits = []
    sample = -1
    rounds = 0
    demo = config['kind']
    logit_pc = None
    if demo.startswith('mnist'):
        ins = program['instructions']
        candidates = [a[0] for a,b,c in zip(ins,ins[1:],ins[2:])
                      if a[2:6]==['OR',15,15,10] and b[2:]==['ADDI',16,14,0,0] and c[2]=='BLT']
        assert len(candidates)==1, 'Unrecognized MNIST logit register layout'
        logit_pc = candidates[0]
    functions = sorted((s for s in program['symbols'] if s['kind']==2),key=lambda s:s['address'])
    starts = [s['address'] for s in functions]
    minimax = 0
    for index,(token,pc) in enumerate(pairs):
        token,pc = int(token),int(pc)
        fam,payload = spec.family(token)
        if fam not in spec.FINAL:
            if fam=='V': payloads.append(payload)
            continue
        inst = code[pc]
        _,_,op,rd,rs1,rs2,imm = inst
        if fam.startswith('F'):
            value = spec.decode_value([spec.tok('V',v) for v in payloads]+[token])
            if rd: regs[rd] = value & 0xffffffff
        elif fam=='J' and rd and op in ('JAL','JALR'):
            regs[rd] = pc+4
        elif fam.startswith(('ST','SS')):
            address = ((regs[rs1]+imm)&0xffffffff)-spec.RAM_BASE
            assert 0 <= address <= len(ram)-({'SB':1,'SH':2,'SW':4}[op])
            word = regs[rs2] if fam.startswith('ST') else sum(v << (15*i) for i,v in enumerate(payloads))
            aligned = address & ~3
            ram[aligned:aligned+4] = struct.pack('<I',word)
        payloads = []
        j = bisect_right(starts,pc)-1
        if j>=0 and functions[j]['name']=='score': minimax += 1
        frame = None
        if fam=='BYTE' and payload<256:
            output += chr(payload)
            if demo=='life' and output.endswith('\n') and output.splitlines()[-1].startswith('Generation '):
                frame = dict(label=output.splitlines()[-1],arrays=arrays_from_ram(ram,config))
            if demo=='tictactoe' and output.endswith('\n\n'):
                frame = dict(label='Board '+str(len(events)),arrays=arrays_from_ram(ram,config),minimax_token_positions=minimax)
            if demo.startswith('mnist') and output.endswith('\n') and output.splitlines()[-1].startswith('MNIST test '):
                sample += 1
                logits = []
        if pc==logit_pc:
            value = regs[15]
            logits.append(value if value<2**31 else value-2**32)
            if len(logits)==10:
                frame = dict(label=f'Test {sample}',sample=sample,arrays=arrays_from_ram(ram,config),logits=logits[:])
        if demo=='sha256' and pc==0x402d0:
            rounds += 1
            frame = dict(label=f'{"empty" if rounds<=64 else "abc"} round {(rounds-1)%64+1}',
                         round=(rounds-1)%64+1,arrays=arrays_from_ram(ram,config),
                         working=[regs[r] for r in (18,31,30,15,7,29,28,5)])
        if frame:
            events.append(dict(offset=index+1,provenance=TRACE,**frame))
    return dict(kind=demo,provenance=TRACE,events=events,
                final_ram=bytes(ram), final_registers=regs,
                reference_digests={s:hashlib.sha256(s.encode()).hexdigest() for s in ('','abc')} if demo=='sha256' else None,
                scope='RAM replayed from generated value/store commits. No reference execution or host-computed instruction results supply application values.')
