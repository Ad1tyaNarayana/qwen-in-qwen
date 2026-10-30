"""Import generated traces or observe fused execution. Reference files only validate."""
import argparse
from bisect import bisect_right
from collections import Counter
import gzip
import json
from pathlib import Path
import re
import struct
import sys
from types import SimpleNamespace
import numpy as np
from common import HERE, ROOT, compiler, sha, write_json

NEURAL = 'read from neural state'
TRACE = 'decoded from generated trace'


def elf_symbols(path):
    raw = path.read_bytes()
    assert raw[:6] == b'\x7fELF\x01\x01', 'ELF32 little endian required'
    shoff = struct.unpack_from('<I',raw,32)[0]
    size,count = struct.unpack_from('<HH',raw,46)
    sections = [struct.unpack_from('<10I',raw,shoff+i*size) for i in range(count)]
    result = []
    for s in sections:
        if s[1] != 2: continue
        strings = sections[s[6]]
        names = raw[strings[4]:strings[4]+strings[5]]
        for off in range(s[4],s[4]+s[5],s[9]):
            name,address,n,info,other,section = struct.unpack_from('<IIIBBH',raw,off)
            if not name or not address or not section: continue
            end = names.find(b'\0',name)
            label = names[name:end].decode(errors='replace')
            if label.startswith(('.', '$')): continue
            result.append(dict(name=label,address=address,size=n,kind=info&15))
    return result


def program_info(inputs):
    from chipformer.rv import elf
    import reference
    symbols = elf_symbols(inputs/'engine.elf')
    code = elf.program(inputs/'engine.elf').text
    instructions = []
    for i,word in enumerate(code):
        d = reference.decode(word,4)
        instructions.append([0x40000+4*i,word,d.op,d.rd,d.rs1,d.rs2,d.imm])
    return dict(provenance='ELF static data',elf_sha256=sha(inputs/'engine.elf'),
                symbols=symbols,instructions=instructions,entry=elf.segments(inputs/'engine.elf')[0])


def decode_trace(tokens, spec, entry, start=0, stop=None):
    stop = len(tokens) if stop is None else min(stop,len(tokens))
    anchor,count = (entry-0x40000)//4,0
    pairs = np.empty((stop-start,2),dtype='<u4')
    begin=0
    if start:
        jbase,jsize=spec.FAMILIES['J']; rbase,rsize=spec.FAMILIES['R']
        for end in range(start,0,-65536):
            lo=max(0,end-65536); block=tokens[lo:end]
            anchors=np.flatnonzero(((block>=jbase)&(block<jbase+jsize))|((block>=rbase)&(block<rbase+rsize)))
            if len(anchors):
                begin=lo+int(anchors[-1]); break
    for i in range(begin,stop):
        token = int(tokens[i])
        fam,payload = spec.family(token)
        pc = 0x40000+4*(anchor+count)
        if i >= start: pairs[i-start] = token,pc
        if fam in ('J','R'): anchor,count = payload,0
        elif fam in spec.FINAL and fam != 'HALT': count += 1
    return pairs


def neural_snapshot(fr,c,offset,selected=('pc','count','regs.rs1','ram3','drom0')):
    from chipformer.q35.ir import A_BIT
    from chipformer.rv import spec
    sys.path.insert(0,str(ROOT/'experiments/tiny-llm-ndot'))
    from worker import read_registers
    sys.path.insert(0,str(ROOT/'experiments/dmc-cuda/source/helpers'))
    from state import read_ram
    tables = {}
    for step in fr.ex.steps:
        if step['kind'] != 'head': continue
        base = 0
        for h in step['heads']:
            if h.name in (*selected,'pc','count'):
                tables[h.name] = step['S'][0,base].detach().cpu().numpy()
            base += len(h.vheads())
    bits = tables['pc'][0,2:18]/A_BIT
    assert np.isin(bits,[0,1]).all()
    anchor = sum(int(b)<<i for i,b in enumerate(bits))
    count = float(tables['count'][0,2])/181
    assert count.is_integer() and 0 <= count <= 63
    regs = read_registers(fr,c)
    ram = read_ram(fr.ex,spec.RAM_BASE,65536)
    slices = {name:dict(provenance=NEURAL,shape=[128,128],values=value.tolist(),
                        value_head=0,scale='raw fp32 S, includes ballast columns 0 and 1') for name,value in tables.items() if name in selected}
    return dict(offset=offset,provenance=NEURAL,pc=0x40000+4*(anchor+int(count)),anchor=anchor,count=int(count),
                registers=regs,ram_hex=ram.hex(),slices=slices)


def decode_app(snapshot,inputs,program):
    if not (inputs/'pack.json').exists():
        from showcase_data import layout, snapshot_app
        config=layout(inputs,program)
        return snapshot_app(snapshot,config) if config else None
    pack=json.loads((inputs/'pack.json').read_text())
    if pack.get('kind')=='Qwen3_5ForCausalLM':
        from tiny_qwen import decode
        return decode(snapshot,pack,program)
    symbols={s['name']:s for s in program['symbols']}
    ram=bytes.fromhex(snapshot['ram_hex'])
    arrays={}
    for name in ('x','xb','xb2','q','tmp','hb','hb2','logits','att','quant','keys','values'):
        if name not in symbols: continue
        sym=symbols[name]
        address=sym['address']; size=sym['size']
        fmt=('b' if pack['kv_bits']==8 else 'h') if name in ('keys','values') else ('h' if name=='quant' else 'i')
        width=struct.calcsize(fmt)
        off=address-0x1000000
        if off < 0 or off+size > len(ram): continue
        vals=list(struct.unpack_from('<'+str(size//width)+fmt,ram,off))
        arrays[name]=dict(address=address,values=vals,format=fmt,provenance=NEURAL,
            fractional_bits=pack['kv_fractional_bits'] if name in ('keys','values') else (None if name=='quant' else 15 if name=='att' else 12),
            shape=[pack['layers'],pack['context'],pack['dim']*pack['kvheads']//pack['heads']] if name in ('keys','values') else [len(vals)])
    return dict(kind='tiny-llm',pack=pack,arrays=arrays,provenance=NEURAL,offset=snapshot['offset'])


def bundle(out,name,pairs,start,program,snapshots,inputs,binding):
    from chipformer.rv import spec
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/'program.json.gz',program)
    (out/f'{name}.trace.u32.gz').write_bytes(gzip.compress(pairs.tobytes(),mtime=0))
    functions=sorted([s for s in program['symbols'] if s['kind']==2],key=lambda s:s['address'])
    starts=[s['address'] for s in functions]
    code={x[0]:x for x in program['instructions']}
    bins=[]
    for lo in range(0,len(pairs),max(1,(len(pairs)+255)//256)):
        chunk=pairs[lo:lo+max(1,(len(pairs)+255)//256)]
        hot=Counter(); ndot=0
        for pc in chunk[:,1]:
            j=bisect_right(starts,int(pc))-1
            f=functions[j] if j>=0 else None
            label=f['name'] if f and int(pc)<f['address']+f['size'] else 'unresolved'
            hot[label]+=1
            ndot+=int(code.get(int(pc),[0,0,''])[2].startswith('NDOT'))
        bins.append(dict(offset=start+lo,hot=hot.most_common(5),ndot=ndot,tokens=len(chunk)))
    byte_events=[]
    for i,t in enumerate(pairs[:,0]):
        f,p=spec.family(int(t))
        if f=='BYTE' and p<256: byte_events.append([start+i+1,p])
    snapshot_refs=[]
    for snapshot in snapshots:
        snapshot['app']=decode_app(snapshot,inputs,program)
        path=f'{name}.snapshot-{snapshot["offset"]}.json.gz'
        write_json(out/path,snapshot)
        snapshot_refs.append(dict(offset=snapshot['offset'],path=path,provenance=NEURAL))
    app_kind=snapshots[0]['app']['kind'] if snapshots and snapshots[0].get('app') else 'tiny-llm'
    record=dict(schema='qwenputer.recording.v1',id=name,title=name.replace('-',' '),start=start,end=start+len(pairs),
        trace=f'{name}.trace.u32.gz',trace_format='Interleaved little-endian uint32 [generated token, decoded instruction PC]',
        program='program.json.gz',snapshots=snapshot_refs,bins=bins,byte_events=byte_events,
        provenance=TRACE,binding=binding,token_families=list(spec.FAMILY_SIZES),controls=spec.CONTROL_NAMES,
        final_families=sorted(spec.FINAL),app_kind=app_kind,
        coverage='Full generated token window. Neural snapshots exist only at listed exact offsets. No state interpolation.',
        pack=json.loads((inputs/'pack.json').read_text()) if (inputs/'pack.json').exists() else None)
    write_json(out/f'{name}.json',record)
    return dict(id=name,title=record['title'],url=f'{name}.json',tokens=len(pairs),snapshots=len(snapshots),elf_sha256=program['elf_sha256'])


def import_run(a):
    c=compiler()
    from chipformer.rv import spec
    import torch
    actual=np.memmap(a.actual/'actual.tokens.u32',dtype='<u4',mode='r')
    expected=np.memmap(a.inputs/'reference.tokens.u32',dtype='<u4',mode='r')
    assert len(actual)==len(expected)
    for lo in range(0,len(actual),1<<20): assert np.array_equal(actual[lo:lo+(1<<20)],expected[lo:lo+(1<<20)])
    ref=json.loads((a.inputs/'reference.json').read_text())
    assert sha(a.inputs/'engine.elf')==ref['elf_sha256']
    assert sha(a.inputs/'reference.tokens.u32')==ref['sha256']
    assert sha(a.actual/'actual.tokens.u32')==ref['sha256']
    program=program_info(a.inputs)
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    assert checkpoint['offset']==len(actual) and checkpoint['binding']['elf']==program['elf_sha256']
    assert checkpoint['binding']['reference']==ref['sha256']
    steps=[]; index=0
    from chipformer.q35.ir import Head
    for L in range(c.n_layers()):
        heads=[h for h in c.nodes if isinstance(h,Head) and h.level==2*L+1]
        if heads:
            steps.append(dict(kind='head',heads=heads,**checkpoint['heads'][index])); index+=1
    assert index==len(checkpoint['heads'])
    final=neural_snapshot(SimpleNamespace(ex=SimpleNamespace(steps=steps)),c,len(actual))
    assert bytes.fromhex(final['ram_hex'])==(a.actual/'actual-ram.bin').read_bytes()==(a.inputs/'reference-ram.bin').read_bytes()
    actual_regs=json.loads((a.actual/'actual-registers.json').read_text())
    assert final['registers']==actual_regs
    assert all(regs==ref['final_registers'] for regs in final['registers'].values())
    print('Observed neural PC',hex(final['pc']),'reference halted PC',hex(ref['final_pc']),flush=True)
    assert int(actual[-1])==spec.tok('HALT')
    assert final['pc']+4==ref['final_pc'], 'HALT is pending feedback in FreeRun'
    final['pending_token']=int(actual[-1])
    final['state_boundary']='After consuming all but the last generated token; last generated token awaits feedback.'
    binding=dict(actual_trace_sha256=sha(a.actual/'actual.tokens.u32'),checkpoint_sha256=sha(a.checkpoint),
        validation='Every generated token equals retained independent trace; all RAM bytes and all 96 register values equal validated exports. Neural PC equals the pending HALT instruction PC, four bytes before the post-HALT reference PC.',
        source='Retained autonomous fused 38-position stories260K run',reference_used='comparison only',
        checkpoint_offset=len(actual),elf_sha256=program['elf_sha256'])
    first=ref['positions'][0]
    stop=ref['positions'][1]['end_token']-ref['positions'][1]['trace_tokens']
    pairs=decode_trace(actual,spec,program['entry'],stop=stop)
    entries=[bundle(a.out,'boot',pairs[:8192],0,program,[],a.inputs,binding),
             bundle(a.out,'stories260k-position-0',pairs,0,program,[],a.inputs,binding)]
    start=len(actual)-4096
    tail=decode_trace(actual,spec,program['entry'],start=start)
    assert int(tail[-1,1])==final['pc']
    entries.append(bundle(a.out,'stories260k-final',tail,start,program,[final],a.inputs,binding))
    final_record=json.loads((a.out/'stories260k-final.json').read_text())
    byte_base,_=spec.FAMILIES['BYTE']
    prior=actual[:start]; positions=np.flatnonzero((prior>=byte_base)&(prior<byte_base+256))
    final_record['prefix_byte_events']=[[int(i)+1,int(prior[i])-byte_base] for i in positions]
    write_json(a.out/'stories260k-final.json',final_record)
    write_json(a.out/'index.json',dict(schema='qwenputer.catalog.v1',recordings=entries))
    write_json(HERE/'data/import-validation.json',dict(**binding,final_pc=final['pc'],final_ram_sha256=sha(a.actual/'actual-ram.bin'),snapshots=1))
    print(json.dumps(entries,indent=2))



def reference_checker(inputs):
    import copy
    import reference
    from chipformer.rv import elf
    machine=reference.Machine.from_program(elf.program(inputs/'engine.elf'))
    machine.maximum=4
    consumed=0
    pending=[]
    future=None
    def check(snapshot,target):
        nonlocal machine,consumed,pending,future
        while consumed<target:
            if not pending:
                future=copy.deepcopy(machine) if target-consumed<70 else machine
                pending=future.record()
                if future is machine:
                    consumed+=len(pending);pending=[]
                    continue
            amount=min(target-consumed,len(pending))
            pending=pending[amount:];consumed+=amount
            if not pending:
                machine=future;future=None
        assert all(regs==machine.x for regs in snapshot['registers'].values()),('registers',target)
        expected=np.asarray(machine.ram,dtype='<u4').tobytes()
        assert bytes.fromhex(snapshot['ram_hex'])==expected,('RAM',target)
        assert snapshot['pc']==machine.pc,('PC',target,snapshot['pc'],machine.pc)
        assert snapshot['anchor']==machine.anchor and snapshot['count']==machine.n,('anchor/count',target)
    return check

def capture(a):
    import torch
    sys.path.insert(0,str(ROOT/'experiments/tiny-llm-ndot-fused'))
    import story
    ref=json.loads((a.inputs/'reference.json').read_text())
    assert sha(a.inputs/'engine.elf')==ref['elf_sha256']
    assert sha(a.inputs/'reference.tokens.u32')==ref['sha256']
    args=SimpleNamespace(repo=ROOT,upstream=a.upstream,out=a.out,inputs=a.inputs,original_circuit=False)
    if a.stop is not None and not 0<a.stop<=ref['tokens']: raise ValueError('--stop outside retained reference comparison bounds')
    fr,info,spec,meta=story.build(args,ref)
    expected=np.memmap(a.inputs/'reference.tokens.u32',dtype='<u4',mode='r')
    target=a.stop or (ref['positions'][1]['end_token']-ref['positions'][1]['trace_tokens'] if 'positions' in ref else ref['tokens'])
    offset=0; generated=[]; snapshots=[]
    check=reference_checker(a.inputs)
    with torch.no_grad():
        while offset<target:
            amount=min(a.every-offset%a.every,target-offset)
            part=fr.run(amount)[0].astype('<u4')
            assert np.array_equal(part,expected[offset:offset+amount]),f'Generated mismatch at {offset}'
            generated.append(part); offset+=amount
            s=neural_snapshot(fr,info['circuit'],offset)
            check(s,offset-1)
            s['verification']=dict(passed=True,reference_consumed_tokens=offset-1,ram_bytes=65536,register_values=96,pc=True,anchor=True,count=True)
            s['pending_token']=int(part[-1]); s['state_boundary']='Last generated token awaits feedback. Exact S state after the previous generated token.'
            snapshots.append(s)
            print(json.dumps(dict(offset=offset,target=target)),flush=True)
    actual=np.concatenate(generated)
    a.out.mkdir(exist_ok=True,parents=True)
    actual.tofile(a.out/'actual.tokens.u32')
    program=program_info(a.inputs)
    pairs=decode_trace(actual,spec,program['entry'])
    entry=bundle(a.out,'fused-capture',pairs,0,program,snapshots,a.inputs,
        dict(source=meta,elf_sha256=program['elf_sha256'],actual_trace_sha256=sha(a.out/'actual.tokens.u32'),
             reference_trace_sha256=ref['sha256'],validation='Every generated token equals independent comparison trace. Snapshots decoded directly from actual executor S.',reference_used='comparison only'))
    from showcase_data import layout, trace_application
    config=layout(a.inputs,program) if not (a.inputs/'pack.json').exists() else None
    if config:
        application=trace_application(pairs,program,a.inputs,config)
        replay_ram=application.pop('final_ram');replay_registers=application.pop('final_registers')
        if offset==ref['tokens']:
            assert replay_ram==bytes.fromhex(snapshots[-1]['ram_hex'])
            assert all(replay_registers==r for r in snapshots[-1]['registers'].values())
        record=json.loads((a.out/'fused-capture.json').read_text())
        record['application']=application;record['app_kind']=config['kind']
        write_json(a.out/'fused-capture.json',record)
    write_json(a.out/'capture-validation.json',dict(passed=True,tokens=offset,mismatches=0,
        every=a.every,snapshots=[dict(offset=s['offset'],**s['verification']) for s in snapshots],
        reference_used='comparison only',boot_state_only=True))
    write_json(a.out/'index.json',dict(schema='qwenputer.catalog.v1',recordings=[entry]))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['import','capture'])
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--actual',type=Path)
    p.add_argument('--upstream',type=Path,default=ROOT/'.upstream/qwenputer')
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--out',type=Path,default=HERE/'data/recordings')
    p.add_argument('--every',type=int,default=32768)
    p.add_argument('--stop',type=int)
    a=p.parse_args()
    if a.every<=0: p.error('--every must be positive')
    if a.mode=='import':
        if not a.actual or not a.checkpoint: p.error('import needs --actual and --checkpoint')
        import_run(a)
    else: capture(a)
