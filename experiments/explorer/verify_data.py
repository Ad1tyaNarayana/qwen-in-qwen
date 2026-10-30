"""Verify real exported graph, generated window PCs, and retained neural state evidence."""
import copy
import gzip
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from common import HERE,ROOT,compiler,write_json
from record import decode_trace,reference_checker


def read(path):
    raw=Path(path).read_bytes()
    return json.loads(gzip.decompress(raw) if str(path).endswith('.gz') else raw)


def main():
    import torch
    torch.set_num_threads(2)
    c=compiler()
    from chipformer.q35.ir import FFN,Head
    from chipformer.q35 import emit
    from chipformer.q35.tiny import BASE,pick_layout
    from chipformer.rv import spec,elf
    import reference
    graph=read(HERE/'data/graph.json.gz')
    assert graph['geometry']==c.stats()
    assert graph['ledger']==c.reuse_report
    assert sum(len(x) for x in graph['dimensions'])==len(c.reuse_report['lifetimes'])
    for h in graph['ledger']['handovers']:
        owners=graph['dimensions'][h['slot']]
        assert any(x['dim']==h['old'] for x in owners) and any(x['dim']==h['new'] for x in owners)
        assert h['last']<=h['clear']<h['born']
    L=8
    nodes=[copy.copy(n) for n in c.nodes if n.level in (2*L+1,2*L+2)]
    for n in nodes:n.level=1 if isinstance(n,Head) else 2
    one=SimpleNamespace(nodes=nodes,n_layers=lambda:1)
    lay=pick_layout(len(c.dims))
    config=dict(BASE,hidden_size=lay.hidden,linear_num_key_heads=16,linear_num_value_heads=32,intermediate_size=9216)
    emitted=emit.emit_layers(one,lay,config,0,'model.')
    detail=read(HERE/f'data/layers/{L}.json.gz')
    chosen=sorted({0,1,10,len(detail['units'])-1})
    for u in chosen:
        values=torch.cat([emitted['model.layers.0.mlp.gate_proj.weight'][u],emitted['model.layers.0.mlp.up_proj.weight'][u],emitted['model.layers.0.mlp.down_proj.weight'][:,u]]).double().numpy()
        want=detail['units'][u][6]
        assert [len(values),int(np.count_nonzero(values)),float(values.min()),float(values.max())]==want[:4]
        assert np.isclose(np.linalg.norm(values),want[4],rtol=1e-14)
    del emitted
    exit_nodes=[copy.copy(n) for n in c.nodes if n.level in (127,128)]
    for n in exit_nodes:n.level=1 if isinstance(n,Head) else 2
    exit_graph=SimpleNamespace(nodes=exit_nodes,dims=c.dims,n_layers=lambda:1)
    exit_weights=emit.emit_layers(exit_graph,lay,config,0,'model.',keep=set(c.keep))
    last_detail=read(HERE/'data/layers/63.json.gz')
    last_unit=len(last_detail['units'])-1
    values=torch.cat([exit_weights['model.layers.0.mlp.gate_proj.weight'][last_unit],exit_weights['model.layers.0.mlp.up_proj.weight'][last_unit],exit_weights['model.layers.0.mlp.down_proj.weight'][:,last_unit]]).double().numpy()
    want=last_detail['units'][last_unit][6]
    assert [len(values),int(np.count_nonzero(values)),float(values.min()),float(values.max())]==want[:4]
    assert np.isclose(np.linalg.norm(values),want[4],rtol=1e-14)
    del exit_weights
    fixture=ROOT/'.cache/tiny-llm/ndot-story38-20261030'
    catalog=read(HERE/'data/recordings/index.json')
    first=read(HERE/'data/recordings/stories260k-position-0.json')
    raw=gzip.decompress((HERE/'data/recordings'/first['trace']).read_bytes())
    pairs=np.frombuffer(raw,dtype='<u4').reshape(-1,2)
    m=reference.Machine.from_program(elf.program(fixture/'engine.elf'));m.maximum=4
    offset=0;records=0
    while offset<len(pairs):
        pc=m.pc
        tokens=m.record()
        amount=min(len(tokens),len(pairs)-offset)
        assert np.array_equal(pairs[offset:offset+amount,0],tokens[:amount])
        assert np.all(pairs[offset:offset+amount,1]==pc)
        offset+=amount;records+=1
    snapshot_ref=read(HERE/'data/recordings/stories260k-final.json')['snapshots'][0]
    snapshot=read(HERE/'data/recordings'/snapshot_ref['path'])
    actual=ROOT/'.cache/tiny-llm/ndot-story38-20261030-actual/main'
    assert bytes.fromhex(snapshot['ram_hex'])==(actual/'actual-ram.bin').read_bytes()
    assert snapshot['registers']==read(actual/'actual-registers.json')
    # The checker must accept actual prefix state and reject a register corruption.
    checker=reference_checker(fixture)
    boot=reference.Machine.from_program(elf.program(fixture/'engine.elf'));boot.maximum=4
    sample=dict(registers={p:list(boot.x) for p in ('regs.rs1','regs.rs2','regs.rd')},ram_hex=np.asarray(boot.ram,dtype='<u4').tobytes().hex(),pc=boot.pc,anchor=boot.anchor,count=boot.n)
    checker(sample,0)
    sample['registers']['regs.rd'][7]^=1
    try:checker(sample,0)
    except AssertionError as e:assert e.args[0][0]=='registers'
    else:raise AssertionError('Missed register corruption')
    report=dict(passed=True,graph_geometry=graph['geometry'],ledger_owners_checked=sum(len(x) for x in graph['dimensions']),
        dense_stock_layer=L,dense_stock_units_checked=chosen,dense_stock_ballast_cleanup_checked=last_unit,generated_window_tokens_checked=len(pairs),reference_records_checked=records,
        final_ram_bytes_checked=65536,final_register_values_checked=96,corruption_control='one changed register copy rejected',
        pc_boundary='Saved FreeRun S is before the last generated token is fed back. Pending HALT PC 0x40014; post-HALT reference PC 0x40018.')
    write_json(HERE/'data-validation.json',report)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
