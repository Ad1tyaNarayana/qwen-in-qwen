"""Import accepted GPU traces and actual final S checkpoints on CPU, never synthesize S."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from common import HERE, ROOT, compiler, sha, write_json
from record import neural_snapshot, program_info, decode_trace, bundle, reference_checker
from showcase_data import DEMOS, layout, trace_application


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--actual',type=Path,default=ROOT/'.cache/showcase/actual-20261030a')
    p.add_argument('--checkpoints',type=Path,default=ROOT/'.cache/explorer/showcase-checkpoints')
    a=p.parse_args()
    import torch
    torch.set_num_threads(2)
    circuit=compiler()
    from chipformer.q35.ir import Head
    from chipformer.rv import spec
    catalog=json.loads((HERE/'data/recordings/index.json').read_text())
    catalog['recordings']=[r for r in catalog['recordings'] if not r['id'].startswith('showcase-')]
    reports=[]
    for demo in DEMOS:
        inputs=ROOT/'.cache/showcase'/demo
        actual=a.actual/demo/'fused'
        checkpoint=a.checkpoints/demo/'fused/checkpoint.pt'
        ref=json.loads((inputs/'reference.json').read_text())
        program=program_info(inputs)
        assert program['elf_sha256']==ref['elf_sha256']
        tokens=np.fromfile(actual/'actual.tokens.u32',dtype='<u4')
        expected=np.fromfile(inputs/'reference.tokens.u32',dtype='<u4')
        assert np.array_equal(tokens,expected) and len(tokens)==ref['tokens']
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        assert saved['offset']==len(tokens)
        assert saved['binding']['elf']==program['elf_sha256']
        assert saved['binding']['reference']==sha(actual/'actual.tokens.u32')==ref['sha256']
        steps=[]
        for layer in range(circuit.n_layers()):
            heads=[h for h in circuit.nodes if isinstance(h,Head) and h.level==2*layer+1]
            if heads: steps.append(dict(kind='head',heads=heads,**saved['heads'][len(steps)]))
        assert len(steps)==len(saved['heads'])
        snapshot=neural_snapshot(SimpleNamespace(ex=SimpleNamespace(steps=steps)),circuit,len(tokens))
        assert bytes.fromhex(snapshot['ram_hex'])==(actual/'actual-ram.bin').read_bytes()==(inputs/'reference-ram.bin').read_bytes()
        assert snapshot['registers']==json.loads((actual/'actual-registers.json').read_text())
        reference_checker(inputs)(snapshot,len(tokens)-1)
        snapshot['pending_token']=int(tokens[-1])
        snapshot['state_boundary']='Retained accepted GPU checkpoint. Last generated HALT awaits feedback.'
        snapshot['verification']=dict(passed=True,reference_consumed_tokens=len(tokens)-1,ram_bytes=65536,register_values=96,pc=True,anchor=True,count=True)
        pairs=decode_trace(tokens,spec,program['entry'])
        config=layout(inputs,program)
        application=trace_application(pairs,program,inputs,config)
        assert application.pop('final_ram')==(actual/'actual-ram.bin').read_bytes(), 'Trace replay RAM'
        assert application.pop('final_registers')==ref['final_registers'], 'Trace replay registers'
        if demo.startswith('mnist'):
            assert len(application['events'])==3
            assert [max(range(10),key=lambda j:e['logits'][j]) for e in application['events']]==[7,2,1]
        out=HERE/'data/recordings'/demo
        entry=bundle(out,'showcase-'+demo,pairs,0,program,[snapshot],inputs,
            dict(actual_trace_sha256=sha(actual/'actual.tokens.u32'),checkpoint_sha256=sha(checkpoint),
                 elf_sha256=program['elf_sha256'],network_weight_sha256=config['network_weight_sha256'],
                 source='Accepted showcase autonomous fused GPU run; final checkpoint imported on CPU',
                 reference_used='comparison only',validation=snapshot['verification']))
        entry['url']=demo+'/'+entry['url']
        record=json.loads((out/('showcase-'+demo+'.json')).read_text())
        record['application']=application
        write_json(out/('showcase-'+demo+'.json'),record)
        catalog['recordings'].append(entry)
        reports.append(dict(demo=demo,tokens=len(tokens),mismatches=0,snapshots=1,application_events=len(application['events']),
                            checkpoint_sha256=sha(checkpoint),verification=snapshot['verification'],trace_replay_matches_neural_ram=True))
        print(json.dumps(reports[-1]),flush=True)
    write_json(HERE/'data/recordings/index.json',catalog)
    write_json(HERE/'data/showcase-validation.json',dict(passed=True,source='Retained final GPU checkpoints, not new periodic captures',runs=reports))


if __name__=='__main__':main()
