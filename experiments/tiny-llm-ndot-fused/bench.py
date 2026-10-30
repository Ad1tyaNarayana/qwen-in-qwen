"""Five alternating normal-FreeRun timings, gated by full isolated raw bits."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
import runtime


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--gate',type=Path,required=True)
    p.add_argument('--tokens',type=int,default=65536)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    comparison=json.loads((a.gate/'comparison.json').read_text())
    assert all(w['tokens']>=65536 and w['digest_mismatches']==0 for w in comparison['windows'].values())
    for side in ('baseline','fused'):
        gate=json.loads((a.gate/side/'isolation.json').read_text())
        assert gate['one_bit_recording_control']
        assert all(w['reference_mismatches']==0 and w['first_state_mismatch'] is None and w['final_S_reconstruction_equal'] for w in gate['windows'].values())
        for name,digest in gate['scripts'].items():
            assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==digest
    assert json.loads((a.gate/'feedback.json').read_text())['mismatched_values']==0
    assert json.loads((a.gate/'apply.json').read_text())['kernels']['barrier']['mismatched_words']==0
    import torch
    import numpy as np
    instances={}
    for side in ('baseline','fused'):
        work=a.out/side
        work.mkdir()
        fr,info,spec,metadata=runtime.build(work,a.inputs)
        from fusion import install
        fused=install(fr) if side=='fused' else None
        fr.capture()
        instances[side]=(fr,fused)
    fixtures,offset=runtime.fixtures(fr,a.inputs,a.tokens)
    report=dict(runs=[],windows={},seed_offset=offset,
        timer='FreeRun.run including self-feedback, synchronization, token copy and CPU conversion; excludes reset, comparison and setup',
        scripts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')})
    for window,machine,want in fixtures:
        for repeat in range(5):
            for side,(fr,fused) in instances.items():
                runtime.reset(fr,machine,fused)
                torch.cuda.synchronize()
                started=time.perf_counter()
                actual=fr.run(a.tokens)[0]
                seconds=time.perf_counter()-started
                mismatches=int(np.count_nonzero(actual!=want))
                assert mismatches==0
                run=dict(window=window,side=side,repeat=repeat,tokens=len(actual),seconds=seconds,
                         us_per_token=seconds/len(actual)*1e6,tokens_per_second=len(actual)/seconds,mismatches=mismatches,
                         actual_sha256=hashlib.sha256(actual.astype('<u4').tobytes()).hexdigest())
                report['runs'].append(run)
                (a.out/'bench.json').write_text(json.dumps(report,indent=2)+'\n')
                print(json.dumps(run),flush=True)
        report['windows'][window]={}
        for side in instances:
            values=[x['us_per_token'] for x in report['runs'] if x['window']==window and x['side']==side]
            median=statistics.median(values)
            report['windows'][window][side]=dict(median_us_per_token=median,range_us_per_token=[min(values),max(values)],
                projected_min_per_position=median*5423298/5/60/1e6,
                projection='Five-position Once workload mean, bounded window rate; not a full execution')
    report['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
    (a.out/'bench.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
