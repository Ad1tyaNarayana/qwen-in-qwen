"""Recheck a collected GPU capture on the Mac and publish only its compressed bundle."""
import argparse
import gzip
import json
import shutil
from pathlib import Path
import numpy as np
from common import HERE, ROOT, compiler, sha, write_json
from record import reference_checker


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--demo',required=True,choices=['mnist','life','stories260k'])
    p.add_argument('--bundle',required=True,type=Path)
    a=p.parse_args()
    compiler()
    inputs=ROOT/('.cache/tiny-llm/ndot-story38-20261030' if a.demo=='stories260k' else '.cache/showcase/'+a.demo)
    record=json.loads((a.bundle/'fused-capture.json').read_text())
    report=json.loads((a.bundle/'capture-validation.json').read_text())
    assert report['passed'] and report['boot_state_only'] and report['mismatches']==0
    assert record['binding']['elf_sha256']==sha(inputs/'engine.elf')
    actual=np.fromfile(a.bundle/'actual.tokens.u32',dtype='<u4')
    expected=np.memmap(inputs/'reference.tokens.u32',dtype='<u4',mode='r')
    assert len(actual)==record['end']==report['tokens'] and np.array_equal(actual,expected[:len(actual)])
    assert sha(a.bundle/'actual.tokens.u32')==record['binding']['actual_trace_sha256']
    assert sha(inputs/'reference.tokens.u32')==record['binding']['reference_trace_sha256']
    pairs=np.frombuffer(gzip.decompress((a.bundle/record['trace']).read_bytes()),dtype='<u4').reshape(-1,2)
    assert np.array_equal(pairs[:,0],actual)
    check=reference_checker(inputs)
    offsets=[]
    for item,verification in zip(record['snapshots'],report['snapshots'],strict=True):
        snapshot=json.loads(gzip.decompress((a.bundle/item['path']).read_bytes()))
        assert snapshot['offset']==item['offset']==verification['offset']
        assert snapshot['verification']['passed'] and verification['passed']
        check(snapshot,snapshot['offset']-1)
        offsets.append(snapshot['offset'])
    assert offsets==list(range(report['every'],len(actual),report['every']))+[len(actual)]
    out=HERE/'data/recordings'/('periodic-'+a.demo)
    out.mkdir(exist_ok=True)
    for name in [record['trace'],record['program'],'fused-capture.json','capture-validation.json',*(s['path'] for s in record['snapshots'])]:
        shutil.copyfile(a.bundle/name,out/name)
    catalog=json.loads((HERE/'data/recordings/index.json').read_text())
    id='periodic-'+a.demo
    record['id']=id;record['title']=a.demo+' periodic neural capture'
    write_json(out/'fused-capture.json',record)
    entry=dict(id=id,title=record['title'],url=id+'/fused-capture.json',tokens=len(actual),snapshots=len(offsets),elf_sha256=record['binding']['elf_sha256'])
    catalog['recordings']=[r for r in catalog['recordings'] if r['id']!=id]+[entry]
    write_json(HERE/'data/recordings/index.json',catalog)
    manifest=ROOT/'experiments/showcase/manifest'/('stories260k-38.json' if a.demo=='stories260k' else a.demo+'.json')
    m=json.loads(manifest.read_text());m['explorer_recording']=entry['url'];write_json(manifest,m)
    write_json(out/'mac-validation.json',dict(passed=True,tokens=len(actual),mismatches=0,snapshots=len(offsets),offsets=offsets,
        reference_used='comparison only',scope='Every actual token and every neural snapshot rechecked independently on Mac; all RAM, registers, PC, anchor and count.',
        compressed_bytes=sum(p.stat().st_size for p in out.iterdir() if p.is_file())))
    print(json.dumps(entry),flush=True)

if __name__=='__main__':main()
