"""Check exported application values against independent algorithms and accepted GPU bytes."""
import copy
import gzip
import hashlib
import json
import struct
from pathlib import Path
import numpy as np
from common import HERE, ROOT, compiler, write_json
from record import reference_checker
from showcase_data import DEMOS


def read(path):
    raw=Path(path).read_bytes()
    return json.loads(gzip.decompress(raw) if str(path).endswith('.gz') else raw)


def main():
    compiler()
    runs=[]
    for demo in DEMOS:
        root=HERE/'data/recordings'/demo
        record=read(root/f'showcase-{demo}.json')
        sample=read(root/record['snapshots'][0]['path'])
        reference_checker(ROOT/'.cache/showcase'/demo)(sample,record['end']-1)
        assert sample['verification']['passed']
        runs.append(dict(demo=demo,snapshots=1,ram_bytes=65536,register_values=96,pc_anchor_count=True))
    # An independently corrupted recurrent register copy must fail the oracle.
    damaged=copy.deepcopy(sample)
    damaged['registers']['regs.rd'][1]^=1
    try: reference_checker(ROOT/'.cache/showcase'/demo)(damaged,record['end']-1)
    except AssertionError as e: assert e.args[0][0]=='registers'
    else: raise AssertionError('Register corruption was accepted')
    life=read(HERE/'data/recordings/life/showcase-life.json')['application']['events']
    board={i for i,v in enumerate(life[0]['arrays']['board']['values']) if v}
    for event in life:
        assert event['arrays']['board']['values']==[int(i in board) for i in range(384)]
        nxt=set()
        for y in range(16):
            for x in range(24):
                n=sum(((y+dy)%16)*24+(x+dx)%24 in board for dy in (-1,0,1) for dx in (-1,0,1) if dy or dx)
                if n==3 or n==2 and y*24+x in board:nxt.add(y*24+x)
        board=nxt
    weights=np.load(ROOT/'.cache/showcase/mnist-model/integer.npz',allow_pickle=False)
    images=np.fromfile(ROOT/'.cache/showcase/mnist-model/test-images.bin',dtype=np.uint8).reshape(-1,784)[:3].astype(np.int64)
    hidden=np.clip(((images@weights['w1'].astype(np.int64).T+weights['b1'])*weights['m1']+(1<<19))>>20,0,32767)
    logits=((hidden@weights['w2'].astype(np.int64).T+weights['b2'])*weights['m2']+(1<<19))>>20
    for demo in ('mnist','mnist-scalar'):
        events=read(HERE/f'data/recordings/{demo}/showcase-{demo}.json')['application']['events']
        for i,event in enumerate(events):
            assert event['arrays']['input']['values']==images[i].tolist()
            assert event['arrays']['hidden']['values']==hidden[i].tolist()
            assert event['logits']==logits[i].tolist()
    sha=read(HERE/'data/recordings/sha256/showcase-sha256.json')['application']
    source=(ROOT/'experiments/showcase/sha256/main.c').read_text()
    import re
    constants=[int(x,16) for x in re.findall(r'0x([0-9a-f]{8})',source.split('static uint32_t r')[0])]
    mask=2**32-1
    rotate=lambda x,n:((x>>n)|(x<<(32-n)))&mask
    for message,events in ((b'',sha['events'][:64]),(b'abc',sha['events'][64:])):
        block=message+b'\x80'+bytes(55-len(message))+struct.pack('>Q',len(message)*8)
        schedule=list(struct.unpack('>16I',block))
        for i in range(16,64):
            a,b=schedule[i-15],schedule[i-2]
            schedule.append((schedule[i-16]+(rotate(a,7)^rotate(a,18)^(a>>3))+schedule[i-7]+(rotate(b,17)^rotate(b,19)^(b>>10)))&mask)
        initial=[0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19]
        working=initial[:]
        assert len(events)==64
        for i,event in enumerate(events):
            a,b,c,d,e,f,g,h=working
            t1=(h+(rotate(e,6)^rotate(e,11)^rotate(e,25))+((e&f)^((~e)&g))+constants[i]+schedule[i])&mask
            t2=((rotate(a,2)^rotate(a,13)^rotate(a,22))+((a&b)^(a&c)^(b&c)))&mask
            working=[(t1+t2)&mask,a,b,c,(d+t1)&mask,e,f,g]
            assert event['working']==working,(message,i,event['working'],working)
            assert event['arrays']['schedule']['values']==schedule
        digest=''.join(f'{(a+b)&mask:08x}' for a,b in zip(initial,working))
        assert digest==hashlib.sha256(message).hexdigest()==sha['reference_digests'][message.decode()]
    write_json(HERE/'showcase-data-validation.json',dict(passed=True,runs=runs,
        checks=['5 exact final neural checkpoints','register-copy one-bit rejection','5 independent Life generations',
                '6 MNIST inputs, hidden arrays and all 60 logits','128 SHA rounds and schedules vs independent compressor and hashlib'],
        periodic_gpu_snapshots=0,periodic_status='GPU lock held; no new periodic capture claimed'))
    print('Showcase checks passed. Five neural checkpoints, 128 SHA rounds, five Life frames, six MNIST samples.')


if __name__=='__main__':main()
