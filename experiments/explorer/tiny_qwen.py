"""Decode the data-driven Qwen guest ABI directly from an observed RAM sample."""
import struct
from record import NEURAL


def decode(snapshot, pack, program):
    header=pack['header']
    arena=next(s for s in program['symbols'] if s['name']=='arena')
    ram=bytes.fromhex(snapshot['ram_hex'])
    base=arena['address']-0x1000000
    assert arena['size']>=header['ram_bytes'] and 0<=base<=len(ram)-arena['size']
    arrays={}
    def array(name,offset,n,fmt='i',frac=12,shape=None,layer=None):
        width=struct.calcsize(fmt)
        assert 0<=offset and offset+n*width<=header['ram_bytes']
        arrays[name]=dict(address=arena['address']+offset,values=list(struct.unpack_from('<'+str(n)+fmt,ram,base+offset)),
                          format=fmt,fractional_bits=frac,shape=shape or [n],layer=layer,provenance=NEURAL)
    for name,n in [('x',header['dim']),('logits',header['vocab'])]:
        array(name,header[name],n)
    layers=[]
    for i,l in enumerate(pack['graph']):
        kind='DeltaNet' if l['kind']==3 else 'Full attention'
        layers.append(dict(index=i,kind=kind))
        if l['kind']==3:
            n=l['value_heads']*l['key_dim']*l['value_dim']
            array(f'layer-{i}-state',l['state'],n,frac=20,shape=[l['value_heads'],l['key_dim'],l['value_dim']],layer=i)
            n=(2*l['key_heads']*l['key_dim']+l['value_heads']*l['value_dim'])*(l['kernel']-1)
            array(f'layer-{i}-convolution',l['conv_state'],n,layer=i)
        else:
            dim=l['kvheads']*l['head_dim']; context=header['context']
            for name in ('keys','values'):
                array(f'layer-{i}-{name}',l[name],context*dim,'h',header['kv_frac'],[context,dim],i)
    return dict(kind='tiny-qwen35',pack=pack,arrays=arrays,layers=layers,provenance=NEURAL,offset=snapshot['offset'])
