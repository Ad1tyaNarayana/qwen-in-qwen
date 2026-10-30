"""Train on bounded, hash-checked IDX data; export integer MLP coefficients."""
import gzip
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'.cache/showcase/mnist-model'
DATA=ROOT/'.cache/showcase/mnist-downloads'
FILES={
 'train-images-idx3-ubyte.gz':('f68b3c2dcbeaaa9fbdd348bbdeb94873',60000,True),
 'train-labels-idx1-ubyte.gz':('d53e105ee54ea40749a09fcbcd1e9432',60000,False),
 't10k-images-idx3-ubyte.gz':('9fb629c4189551a2d022fa330f9573f3',10000,True),
 't10k-labels-idx1-ubyte.gz':('ec29112dd5afa0611ce80d1b7f02629c',10000,False)}


def load(name,info):
    md5,n,images=info
    path=DATA/name
    url='https://storage.googleapis.com/cvdf-datasets/mnist/'+name
    if not path.exists():
        subprocess.run(['curl','--fail','--location','--max-time','120','--max-filesize','11000000',url,'-o',str(path)],check=True)
    raw=path.read_bytes()
    if hashlib.md5(raw).hexdigest()!=md5:raise ValueError('Known MNIST checksum mismatch')
    size=16+n*784 if images else 8+n
    with gzip.open(path,'rb') as src:
        decoded=src.read(size+1)
    if len(decoded)!=size:raise ValueError('IDX length mismatch')
    expected=(2051,n,28,28) if images else (2049,n)
    if struct.unpack_from('>'+'I'*len(expected),decoded)!=expected:raise ValueError('IDX shape mismatch')
    array=np.frombuffer(decoded,dtype=np.uint8,offset=4*len(expected)).copy()
    array=array.reshape(n,784) if images else array
    if not images and np.max(array)>9:raise ValueError('Label out of range')
    return array,dict(url=url,compressed_sha256=hashlib.sha256(raw).hexdigest(),
        idx_sha256=hashlib.sha256(decoded).hexdigest(),known_md5=md5,bytes=len(raw),decoded_bytes=size)


def main():
    OUT.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(20261030)
    provenance={};arrays=[]
    for name,info in FILES.items():
        values,record=load(name,info);arrays.append(values);provenance[name]=record
    train,labels,test,testlabels=arrays
    x=torch.from_numpy(train.astype(np.float32)/255)
    y=torch.from_numpy(labels.astype(np.int64))
    model=torch.nn.Sequential(torch.nn.Linear(784,32),torch.nn.ReLU(),torch.nn.Linear(32,10))
    optimizer=torch.optim.Adam(model.parameters(),lr=.002)
    for epoch in range(8):
        order=torch.randperm(len(x))
        for ids in order.split(256):
            optimizer.zero_grad();loss=torch.nn.functional.cross_entropy(model(x[ids]),y[ids]);loss.backward();optimizer.step()
        print('epoch',epoch+1,'loss',float(loss.detach()),flush=True)
    weights={}
    for name,value in model.state_dict().items():weights[name]=value.detach().numpy()
    np.savez(OUT/'float.npz',**weights)
    with torch.no_grad():
        logits=model(torch.from_numpy(test.astype(np.float32)/255)).numpy()
    float_correct=int(np.sum(np.argmax(logits,axis=1)==testlabels))
    # Inputs are raw 0..255. Hidden outputs use a fixed Q8 scale.
    w1=weights['0.weight'];s1=np.max(np.abs(w1),axis=1)/127
    q1=np.rint(w1/s1[:,None]).astype(np.int8)
    b1=np.rint(weights['0.bias']*255/s1).astype(np.int32)
    m1=np.rint(s1/255*256*(1<<20)).astype(np.int32)
    w2=weights['2.weight'];s2=np.max(np.abs(w2),axis=1)/127
    q2=np.rint(w2/s2[:,None]).astype(np.int8)
    b2=np.rint(weights['2.bias']*256/s2).astype(np.int32)
    m2=np.rint(s2/256*4096*(1<<20)).astype(np.int32)
    np.savez(OUT/'integer.npz',w1=q1,b1=b1,m1=m1,w2=q2,b2=b2,m2=m2)
    hidden=np.clip(( (test.astype(np.int64)@q1.astype(np.int64).T+b1)*m1+(1<<19))>>20,0,32767)
    integer=((hidden@q2.astype(np.int64).T+b2)*m2+(1<<19))>>20
    pred=integer.argmax(axis=1)
    int_correct=int(np.sum(pred==testlabels))
    (OUT/'test-images.bin').write_bytes(test.tobytes())
    (OUT/'test-labels.bin').write_bytes(testlabels.tobytes())
    (OUT/'test-predictions.bin').write_bytes(pred.astype(np.uint8).tobytes())
    # Samples are fixed before model evaluation, with no correctness filtering.
    ids=[0,1,2]
    header='#define HIDDEN 32\n#define SAMPLES 3\n'
    for name,a in [('w1',q1),('b1',b1),('m1',m1),('w2',q2),('b2',b2),('m2',m2),('images',test[ids]),('labels',testlabels[ids])]:
        ctype='int8_t' if a.dtype==np.int8 else 'uint8_t' if a.dtype==np.uint8 else 'int32_t'
        header+=f'static const _Alignas(8) {ctype} {name}[{a.size}]={{'+','.join(map(str,a.reshape(-1)))+'};\n'
    (OUT/'model.h').write_text(header)
    report=dict(seed=20261030,epochs=8,architecture=[784,32,10],training_examples=60000,test_examples=10000,
        float_correct=float_correct,integer_correct=int_correct,float_accuracy=float_correct/10000,
        integer_accuracy=int_correct/10000,sample_indices=ids,sample_labels=testlabels[ids].tolist(),
        sample_predictions=pred[ids].tolist(),sources=provenance,
        model_header_sha256=hashlib.sha256(header.encode()).hexdigest(),
        integer_arithmetic='int8 weights, int16 activations, int32 dot, per-row Q20 rescale via int64, ReLU clamp 0..32767',
        integer_accuracy_scope='NumPy integer arithmetic; native C all-test agreement checked separately')
    (OUT/'training.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
