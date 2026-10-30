"""Exercise real save/restore entry points on CPU tensors, including bit corruption."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/tiny-llm-ndot'))
from checkpoint_integrity import CheckpointIntegrityError


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def machine():
    return SimpleNamespace(x=torch.tensor([[1.,-2.]]),tok=torch.tensor([42]),
        ex=SimpleNamespace(y=torch.zeros(2),steps=[dict(kind='head',S=torch.arange(6,dtype=torch.float32).reshape(2,3),hist=torch.tensor([1.,2.])),dict(kind='head',S=torch.ones(2,2),hist=torch.zeros(0))]))


def tensors(value,path=()):
    if isinstance(value,torch.Tensor): yield path,value
    elif isinstance(value,dict):
        for key,item in value.items(): yield from tensors(item,path+(key,))
    elif isinstance(value,list):
        for i,item in enumerate(value): yield from tensors(item,path+(i,))


def main():
    results={}
    modules=[('ndot','experiments/tiny-llm-ndot/worker.py'),('neural','experiments/tiny-llm-neural/worker.py'),('doom','experiments/doom-showcase/showcase_worker.py'),('legacy_doom','experiments/dmc-cuda/source/runtime.py')]
    for name,path in modules:
        module=load(name+'_worker',path)
        fr=machine();fused={'scratch_y':torch.zeros(3)}
        with tempfile.TemporaryDirectory() as directory:
            checkpoint=Path(directory)/'checkpoint.pt'
            if name=='doom':
                module.save_checkpoint(fr,fused,checkpoint,{'identity':{'fixture':'test'},'offset':12},np.zeros((2,2),dtype='<u2'),np.zeros((2,2),dtype=bool))
                restore=lambda:module.restore_checkpoint(fr,fused,checkpoint,{'fixture':'test'})
            elif name=='legacy_doom':
                module.save_checkpoint(fr,12,checkpoint)
                restore=lambda:module.restore_checkpoint(fr,checkpoint)
            else:
                module.save(fr,fused,checkpoint,{'fixture':'test'},12,1.25,2.5)
                restore=lambda:module.restore(fr,fused,checkpoint,{'fixture':'test'})
            original=torch.load(checkpoint,weights_only=True)
            for _,tensor in tensors({'x':fr.x,'tok':fr.tok,'steps':fr.ex.steps}): tensor.fill_(7)
            restore()
            assert torch.equal(fr.x,original['x']) and torch.equal(fr.tok,original['tok'])
            for step,saved in zip(fr.ex.steps,original['heads']):
                assert all(torch.equal(step[key],saved[key]) for key in ('S','hist'))
            cases=[]
            for tensor_path,tensor in tensors(original):
                if not tensor.numel(): continue
                mutated=copy.deepcopy(original);target=mutated
                for key in tensor_path: target=target[key]
                raw=target.reshape(-1).view(torch.uint8)
                raw[0]^=1
                cases.append(('/'.join(map(str,tensor_path)),mutated))
            old=copy.deepcopy(original);del old['integrity'];cases.append(('legacy_without_checksum',old))
            metadata=copy.deepcopy(original)
            if name=='doom': metadata['metadata']['offset']+=1
            else: metadata['offset']+=1
            cases.append(('offset',metadata))
            if 'compute_seconds' in original:
                for key in ('compute_seconds','retained_worker_seconds'):
                    mutated=copy.deepcopy(original);mutated[key]+=1;cases.append((key,mutated))
            rejections=[]
            for case,mutated in cases:
                before=[tensor.clone() for _,tensor in tensors({'x':fr.x,'tok':fr.tok,'steps':fr.ex.steps})]
                torch.save(mutated,checkpoint)
                try: restore()
                except CheckpointIntegrityError as error: rejections.append({'case':case,'error':str(error)})
                else: raise AssertionError((name,case,'accepted corrupt checkpoint'))
                after=[tensor for _,tensor in tensors({'x':fr.x,'tok':fr.tok,'steps':fr.ex.steps})]
                assert all(torch.equal(a,b) for a,b in zip(before,after)), 'Partial restore after checksum failure'
            torch.save(original,checkpoint);restore()
            results[name]={'unmodified_restores':True,'mutations_rejected_before_state_copy':rejections}
    out=ROOT/'experiments/checkpoint-integrity/evidence/cpu-restore.json'
    out.write_text(json.dumps(results,indent=2)+'\n')
    print({name:len(result['mutations_rejected_before_state_copy']) for name,result in results.items()})


if __name__=='__main__':main()
