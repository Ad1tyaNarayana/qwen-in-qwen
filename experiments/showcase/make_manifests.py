"""Publish measured gates with explicit nulls for incomplete or unavailable evidence."""
import hashlib
import json
from pathlib import Path
import shutil
import struct
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent


def read(path):return json.loads((ROOT/path).read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def drom_hash(path):
    raw=path.read_bytes();phoff=struct.unpack_from('<I',raw,28)[0];size,num=struct.unpack_from('<HH',raw,42)
    image=bytearray(4194304)
    for i in range(num):
        kind,off,addr,_,length,memsize,_,_=struct.unpack_from('<8I',raw,phoff+i*size)
        if kind==1 and 0x02000000<=addr<0x02400000:
            assert addr+memsize<=0x02400000
            image[addr-0x02000000:addr-0x02000000+length]=raw[off:off+length]
    return hashlib.sha256(image).hexdigest()
def save(name,value):
    correction=ROOT/'experiments/checkpoint-integrity/evidence/timing-correction.json'
    if name=='stories260k-38' and correction.exists():
        value['timing_correction_2026_10_31']=json.loads(correction.read_text())
    (HERE/'manifest'/f'{name}.json').write_text(json.dumps(value,indent=2)+'\n')


def main():
    weights=read('experiments/showcase/evidence/weights.json')['sha256']
    original=read('experiments/showcase/evidence/weights-original.json')['sha256']
    for name in ['sha256','life','tictactoe','mnist','mnist-scalar']:
        cache=ROOT/'.cache/showcase'/name
        ref=json.loads((cache/'reference.json').read_text());build=json.loads((cache/'build.json').read_text())
        assert sha(cache/'engine.elf')==ref['elf_sha256']==build['elf_sha256']
        assert sha(cache/'reference.tokens.u32')==ref['sha256']
        assert sha(cache/'drom.bin')==build['drom_sha256']
        dest=HERE/name/'evidence';dest.mkdir(parents=True,exist_ok=True)
        for f in ['reference.json','build.json']:
            shutil.copyfile(cache/f,dest/f)
        (dest/'output.txt').write_bytes((cache/'reference-output.bin').read_bytes())
        neuralpath=dest/'neural-validation.json'
        neural=json.loads(neuralpath.read_text()) if neuralpath.exists() else None
        if neural:
            assert neural['elf_sha256']==build['elf_sha256']==ref['elf_sha256']
            assert neural['drom_sha256']==build['drom_sha256']
            assert neural['network_weight_sha256']==weights
            assert neural['actual_trace_sha256']==ref['sha256']
            assert neural['trace_tokens']==ref['tokens'] and neural['mismatches']==0
        result=dict(demo=name,status='verified' if neural else 'reference-verified-neural-pending',
            elf_sha256=build['elf_sha256'],drom_image_sha256=build['drom_sha256'],
            network_weight_sha256=weights,network='unified W=4 full-map',
            instruction_count=ref['instructions'],trace_tokens=ref['tokens'],mismatches=neural['mismatches'] if neural else None,
            generation_seconds=neural['generation_seconds'] if neural else None,
            tokens_per_second=neural['tokens_per_second'] if neural else None,
            outputs=neural['output'] if neural else ref['output'],output_source='actual self-fed GPU BYTE records' if neural else 'native/software reference only',
            stock_hf_prefix_tokens_checked=neural['stock_hf_prefix_tokens_checked'] if neural else 0,
            evidence_paths=[str(p.relative_to(ROOT)) for p in sorted(dest.glob('*'))],
            network_weight_evidence='experiments/showcase/evidence/weights.json',
            timing_scope='one complete autonomous execution; generation only; not repeated benchmark' if neural else None)
        save(name,result)
    story=read('experiments/tiny-llm-ndot-fused/evidence/story-validation.json')
    fixture=ROOT/'.cache/tiny-llm/ndot-story38-20261030'
    reference=json.loads((fixture/'reference.json').read_text())
    assert sha(fixture/'engine.elf')==reference['elf_sha256']
    save('stories260k-38',dict(demo='stories260K ndot 38-position story',status='historical-verified',
        elf_sha256=reference['elf_sha256'],drom_image_sha256=drom_hash(fixture/'engine.elf'),
        network_weight_sha256=weights,network='unified W=4 full-map',
        weight_hash_scope='retrospective stock emission from hash-bound constructor, not a hash of historical GPU tensor dumps',
        instruction_count=reference['instructions'],trace_tokens=story['tokens'],mismatches=story['mismatches'],
        generation_seconds=story['compute_seconds'],tokens_per_second=story['tokens_per_second'],
        outputs=story['story_text'],output_source=story['output_source'],stock_hf_prefix_tokens_checked=0,
        stock_note='32,768 checked tokens exist for the related five-position ELF, not this 38-position ELF',
        evidence_paths=['experiments/tiny-llm-ndot-fused/evidence/story-validation.json','experiments/tiny-llm-ndot-fused/evidence/story-fixture.json','experiments/showcase/evidence/weights.json']))
    phase=read('experiments/tiny-llm-neural/evidence/validation.json')
    ref=read('experiments/tiny-llm-neural/evidence/reference.json')
    elf=ROOT/'.cache/tiny-llm/neural-stories/engine.elf';assert sha(elf)==ref['elf_sha256']
    hf=read('experiments/tiny-llm-neural/evidence/hf.json')
    save('phase1',dict(demo='Phase 1 stories260K five positions',status='historical-verified',elf_sha256=ref['elf_sha256'],
        drom_image_sha256=drom_hash(elf),network_weight_sha256=original,network='original full-map circuit; different weights from unified W=4',
        weight_hash_scope='retrospective original stock emission; historical GPU tensor dump not hashed',
        instruction_count=ref['instructions'],trace_tokens=phase['tokens_checked'],mismatches=phase['mismatches'],
        generation_seconds=phase['compute_minutes']*60,tokens_per_second=phase['tokens_per_second'],outputs=phase['generated_text'],
        output_source=phase['output_source'],stock_hf_prefix_tokens_checked=hf['generated_tokens'],
        stock_note='Historical deadline-bounded exact prefix; original requested 100,000-token target was not completed',
        evidence_paths=['experiments/tiny-llm-neural/evidence/validation.json','experiments/tiny-llm-neural/evidence/reference.json','experiments/tiny-llm-neural/evidence/hf.json','experiments/showcase/evidence/weights-original.json']))
    doom=read('experiments/dmc-cuda/evidence/full-report.json')
    validation=read('experiments/dmc-cuda/evidence/independent-validation.json')
    elf=ROOT/'.cache/showcase/historical-doom/doom.elf'
    same=sha(elf)==doom['elf_sha256']
    save('doom-first-frame',dict(demo='Doom first frame',status='historical-verified',elf_sha256=doom['elf_sha256'],
        drom_image_sha256=drom_hash(elf) if same else None,
        network_weight_sha256=original,network='original full-map circuit; different weights from unified W=4',
        weight_hash_scope='retrospective original stock emission; historical GPU tensor dump not hashed',
        instruction_count=3885544,trace_tokens=validation['generated_tokens_compared'],mismatches=validation['token_mismatches'],
        generation_seconds=doom['compute_seconds_total'],tokens_per_second=doom['generated_tokens']/doom['compute_seconds_total'],
        outputs='320x240 E1M1 first frame, 25 strips, RGB565 SHA-256 '+validation['actual_rgb565_sha256'],
        output_source='actual GPU recurrent RAM strips and palette',stock_hf_prefix_tokens_checked=0,
        stock_note='No matching-ELF historical stock prefix imported',
        evidence_paths=['experiments/dmc-cuda/evidence/full-report.json','experiments/dmc-cuda/evidence/independent-validation.json','experiments/showcase/evidence/weights-original.json']))
    shutil.copyfile(ROOT/'.cache/showcase/mnist-model/training.json',HERE/'mnist/evidence/training.json')
    print('Wrote eight manifests.')


if __name__=='__main__':main()
