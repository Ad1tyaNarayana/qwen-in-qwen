"""Fail closed before GPU execution unless Parts A through C have passed."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from verify import code, records
from pack import ROOT, HERE


def gate(legacy, program, quality):
    parity = json.loads((legacy/'legacy-parity.json').read_text())
    assert parity['status'] == 'pass' and parity['asan_ubsan'] and parity['total_positions'] == 579
    sources = {}
    reference_reports = []
    expected_code = code(program/'engine.elf')
    for name,count in [('stories-0',64),('custom-0',129)]:
        directory = legacy/name
        result = json.loads((directory/'reference.json').read_text())
        assert result['elf_sha256'] == hashlib.sha256((directory/'engine.elf').read_bytes()).hexdigest()
        assert result['status'] == 'pass' and result['complete'] and result['native_output_exact']
        actual = (directory/'reference-output.bin').read_bytes()
        assert actual == (directory/'old-output.bin').read_bytes()
        assert len(records(actual)) == count
        assert result['steps'] == count and result['stack_bytes'] <= 1024
        assert code(directory/'engine.elf') == expected_code
        reference_reports.append(dict(model=name,positions=count,tokens=result['tokens'],
                                      output_sha256=result['output_sha256']))
        sources[str(directory/'reference.json')] = hashlib.sha256((directory/'reference.json').read_bytes()).hexdigest()
    trained = json.loads(quality.read_text())
    checkpoint = Path(trained['checkpoint'])
    training = json.loads((checkpoint.parent/'done.json').read_text())
    environment = json.loads((checkpoint.parent/'environment.json').read_text())
    assert environment['model_class'] == 'Qwen3_5ForCausalLM'
    assert environment['tokenizer_sha256'] == hashlib.sha256((ROOT/'.cache/tiny-llm/custom/tok1024.model').read_bytes()).hexdigest()
    assert training['status'] == 'complete' and 0 < training['wall_seconds'] <= 3600
    for name,info in training['hf_files'].items():
        assert trained['checkpoint_files'][name] == info['sha256']
        assert hashlib.sha256((checkpoint/name).read_bytes()).hexdigest() == info['sha256']
    assert trained['free_running_agreement'] >= 0.9, 'Quantized greedy quality needs investigation'
    assert all(c['kv_clips'] == 0 and not c['numeric_clips'] for c in trained['cases'])
    ref = json.loads((program/'reference.json').read_text())
    meta = json.loads((program/'build.json').read_text())
    assert ref['status'] == 'pass' and ref['native_output_exact'] and ref['steps'] >= 6
    assert ref['steps']-len(meta['prompt_tokens'])+1 >= 5
    assert ref['stack_bytes'] <= 1024 and ref['stack_clearance'] >= 1024
    assert ref['ndot4_instructions'] > 0
    assert not any(r[4] for r in records((program/'reference-output.bin').read_bytes()))
    assert not re.search(rb'^!\d+\n',(program/'reference-output.bin').read_bytes(),re.M)
    assert meta['runtime_sha256'] == hashlib.sha256((HERE/'runtime.c').read_bytes()).hexdigest()
    assert meta['runtime_binding']['native_sha256'] == trained['runtime_sha256']
    assert meta['quantization'] == trained['quantization']
    assert meta['calibration_sha256'] == trained['calibration_sha256']
    if trained['calibration_file']:
        assert hashlib.sha256(Path(trained['calibration_file']).read_bytes()).hexdigest() == trained['calibration_sha256']
    assert meta['linked_margin'] >= 1024 and meta['rom_bytes'] <= 3*2**20
    assert hashlib.sha256((program/'engine.elf').read_bytes()).hexdigest() == ref['elf_sha256']
    assert hashlib.sha256((program/'reference.tokens.u32').read_bytes()).hexdigest() == ref['sha256']
    sources.update({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                    [legacy/'legacy-parity.json',quality,program/'reference.json',program/'engine.elf']})
    result = dict(status='pass',parts=['A','B','C'],legacy_reference=reference_reports,
                  native_legacy_positions=579,trained_free_running_agreement=trained['free_running_agreement'],
                  qwen_reference_positions=ref['steps'],qwen_trace_tokens=ref['tokens'],
                  executable_sha256=[hashlib.sha256(c[2]).hexdigest() for c in expected_code],
                  same_executable_for_llama_and_qwen=True,sources=sources,
                  projected_minutes_per_position_at_3735=ref['tokens']/3735/60/ref['steps'])
    (HERE/'evidence/mac-gate.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--legacy',type=Path,required=True)
    p.add_argument('--program',type=Path,required=True)
    p.add_argument('--quality',type=Path,required=True)
    a = p.parse_args()
    print(json.dumps(gate(a.legacy,a.program,a.quality),indent=2))
