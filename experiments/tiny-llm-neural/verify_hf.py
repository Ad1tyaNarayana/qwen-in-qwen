"""Verify and export a bounded stock-HF report without publishing raw trace tokens."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--report',type=Path,required=True)
p.add_argument('--reference',type=Path,default=ROOT / '.cache/tiny-llm/neural-stories')
a = p.parse_args()
h = json.loads(a.report.read_text())
r = json.loads((a.reference / 'reference.json').read_text())
actual = np.asarray(h.pop('tokens'),dtype='<u4')
expected = np.memmap(a.reference / 'reference.tokens.u32',mode='r',dtype='<u4')
assert h['generated_tokens'] == len(actual) > 0
assert np.array_equal(actual,expected[:len(actual)])
assert h['mismatches'] == 0 and h['firmware_sha256'] == r['elf_sha256']
assert h['reference_trace_sha256'] == r['sha256']
assert h['cuda_peak_allocated_bytes'] < h['allocator_cap_bytes'] == 14*2**30
assert h['minimum_free_vram_bytes'] >= 2**30
assert h['layer_offload_used'] is False
assert h['main_run_complete_before_hf'] is True
assert h['ram_bytes'] == 65536 and h['drom_bytes'] == 4194304
assert h['script_sha256'] == hashlib.sha256(Path(__file__).with_name('hf_prefix.py').read_bytes()).hexdigest()
h['independent_mac_token_comparison'] = True
h['actual_prefix_sha256'] = hashlib.sha256(actual.tobytes()).hexdigest()
assert len(actual) < r['positions'][0]['end_token']
h['scope'] = 'Self-fed program prefix from ELF boot, including the beginning of guest transformer inference. No complete guest LLM prediction is reached.'
h['first_guest_prediction_end_trace_token'] = r['positions'][0]['end_token']
emitter = json.loads((Path(__file__).parent / 'evidence/emitter.json').read_text())
assert h['streamed_emitter_sha256'] == emitter['source_sha256']
(Path(__file__).parent / 'evidence/hf.json').write_text(json.dumps(h,indent=2)+'\n')
print(json.dumps(h,indent=2))
