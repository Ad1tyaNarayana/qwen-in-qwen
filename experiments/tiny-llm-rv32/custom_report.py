"""Collect verified custom-model fit, quality, and deterministic work counts."""
import argparse
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--build',type=Path,default=ROOT / '.cache/tiny-llm/custom-selected')
p.add_argument('--neural-report',type=Path,default=ROOT / 'experiments/tiny-llm-neural/evidence/gpu.json')
a = p.parse_args()
b = json.loads((a.build / 'build.json').read_text())
r = json.loads((a.build / 'reference.json').read_text())
assert r['complete'] and r['bit_exact'] and r['stack_bytes'] <= b['pack']['stack_reserved']
assert r['stack_clearance'] > 0
static = b['elf']['symbols']['__scratch_end']-0x01000100
ram = dict(static_bytes=static,bottom_reserved=256,stack_reserved=b['pack']['stack_reserved'],
           total_reserved=256+static+b['pack']['stack_reserved'],
           margin_bytes=65536-256-static-b['pack']['stack_reserved'],observed_stack_bytes=r['stack_bytes'])
assert ram['margin_bytes'] >= 1024
q = json.loads((ROOT / 'experiments/tiny-llm-rv32/evidence/custom-quality.json').read_text())
selected = next(s for s in q['format_summary'] if s['kv_bits'] == 16)
assert selected['context'] == b['pack']['context'] == 129
assert selected['free_running_matches'] == selected['generated_predictions']
rate = None
if a.neural_report.exists():
    n = json.loads(a.neural_report.read_text())
    assert n['complete'] and n['mismatches'] == 0
    rate = n['tokens_per_second']
positions = []
macs = 6*(2*192*192+2*16*192+3*512*192)+1024*192
for pos in r['positions']:
    assert pos['dot_body_instructions'] == 7*macs
    item = dict(pos)
    item['projected_minutes_at_4380'] = pos['trace_tokens']/4380/60
    if rate:
        item['projected_minutes_at_measured_stories_rate'] = pos['trace_tokens']/rate/60
    positions.append(item)
decode = [p for p in positions if p['phase'] == 'decode']
report = dict(kind='Custom model native quality and software RV32 execution. No custom-model neural timing.',
              fit=ram,kv_bits=16,kv_fractional_bits=10,context=129,pack=b['pack'],
              elf_sha256=b['elf']['sha256'],quality=selected,reference={k:v for k,v in r.items() if k != 'positions'},
              positions=positions,mean_decode_instructions=sum(p['instructions'] for p in decode)/len(decode),
              mean_decode_trace_tokens=sum(p['trace_tokens'] for p in decode)/len(decode),
              macs_per_position=macs,
              measured_stories_trace_rate=rate,cost_scope='Instruction and trace-token counts are measured on the reference CPU. GPU minutes for the custom model are projections only.')
(ROOT / 'experiments/tiny-llm-rv32/evidence/custom.json').write_text(json.dumps(report,indent=2)+'\n')
rows = []
for index in (0,4,32,64,128):
    pos = positions[index]
    measured_projection = f"{pos['trace_tokens']/rate/60:.2f}" if rate else 'Not available'
    rows.append(f"| {index} | {pos['instructions']:,} | {pos['trace_tokens']:,} | {pos['projected_minutes_at_4380']:.2f} | {measured_projection} |")
result_text = f'''The full 129-position RV32 run reaches clean HALT and matches native output
byte for byte, including all argmax IDs, logits hashes, decoded text, and clipping
counts. It retires {r['instructions']:,} instructions and emits {r['trace_tokens']:,}
trace tokens. Observed stack use is {r['stack_bytes']} bytes. The 1,024-byte stack
reservation is {1024/r['stack_bytes']:.1f} times that observation. Lowest SP remains
{r['stack_clearance']:,} bytes above static storage. The final margin-gated build
has identical loadable bytes to this full-reference ELF in
[custom-reproduction.json](evidence/custom-reproduction.json).

There are {macs:,} scalar matrix MACs per position. The profiler verifies the
actual seven-instruction dot loop retires exactly that many MACs at every
position. Decode averages {report['mean_decode_instructions']:,.0f} instructions
and {report['mean_decode_trace_tokens']:,.0f} trace tokens per prediction.

| Position | Instructions | Trace tokens | Projected minutes at 4,380/s | Projected minutes at measured stories260K rate |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

The instruction and trace counts are measured. Both GPU-minute columns are
projections for the custom model. No custom-model neural run is claimed.
See [custom.json](evidence/custom.json) for every position and the timing scope.
'''
readme = Path(__file__).resolve().parent / 'README.md'
content = readme.read_text()
content,count = re.subn(r'<!-- custom-results:start -->.*?<!-- custom-results:end -->',
    '<!-- custom-results:start -->\n'+result_text+'<!-- custom-results:end -->',content,flags=re.S)
assert count == 1
readme.write_text(content)
print(json.dumps({k:v for k,v in report.items() if k not in ['pack','reference','positions']},indent=2))
