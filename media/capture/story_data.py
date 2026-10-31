"""Write the video's story tokens from the retained story evidence."""
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
SOURCE = 'experiments/tiny-llm-ndot-fused/evidence/story-validation-20261031.json'

evidence = json.loads((ROOT / SOURCE).read_text())
tokens = re.findall(r'^>(.*)$', evidence['output'], flags=re.M)
assert ''.join(tokens) == evidence['story_text']
assert len(tokens) == evidence['positions'] == 38 and evidence['mismatches'] == 0
story = dict(source=SOURCE, tokens=tokens, text=evidence['story_text'], positions=evidence['positions'],
             traceTokens=evidence['tokens'], mismatches=evidence['mismatches'],
             computeSeconds=evidence['compute_seconds'], minutesPerPosition=evidence['min_per_position'])
(ROOT / 'media/video/src/story.json').write_text(json.dumps(story, indent=1) + '\n')
