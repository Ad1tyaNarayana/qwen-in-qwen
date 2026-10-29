import hashlib
import json
from pathlib import Path
import struct
import subprocess

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / '.cache/tiny-llm/custom'
BASELINE = ROOT / '.cache/tiny-llm/downloads/stories260K'
SOURCE = ROOT / '.cache/tiny-llm/downloads/llama2.c'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip().startswith('350e04fe')
    dim, hidden, layers, heads, kv_heads, vocab, context = struct.unpack('<7i', (ARTIFACTS / 'model.bin').read_bytes()[:28])
    assert vocab > 0
    head_dim = dim // heads
    kv_dim = kv_heads * head_dim
    matrix_weights = vocab * dim + layers * (2 * dim * dim + 2 * dim * kv_dim + 3 * dim * hidden)
    norm_weights = (2 * layers + 1) * dim
    q8_bytes = matrix_weights + matrix_weights // 64 * 4 + norm_weights * 4 + 256
    kv_bytes = layers * 2 * kv_dim * 128
    assert dim <= 192 and kv_bytes <= 24576 and q8_bytes <= 3 * 1024**2
    assert (ARTIFACTS / 'model-q8.bin').stat().st_size == q8_bytes
    assert (ARTIFACTS / 'model.bin').stat().st_size == 28 + (matrix_weights + norm_weights + context * head_dim) * 4
    qheader = (ARTIFACTS / 'model-q8.bin').read_bytes()[:41]
    assert struct.unpack('<II', qheader[:8]) == (0x616b3432, 2)
    assert qheader[36] == 1 and struct.unpack('<i', qheader[37:41])[0] == 64
    exported = json.loads((ARTIFACTS / 'export-evidence.json').read_text())
    for name, expected in exported['artifacts'].items():
        path = ARTIFACTS / name
        assert path.stat().st_size == expected['bytes'] and sha256(path) == expected['sha256'], name
    assert sha256(SOURCE / 'run.c') == sha256(ARTIFACTS / 'source/run.c')
    subprocess.run(['clang', '-O3', '-o', str(ARTIFACTS / 'run'), str(SOURCE / 'run.c'), '-lm'], check=True)
    prompts = ['Once upon a time, there was a little', 'Lily found a small bird in the garden.', 'Tom was sad because he lost his toy.']
    samples = {}
    for label, model, tokenizer in [('custom', ARTIFACTS / 'model.bin', ARTIFACTS / 'tok1024.bin'), ('stories260K', BASELINE / 'stories260K.bin', BASELINE / 'tok512.bin')]:
        samples[label] = []
        for prompt in prompts:
            result = subprocess.run([str(ARTIFACTS / 'run'), str(model), '-z', str(tokenizer), '-t', '0', '-s', '1', '-n', '128', '-i', prompt], capture_output=True, text=True, check=True)
            assert result.stdout.startswith(prompt)
            assert len(result.stdout) > len(prompt) + 40
            samples[label].append({'prompt': prompt, 'text': result.stdout.strip()})
    evidence = {'architecture': {'dim': dim, 'hidden_dim': hidden, 'layers': layers, 'heads': heads, 'kv_heads': kv_heads, 'head_dim': head_dim, 'kv_dim': kv_dim, 'vocab': vocab, 'training_context': context, 'shared_classifier': True}, 'parameter_count': matrix_weights + norm_weights, 'matrix_weights': matrix_weights, 'norm_weights': norm_weights, 'group_size': 64, 'q8_scale_bytes': matrix_weights // 64 * 4, 'q8_file_bytes': q8_bytes, 'int8_kv_bytes_at_128': kv_bytes, 'export': exported, 'loss': json.loads((ARTIFACTS / 'evaluation.json').read_text()), 'final_training_loss': json.loads((ARTIFACTS / 'loss.jsonl').read_text().splitlines()[-1]), 'source_commit': subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip(), 'run_c_sha256': sha256(SOURCE / 'run.c'), 'baseline': {p.name: {'bytes': p.stat().st_size, 'sha256': sha256(p)} for p in BASELINE.iterdir() if p.is_file()}, 'samples': samples}
    assert sha256(ARTIFACTS / 'heldout_stories.json') == evidence['loss']['heldout_sha256']
    stories = json.loads((ARTIFACTS / 'heldout_stories.json').read_text())
    corpus = ARTIFACTS / 'heldout-texts.bin'
    with corpus.open('wb') as output:
        output.write(struct.pack('<I', len(stories)))
        for story in stories:
            encoded = story.encode('utf-8')
            output.write(struct.pack('<I', len(encoded)))
            output.write(encoded)
    native = ARTIFACTS / 'evaluate-native'
    subprocess.run(['clang', '-O3', '-I', str(SOURCE), str(Path(__file__).parent / 'evaluate_native.c'), '-lm', '-o', str(native)], check=True)
    evidence['native_loss'] = {}
    for label, model, tokenizer in [('custom', ARTIFACTS / 'model.bin', ARTIFACTS / 'tok1024.bin'), ('stories260K', BASELINE / 'stories260K.bin', BASELINE / 'tok512.bin')]:
        result = subprocess.run([str(native), str(model), str(tokenizer), str(corpus)], check=True, capture_output=True, text=True)
        measured = json.loads(result.stdout)
        reference = evidence['loss'][label]
        assert measured['tokens'] == reference['tokens'] and measured['utf8_bytes'] == reference['utf8_bytes']
        assert abs(measured['nats_per_token'] - reference['nats_per_token']) < 0.02
        evidence['native_loss'][label] = measured
    destination = Path(__file__).parent
    (destination / 'evidence.json').write_text(json.dumps(evidence, indent=2))
    (ARTIFACTS / 'samples.json').write_text(json.dumps(samples, indent=2))
    print(json.dumps({'parameter_count': matrix_weights + norm_weights, 'q8_bytes': q8_bytes, 'kv_bytes': kv_bytes, 'samples': samples}, indent=2))


if __name__ == '__main__':
    main()
