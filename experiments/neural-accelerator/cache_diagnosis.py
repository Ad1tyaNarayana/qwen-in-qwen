"""Observe stock generated inputs in independent fresh-boot replay caches."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import traceback

from prepare import ROOT
from stock_m3 import save, workload


def differences(circuit, left, right):
    import torch
    from chipformer.rv.hf import head_slots
    slots = {slot: circuit.nodes[index] for index, slot in head_slots(circuit).items()}
    ratio = circuit.target.value_heads // circuit.target.key_heads
    records = []
    for layer in range(circuit.n_layers()):
        for field in ('recurrent_states', 'conv_states'):
            a = getattr(left.layers[layer], field)[0]
            b = getattr(right.layers[layer], field)[0]
            bits_a = a.contiguous().view(torch.int32 if a.dtype == torch.float32 else torch.int16)
            bits_b = b.contiguous().view(bits_a.dtype)
            coords = (bits_a != bits_b).nonzero().cpu().tolist()
            for coord in coords:
                index = tuple(coord)
                x, y = a[index].item(), b[index].item()
                width = a.element_size() * 8
                record = dict(layer=layer, field=field, coordinate=coord,
                              cached=x, chunked=y,
                              cached_bits=f'0x{bits_a[index].item() & ((1 << width)-1):0{width//4}x}',
                              chunked_bits=f'0x{bits_b[index].item() & ((1 << width)-1):0{width//4}x}',
                              signed_zero=x == y == 0)
                if field == 'recurrent_states':
                    _, head, row, column = coord
                    node = slots.get((layer, head // ratio))
                    record.update(value_head=head, key_head=head // ratio, key_row=row, value_column=column,
                                  node=node.name if node else None)
                    if node:
                        vh = node.vheads()[head % ratio] if head % ratio < len(node.vheads()) else None
                        record.update(query_rows=sorted(node.q), key_rows=sorted(node.k),
                                      output_columns=[col for col, _, _ in vh.out] if vh else [],
                                      value_columns=sorted(vh.v) if vh else [])
                records.append(record)
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream', type=Path, default=ROOT / '.upstream/qwenputer')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--kind', choices=['accelerator', 'original'], default='accelerator')
    p.add_argument('--tokens', type=int, default=1804)
    p.add_argument('--chunks', default='1,2,4,8,16,64')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--integer-counters', action='store_true')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    path = a.out / 'report.json'
    report = dict(status='running', kind=a.kind, started_utc=datetime.now(timezone.utc).isoformat(),
                  generated_tokens=0, comparisons=[], boot_injection_only=True,
                  replay_inputs='only inputs consumed by authoritative stock generate',
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (a.out / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    save(path, report)
    try:
        import torch
        torch.set_num_threads(4)
        prep_spec = importlib.util.spec_from_file_location('diagnosis_prepare', ROOT / 'experiments/memory-map/prepare.py')
        prep = importlib.util.module_from_spec(prep_spec)
        prep_spec.loader.exec_module(prep)
        report['prepare_manifest'] = prep.prepare(a.upstream, a.out / 'scratch', 16, 8)
        sys.path.insert(0, str(a.out / 'scratch'))
        from chipformer.rv import spec, core, hf, ref
        from chipformer.q35.fast import exact_matmuls
        from stock import build_model, boot_cache
        from constructor import build
        import reference
        from transformers import StoppingCriteria, StoppingCriteriaList
        from transformers.generation.streamers import BaseStreamer
        exact_matmuls()
        if a.device == 'cuda':
            torch.cuda.set_per_process_memory_fraction(12 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
        if a.kind == 'accelerator':
            circuit = build(4)
        else:
            circuit = core.build()
            circuit.accelerator_module = core
        program, source = workload(0, 7)
        if a.kind == 'original':
            from chipformer.rv.asm import assemble, enc_r
            scalar = '\n'.join(f'lb t3, {lane}(s1)\nlh t4, {2*lane}(s2)\n'
                               'mul t3, t3, t4\nadd s0, s0, t3' for lane in range(4))
            source = source.replace(f'.word {enc_r(32, 18, 9, 0, 8, 11)}', scalar)
            program = assemble(source)
        initial = (reference.Machine if a.kind == 'accelerator' else ref.Machine).from_program(program)
        initial.maximum = 4
        machine = copy.deepcopy(initial)
        wanted, commits = [], {}
        while not machine.halted:
            record = machine.record()
            wanted.extend(record)
            if record[-1] != spec.tok('HALT'):
                commits[len(wanted)] = (tuple(machine.x), tuple(machine.ram), machine.anchor, machine.n, machine.pc)
        model, metadata = build_model(circuit, a.device)
        report.update(model=metadata, geometry=circuit.stats(), torch=torch.__version__,
                      transformers=__import__('transformers').__version__, program_source=source)
        save(path, report)
        chunks = [int(x) for x in a.chunks.split(',')]
        replay, pending, predictions = {}, {}, {}
        consumed = 0
        nested = False
        first = set()
        early_states, actual_inputs = {}, []
        checked_commits = 0
        from stock_m3 import state_observer
        observe = state_observer(circuit)

        class Checker(BaseStreamer):
            def __init__(self):
                self.tokens, self.initial = [], True
            def put(self, value):
                if self.initial:
                    self.initial = False
                    return
                for token in value.cpu().tolist():
                    index = len(self.tokens)
                    assert token == wanted[index], ('generated token', index, token, wanted[index])
                    self.tokens.append(token)
            def end(self):
                pass

        class Guard(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                if a.device == 'cuda':
                    assert torch.cuda.mem_get_info()[0] >= 512 * 2**20, 'free VRAM guard'
                return False

        with torch.inference_mode():
            cache = boot_cache(model, circuit, initial)
            for chunk in chunks:
                replay[chunk] = boot_cache(model, circuit, initial)
                pending[chunk], predictions[chunk] = [], []
            checker = Checker()

            def compare(chunk):
                nonlocal nested
                nested = True
                try:
                    out = model(input_ids=torch.tensor([pending[chunk]], device=a.device),
                                past_key_values=replay[chunk], use_cache=True)
                finally:
                    nested = False
                predictions[chunk].extend(out.logits.argmax(-1)[0].tolist())
                assert predictions[chunk] == wanted[:consumed], ('replay token', chunk, consumed)
                delta = differences(circuit, cache, replay[chunk])
                result = dict(chunk_size=chunk, consumed_inputs=consumed, differences=delta,
                              numerical_differences=sum(not d['signed_zero'] for d in delta))
                report['comparisons'].append(result)
                report['generated_tokens'] = len(checker.tokens)
                if delta and chunk not in first:
                    first.add(chunk)
                    torch.save(dict(cached=cache, chunked=replay[chunk], consumed=consumed),
                               a.out / f'first-difference-chunk{chunk}.pt')
                    print(json.dumps(result), flush=True)
                pending[chunk] = []
                save(path, report)

            def hook(module, args, kwargs, out):
                nonlocal consumed, checked_commits
                if nested:
                    return
                ids = kwargs['input_ids'][0].tolist()
                assert len(ids) == 1
                consumed += 1
                actual_inputs.extend(ids)
                if consumed <= 64:
                    early_states[consumed] = cache.layers[0].recurrent_states[0].cpu().clone()
                if consumed - 1 in commits:
                    observe(cache, commits[consumed - 1], consumed - 1)
                    checked_commits += 1
                for chunk in chunks:
                    pending[chunk].extend(ids)
                    if len(pending[chunk]) == chunk:
                        compare(chunk)
                if consumed % 64 == 0:
                    print(json.dumps(dict(progress_inputs=consumed)), flush=True)

            handle = model.register_forward_hook(hook, with_kwargs=True)
            try:
                model.generate(input_ids=torch.tensor([[spec.VOCAB, spec.tok('GO')]], device=a.device),
                    past_key_values=cache, max_new_tokens=min(a.tokens, len(wanted)), do_sample=False,
                    eos_token_id=spec.tok('HALT'), pad_token_id=spec.VOCAB,
                    streamer=checker, stopping_criteria=StoppingCriteriaList([Guard()]))
                for chunk in chunks:
                    if pending[chunk]:
                        compare(chunk)
            finally:
                handle.remove()
            report['prefix_refinement'] = []
            if 16 in chunks and len(actual_inputs) >= 32:
                for length in [1, 2, 16, 24, 20, 18, 17, 19, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32]:
                    prefix = boot_cache(model, circuit, initial)
                    for offset in range(0, length, 16):
                        out = model(input_ids=torch.tensor([actual_inputs[offset:min(offset + 16, length)]], device=a.device),
                                    past_key_values=prefix, use_cache=True)
                    x = early_states[length]
                    y = prefix.layers[0].recurrent_states[0].cpu()
                    coords = (x.view(torch.int32) != y.view(torch.int32)).nonzero().tolist()
                    record = dict(consumed_inputs=length, last_input=actual_inputs[length-1],
                                  last_input_family=spec.family(actual_inputs[length-1]),
                                  layer=0, coordinates=coords,
                                  cached=[x[tuple(c)].item() for c in coords],
                                  chunked=[y[tuple(c)].item() for c in coords],
                                  cached_bits=[f'0x{x.view(torch.int32)[tuple(c)].item() & 0xffffffff:08x}' for c in coords],
                                  chunked_bits=[f'0x{y.view(torch.int32)[tuple(c)].item() & 0xffffffff:08x}' for c in coords])
                    report['prefix_refinement'].append(record)
                    save(path, report)
                    del prefix, out
            report.update(status='diagnosed', generated_tokens=len(checker.tokens), consumed_inputs=consumed,
                          checked_commits=checked_commits, acceptance='FAIL readable state differs' if first else 'exact',
                          generated_token_sha256=hashlib.sha256(torch.tensor(checker.tokens, dtype=torch.int32).numpy().tobytes()).hexdigest())
    except Exception as error:
        report.update(status='error', error=repr(error), traceback=traceback.format_exc())
        print(report['traceback'], flush=True)
    finally:
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save(path, report)
    raise SystemExit(report['status'] == 'error')


if __name__ == '__main__':
    main()
