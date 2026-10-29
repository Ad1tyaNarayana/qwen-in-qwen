"""Autonomous stock-HF gate with independent commit observers and exact cache replay."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import time
import traceback

from prepare import ROOT


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    for attempt in range(101):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 100:
                raise
            time.sleep(0.02)


def workload(variant, data):
    from chipformer.rv.asm import assemble, enc_r
    word = enc_r(32, 18, 9, 0, 8, 11)
    partial = 'sh s0, 32(t0)\nlh t1, 32(t0)' if variant else 'sw s0, 32(t0)\nlw t1, 32(t0)'
    source = f'''_start:
la s1, weights
la s2, values
li s3, 128
loop:
.word {word}
andi t0, s0, 7
slli t0, t0, 2
add t0, t0, s2
{partial}
andi t2, t1, 1
beqz t2, even
addi s0, s0, 1
even:
xor s0, s0, t1
addi s3, s3, -1
bnez s3, loop
li a7, 93
li a0, 0
ecall
.data
.align 4
values: .half -32768, 32767, -17, 29
.space 128
.rodata
.align 3
weights: .byte {data}, -128, 127, -3
'''
    return assemble(source), source


def expected_execution(initial):
    from chipformer.rv import spec
    import reference
    from oracle import execute, serialize
    machine = copy.deepcopy(initial)
    tokens, commits, dots = [], {}, 0
    while not machine.halted:
        word = machine.rom[(machine.pc - spec.ROM_BASE) // 4]
        ins = reference.decode(word, 4)
        if machine.n != spec.N_MAX and ins.op == 'NDOT4':
            outcome = execute(word, machine.x, struct.pack('<' + 'I' * len(machine.ram), *machine.ram), machine.drom, 4)
            independent = serialize(outcome, word, machine.pc)
            record = machine.record()
            assert record == independent and tuple(machine.x) == outcome.regs and outcome.fault is None
            dots += 1
        else:
            record = machine.record()
        tokens.extend(record)
        if record[-1] != spec.tok('HALT'):
            commits[len(tokens)] = (tuple(machine.x), tuple(machine.ram), machine.anchor, machine.n, machine.pc)
    assert machine.steps >= 1024 and dots == 128 and machine.fault is None
    return tokens, commits, machine.steps, dots


def state_observer(circuit):
    import numpy as np
    from chipformer.q35.ir import A_BIT
    from chipformer.rv import hf, spec
    slots = hf.head_slots(circuit)
    by_identity = {id(circuit.nodes[i]): slot for i, slot in slots.items()}
    ratio = circuit.target.value_heads // circuit.target.key_heads

    def table(cache, name, value_head=0):
        layer, head = by_identity[id(circuit.tables[name])]
        return cache.layers[layer].recurrent_states[0][0, head * ratio + value_head].float().cpu().numpy()

    def check(cache, expected, offset):
        registers, ram, anchor, count, pc = expected
        for port in ('rs1', 'rs2', 'rd'):
            rows = table(cache, 'regs.' + port)
            for register in range(1, 32):
                bits = np.array([sum(rows[register, col] for col in circuit.reg_cols[bit]) / A_BIT for bit in range(32)])
                wanted = np.array([(registers[register] >> bit) & 1 for bit in range(32)])
                assert np.array_equal(bits, wanted), ('register', offset, port, register, bits.tolist(), wanted.tolist())
        pc_bits = table(cache, 'pc')[0, 2:18] / A_BIT
        assert np.array_equal(pc_bits, [(anchor >> b) & 1 for b in range(16)]), ('PC anchor', offset, pc_bits.tolist(), anchor)
        count_value = table(cache, 'count')[0, 2]
        wanted_count = 181 * count if getattr(circuit, 'stock_integer_counters', False) else A_BIT * 2.0**-count
        assert count_value == wanted_count, ('PC count', offset, float(count_value), count)
        assert pc == spec.ROM_BASE + 4 * (anchor + count)
        decoded = np.zeros((spec.RAM_WORDS, 32))
        for bank in range(spec.RAM_WORDS >> 8):
            indices = np.array([i for i in range(spec.RAM_WORDS) if i & 15 | (i >> 12) << 4 == bank])
            for half in range(2):
                selected = indices[(indices >> 4 & 1) == half]
                rows = table(cache, f'ram{bank}', half)[selected >> 5 & 127]
                decoded[selected] = (rows[:, 2:34] + rows[:, 34:66]) / A_BIT
        wanted = (np.asarray(ram, dtype=np.uint32)[:, None] >> np.arange(32)) & 1
        assert np.array_equal(decoded, wanted), ('RAM', offset, np.argwhere(decoded != wanted)[:8].tolist())
    return check


def cache_words(cache, count):
    result = []
    for layer in cache.layers[:count]:
        result.extend(t.detach().cpu().clone() for t in (layer.recurrent_states[0], layer.conv_states[0]))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream', type=Path, default=ROOT / '.upstream/qwenputer')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', milestone='M3', stock_generate=True, runs=[], controls=[],
                  started_utc=datetime.now(timezone.utc).isoformat(),
                  reference_tokens_used_as_inputs=False, boot_injection_only=True)
    report['script_sha256'] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                              for name in ('stock_m3.py', 'stock.py', 'constructor.py', 'integer_counters.py', 'reuse.py', 'oracle.py', 'reference.py')}
    output = a.out / 'report.json'
    save(output, report)
    try:
        import numpy as np
        import torch
        torch.set_num_threads(4)
        prep_spec = importlib.util.spec_from_file_location('m3_memory_prepare', ROOT / 'experiments/memory-map/prepare.py')
        prep = importlib.util.module_from_spec(prep_spec)
        prep_spec.loader.exec_module(prep)
        report['prepare_manifest'] = prep.prepare(a.upstream, a.out / 'scratch', 16, 8)
        sys.path.insert(0, str(a.out / 'scratch'))
        from chipformer.rv import spec
        from chipformer.q35.fast import exact_matmuls
        from constructor import build
        from stock import build_model, boot_cache
        from transformers import StoppingCriteria, StoppingCriteriaList
        from transformers.generation.streamers import BaseStreamer
        import reference
        exact_matmuls()
        if a.device == 'cuda':
            torch.cuda.set_per_process_memory_fraction(12 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
        circuit = build(4)
        model, metadata = build_model(circuit, a.device)
        report.update(model=metadata, geometry=circuit.stats(), constructor=circuit.source_manifest,
                      torch=torch.__version__, transformers=__import__('transformers').__version__,
                      forward_implementation=model.forward.__func__.__module__,
                      generate_implementation=model.generate.__func__.__module__,
                      commit_boundary='after a generated final-record token is consumed; generated HALT is not drained')
        save(output, report)
        observe = state_observer(circuit)
        versions = [p._version for p in model.parameters()]

        class Checker(BaseStreamer):
            def __init__(self, wanted):
                self.wanted, self.tokens, self.initial = wanted, [], True
            def put(self, value):
                values = value.cpu().tolist()
                if self.initial:
                    self.initial = False
                    assert values == [[spec.VOCAB, spec.tok('GO')]]
                    return
                for token in values:
                    index = len(self.tokens)
                    self.tokens.append(token)
                    assert token == self.wanted[index], ('token', index, token, self.wanted[index])
            def end(self):
                pass

        class Limit(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                if a.device == 'cuda':
                    assert torch.cuda.mem_get_info()[0] >= 512 * 2**20, 'free VRAM guard'
                return False

        with torch.inference_mode():
            for variant, data in ((0, 7), (0, -13), (1, 7), (1, -13)):
                program, source = workload(variant, data)
                initial = reference.Machine.from_program(program)
                initial.maximum = 4
                wanted, commits, instructions, dots = expected_execution(initial)
                run = dict(program=variant, drom_first_byte=data, instructions=instructions,
                           dots=dots, target_tokens=len(wanted), generated_tokens=0,
                           checked_commits=0, status='running')
                report['runs'].append(run)
                save(output, report)
                cache = boot_cache(model, circuit, initial)
                streamer = Checker(wanted)
                consumed = 0

                def hook(module, args, kwargs, out):
                    nonlocal consumed
                    ids = kwargs['input_ids'][0].tolist()
                    assert len(ids) == 1
                    token = spec.tok('GO') if consumed == 0 else streamer.tokens[consumed - 1]
                    assert ids[0] == token
                    if consumed in commits:
                        observe(out.past_key_values, commits[consumed], consumed)
                        run['checked_commits'] += 1
                    consumed += 1

                handle = model.register_forward_hook(hook, with_kwargs=True)
                try:
                    generated = model.generate(input_ids=torch.tensor([[spec.VOCAB, spec.tok('GO')]], device=a.device),
                        past_key_values=cache, max_new_tokens=len(wanted), do_sample=False,
                        eos_token_id=spec.tok('HALT'), pad_token_id=spec.VOCAB,
                        streamer=streamer, stopping_criteria=StoppingCriteriaList([Limit()]))
                finally:
                    handle.remove()
                    run['generated_tokens'] = len(streamer.tokens)
                    save(output, report)
                tokens = generated[0, 2:].tolist()
                assert tokens == streamer.tokens == wanted
                assert run['checked_commits'] == len(commits)
                saved_cache = cache_words(cache, circuit.n_layers())
                del generated
                sequence = [spec.tok('GO')] + tokens[:-1]
                replay = boot_cache(model, circuit, initial)
                replay_predictions = []
                for offset in range(0, len(sequence), 16):
                    out = model(input_ids=torch.tensor([sequence[offset:offset + 16]], device=a.device),
                                past_key_values=replay, use_cache=True)
                    replay_predictions.extend(out.logits.argmax(-1)[0].tolist())
                assert replay_predictions == tokens, ('chunk token mismatch', variant, data)
                replay_cache = cache_words(replay, circuit.n_layers())
                from cache_diagnosis import differences
                run['cached_chunked_differences'] = differences(circuit, cache, replay)
                save(output, report)
                assert not run['cached_chunked_differences'], ('chunk cache bit mismatch', run['cached_chunked_differences'])
                assert len(saved_cache) == len(replay_cache)
                for index, (cached, chunked) in enumerate(zip(saved_cache, replay_cache)):
                    assert torch.equal(cached, chunked), ('chunk cache mismatch', index,
                        int(torch.count_nonzero(cached != chunked)))
                assert versions == [p._version for p in model.parameters()]
                run.update(status='pass', generated_tokens=len(tokens), mismatches=0,
                           cached_chunked_tokens_exact=True, cached_chunked_state_exact=True,
                           token_sha256=hashlib.sha256(np.array(tokens, dtype='<u4').tobytes()).hexdigest(),
                           program_source=source)
                save(output, report)
                print(json.dumps({k: v for k, v in run.items() if k != 'program_source'}), flush=True)
                del cache, replay, out, saved_cache, replay_cache
            from chipformer.rv import hf
            from chipformer.q35.emit import row
            from chipformer.dsl.graph import Expr
            slots = hf.head_slots(circuit)
            locations = {id(circuit.nodes[i]): slot for i, slot in slots.items()}
            program, _ = workload(0, 7)
            baseline = reference.Machine.from_program(program)
            wanted, _, _, _ = expected_execution(baseline)
            for control in ('memory-bit', 'accumulator-bit', 'pc-bit', 'write-gate-bit', 'token-family-bit'):
                initial = copy.deepcopy(baseline)
                restore = []
                if control == 'memory-bit':
                    initial.drom = bytes([initial.drom[0] ^ 1]) + initial.drom[1:]
                elif control == 'accumulator-bit':
                    initial.x[8] ^= 1
                elif control == 'pc-bit':
                    initial.anchor ^= 1
                elif control == 'write-gate-bit':
                    for bank in range(spec.RAM_WORDS >> 8):
                        layer, head = locations[id(circuit.tables[f'ram{bank}'])]
                        weight = model.model.layers[layer].linear_attn.in_proj_b.weight
                        for half in range(2):
                            selected = weight[head * 2 + half]
                            restore.append((selected, selected.clone()))
                            selected.zero_()
                            row(Expr({0: -128}), model.layout, selected)
                else:
                    dim, = circuit.code[0].terms
                    selected = model.model.norm.weight[model.layout.dim(dim)]
                    restore.append((selected, selected.clone()))
                    selected.neg_()
                if control == 'token-family-bit':
                    expected_boundary = 0
                else:
                    predicted = copy.deepcopy(initial)
                    corrupted_tokens = []
                    while not predicted.halted and len(corrupted_tokens) < 64:
                        original_ram = predicted.ram.copy()
                        corrupted_tokens.extend(predicted.record())
                        if control == 'write-gate-bit':
                            predicted.ram = original_ram
                    expected_boundary = next(i for i, (x, y) in enumerate(zip(wanted, corrupted_tokens)) if x != y)
                checker = Checker(wanted)
                detected = None
                try:
                    cache = boot_cache(model, circuit, initial)
                    model.generate(input_ids=torch.tensor([[spec.VOCAB, spec.tok('GO')]], device=a.device),
                        past_key_values=cache, max_new_tokens=64, do_sample=False,
                        eos_token_id=spec.tok('HALT'), pad_token_id=spec.VOCAB, streamer=checker)
                except AssertionError as error:
                    if error.args and isinstance(error.args[0], tuple) and error.args[0][0] == 'token':
                        detected = error.args[0]
                    else:
                        raise
                finally:
                    for selected, original in restore:
                        selected.copy_(original)
                result = dict(control=control, detected=detected is not None,
                              mismatch=detected, expected_boundary=expected_boundary,
                              generated_tokens=len(checker.tokens))
                report['controls'].append(result)
                save(output, report)
                assert result['detected'], result
                assert detected[1] == expected_boundary, result
                del cache
        report['status'] = 'pass'
        save(output, report)
    except Exception as error:
        report.update(status='fail', error=repr(error), traceback=traceback.format_exc())
        print(report['traceback'], flush=True)
    finally:
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save(output, report)
    raise SystemExit(0 if report['status'] == 'pass' else 1)


if __name__ == '__main__':
    main()
