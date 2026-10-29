"""Full-map stock allocation and bounded autonomous guests, gated on accepted M3."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import traceback

from prepare import ROOT
from stock_m3 import save, state_observer, workload


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream', type=Path, default=ROOT / '.upstream/qwenputer')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--m3-report', type=Path, required=True)
    p.add_argument('--stories', type=Path, required=True)
    p.add_argument('--device', choices=['cuda'], default='cuda')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    path = a.out / 'report.json'
    report = dict(status='running', milestone='M4', runs=[], stock_generate=True,
                  boot_injection_only=True, reference_tokens_used_as_inputs=False,
                  started_utc=datetime.now(timezone.utc).isoformat())
    report['worker_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    save(path, report)
    try:
        m3 = json.loads(a.m3_report.read_text())
        assert m3['status'] == 'pass' and len(m3['runs']) == 4 and len(m3['controls']) == 5
        report['m3_report_sha256'] = hashlib.sha256(a.m3_report.read_bytes()).hexdigest()
        for name, digest in m3['script_sha256'].items():
            assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() == digest, name
        import torch
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(15 * 2**30 / total)
        torch.cuda.reset_peak_memory_stats()
        report.update(torch=torch.__version__, transformers=__import__('transformers').__version__,
                      allocator_cap_bytes=15*2**30, free_vram_guard_bytes=512*2**20,
                      gpu=torch.cuda.get_device_name(), total_vram_bytes=total)

        def memory():
            free, total = torch.cuda.mem_get_info()
            return dict(allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved(),
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                        free_bytes=free, total_bytes=total)

        def guard(required=0):
            free, _ = torch.cuda.mem_get_info()
            cached = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
            if free-max(0, required-cached) < 512*2**20:
                before = memory()
                torch.cuda.empty_cache()
                free, _ = torch.cuda.mem_get_info()
                report.setdefault('allocator_reclaims', []).append(dict(before=before, after=memory(), next_allocation=required))
                save(path, report)
                if free-required < 512*2**20:
                    raise RuntimeError(f'free VRAM guard: free={free}, next_allocation={required}')

        report['before_allocation'] = memory()
        save(path, report)
        prep_spec = importlib.util.spec_from_file_location('m4_memory_prepare', ROOT / 'experiments/memory-map/prepare.py')
        prep = importlib.util.module_from_spec(prep_spec)
        prep_spec.loader.exec_module(prep)
        report['prepare_manifest'] = prep.prepare(a.upstream, a.out / 'scratch', 64, 4096)
        sys.path.insert(0, str(a.out / 'scratch'))
        from chipformer.rv import elf, spec
        from constructor import build
        from stock import build_model, boot_cache
        import reference
        from oracle import execute, serialize
        from transformers import StoppingCriteria, StoppingCriteriaList
        from transformers.generation.streamers import BaseStreamer
        circuit = build(4, stock_integer_counters=True)
        report.update(geometry=circuit.stats(), constructor=circuit.source_manifest)
        from chipformer.q35.tiny import BASE, pick_layout
        from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
        layout = pick_layout(len(circuit.dims))
        count = circuit.n_layers()
        config = Qwen3_5TextConfig(**dict(BASE, hidden_size=layout.hidden,
            vocab_size=-(-(spec.VOCAB+1)//64)*64, num_hidden_layers=count+1,
            layer_types=['linear_attention']*count+['full_attention'],
            linear_num_key_heads=circuit.target.key_heads,
            linear_num_value_heads=circuit.target.value_heads,
            intermediate_size=circuit.target.intermediate))
        with torch.device('meta'):
            shape_model = Qwen3_5ForCausalLM(config)
        parameter_count = sum(p.numel() for p in shape_model.parameters())
        report['required_model'] = dict(parameters=parameter_count, bf16_parameter_bytes=parameter_count*2,
                                        hidden=layout.hidden, config=config.to_dict())
        del shape_model
        save(path, report)
        model, metadata = build_model(circuit, 'cuda', memory_guard=guard)
        report.update(model=metadata, after_allocation=memory())
        save(path, report)
        observe = state_observer(circuit)
        versions = [p._version for p in model.parameters()]
        from chipformer.rv.asm import assemble
        _, source = workload(0, 7)
        program = assemble(source.replace('li s3, 128', 'li s3, 2048'))
        guests = [('ndot', program), ('stories260K', None)]
        with torch.inference_mode():
            for name, program in guests:
                if program is None:
                    binding = json.loads(Path(__file__).with_name('legacy-stories.json').read_text())
                    digest = hashlib.sha256(a.stories.read_bytes()).hexdigest()
                    assert digest == binding['elf_sha256']
                    program = elf.program(a.stories)
                initial = reference.Machine.from_program(program)
                initial.maximum = 4
                run = dict(guest=name, status='running', generated_tokens=0, checked_commits=0,
                           independent_dot_checks=0, memory_before=memory())
                if name == 'stories260K':
                    run['elf_sha256'] = digest
                report['runs'].append(run)
                save(path, report)
                import copy
                machine = copy.deepcopy(initial)

                class Checker(BaseStreamer):
                    def __init__(self):
                        self.tokens, self.record, self.at, self.boundary = [], [], 0, -1
                        self.initial = True
                    def put(self, value):
                        if self.initial:
                            self.initial = False
                            return
                        for token in value.cpu().tolist():
                            if self.at == len(self.record):
                                word = machine.rom[(machine.pc-spec.ROM_BASE)//4]
                                ins = reference.decode(word, 4)
                                independent = None
                                if machine.n != spec.N_MAX and ins.op == 'NDOT4':
                                    result = execute(word, machine.x, struct.pack('<'+'I'*len(machine.ram), *machine.ram), machine.drom, 4)
                                    independent = serialize(result, word, machine.pc)
                                self.record = machine.record()
                                if independent is not None:
                                    assert self.record == independent and tuple(machine.x) == result.regs
                                    run['independent_dot_checks'] += 1
                                self.at = 0
                            assert token == self.record[self.at], ('token', len(self.tokens), token, self.record[self.at])
                            self.tokens.append(token)
                            self.at += 1
                            if self.at == len(self.record):
                                self.boundary = len(self.tokens)
                            run['generated_tokens'] = len(self.tokens)
                            if len(self.tokens) % 1024 == 0:
                                run['memory'] = memory()
                                (a.out / f'{name}.tokens.u32').write_bytes(struct.pack('<'+'I'*len(self.tokens), *self.tokens))
                                save(path, report)
                                print(json.dumps(dict(guest=name, generated_tokens=len(self.tokens))), flush=True)
                    def end(self):
                        pass

                class Guard(StoppingCriteria):
                    def __call__(self, input_ids, scores, **kwargs):
                        guard()
                        return False

                checker = Checker()
                cache = boot_cache(model, circuit, initial)
                consumed = 0

                def hook(module, args, kwargs, out):
                    nonlocal consumed
                    assert kwargs['input_ids'].shape[1] == 1
                    if consumed == checker.boundary and consumed > 0:
                        expected = (tuple(machine.x), tuple(machine.ram), machine.anchor, machine.n, machine.pc)
                        observe(out.past_key_values, expected, consumed)
                        run['checked_commits'] += 1
                    consumed += 1

                handle = model.register_forward_hook(hook, with_kwargs=True)
                try:
                    generated = model.generate(input_ids=torch.tensor([[spec.VOCAB, spec.tok('GO')]], device='cuda'),
                        past_key_values=cache, max_new_tokens=8192, do_sample=False,
                        eos_token_id=spec.tok('HALT'), pad_token_id=spec.VOCAB,
                        streamer=checker, stopping_criteria=StoppingCriteriaList([Guard()]))
                finally:
                    handle.remove()
                    raw = struct.pack('<'+'I'*len(checker.tokens), *checker.tokens)
                    (a.out / f'{name}.tokens.u32').write_bytes(raw)
                    run['token_sha256'] = hashlib.sha256(raw).hexdigest()
                    save(path, report)
                assert len(checker.tokens) == 8192
                assert generated[0, 2:].tolist() == checker.tokens
                assert versions == [p._version for p in model.parameters()]
                run.update(status='pass', mismatches=0, token_sha256=hashlib.sha256(raw).hexdigest(), memory_after=memory())
                save(path, report)
                del generated, cache
        report['status'] = 'pass'
    except Exception as error:
        if report['runs'] and report['runs'][-1]['status'] == 'running':
            report['runs'][-1].update(status='fail', error=repr(error))
        report.update(status='oom' if type(error).__name__ in ('OutOfMemoryError', 'AcceleratorError') and 'memory' in str(error).lower() else 'fail',
                      error=repr(error), traceback=traceback.format_exc())
        print(report['traceback'], flush=True)
    finally:
        if 'torch' in locals() and torch.cuda.is_available():
            report['final_memory'] = memory()
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save(path, report)
    raise SystemExit(report['status'] != 'pass')


if __name__ == '__main__':
    main()
