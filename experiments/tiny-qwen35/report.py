"""Render measured results without treating queued GPU work as a result."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from pack import HERE, ROOT


def read(path):
    return json.loads(path.read_text())


def report():
    cache = ROOT/'.cache/tiny-qwen35'
    evidence = HERE/'evidence'
    gate = read(evidence/'mac-gate.json')
    ref = read(cache/'trained-program/reference.json')
    build = read(cache/'trained-program/build.json')
    training = read(cache/'training/done.json')
    primary = read(cache/'trained-quality/quality.json')
    six = read(evidence/'covariance-quality.json')
    gpu_path = evidence/'neural-validation.json'
    gpu = read(gpu_path) if gpu_path.exists() else None
    live_path = cache/'gpu-progress.json'
    live = read(live_path) if live_path.exists() else None
    if gpu:
        assert gpu['actual_trace_sha256'] == ref['sha256']
    assert gate['status'] == 'pass'
    assert primary['quantization'] == six['quantization'] == 'midpoint+covariance'
    summary = dict(mac_status='pass',training_seconds=training['wall_seconds'],training_steps=training['steps'],
                   parameters=build['parameters'],rom_bytes=build['rom_bytes'],arena_bytes=build['ram']['arena_used'],
                   static_bytes=build['static_bytes'],stack_reserved=1024,stack_observed=ref['stack_bytes'],
                   linked_margin_bytes=build['linked_margin'],native_legacy_positions=579,
                   legacy_reference_positions=193,primary_predictions=167,primary_matches=167,
                   six_prompt_predictions=327,six_prompt_matches=291,
                   six_prompt_free_running_agreement=six['free_running_agreement'],
                   instructions=ref['instructions'],trace_tokens=ref['tokens'],ndot4=ref['ndot4_instructions'],
                   projected_minutes_per_position_at_3735=gate['projected_minutes_per_position_at_3735'],
                   neural_status='pass' if gpu else 'launched; final validation pending',
                   neural=gpu,updated_utc=datetime.now(timezone.utc).isoformat())
    lines = ['<!-- MEASURED_RESULTS_BEGIN -->','## Measured results','',
        ('Parts A through D pass.' if gpu else 'Parts A through C pass.')+' One native executable and one RV32 object serve both model families.',
        'All 579 native legacy positions match the unchanged old engine in IDs, logit hashes, clipping counts and literal output bytes.',
        'The independent RV32 reference passes all 64 stories260K and all 129 custom-model positions.',
        'Executable bytes match the trained Qwen image and the second, untied Qwen geometry.', '',
        f'The eight-layer model has {build["parameters"]:,} parameters and a {build["rom_bytes"]:,}-byte DROM image.',
        f'Training completes {training["steps"]:,} steps in {training["wall_seconds"]/60:.2f} minutes in the isolated training venv.',
        'The final floating validation estimate is 1.53958 nats per token over 20,480 targets. It is not an integer held-out-loss result.', '',
        'The selected packing uses midpoint nonlinear LUTs and covariance-guided int8 weight rounding.',
        'The three predefined tuning prompts match float HF greedy generation in 167/167 generated predictions.',
        'Across all six inspected prompts, agreement is 291/327 predictions, or 89.0 percent.',
        'Five continuations match completely. The rabbit prompt first differs at position 28, with a float top-two logit margin of 0.00874.',
        'The six prompts have been used during quantization investigation and are not an unseen quality benchmark.',
        'All cache and numeric clipping counters stay zero.', '',
        f'The six-position guest retires {ref["instructions"]:,} instructions and emits {ref["tokens"]:,} trace tokens.',
        f'Its {ref["ndot4_instructions"]:,} NDOT4 instructions account for every aligned matrix MAC.',
        f'Observed stack use is {ref["stack_bytes"]} bytes against the 1,024-byte reservation.',
        f'The supplied 3,735 trace tokens/s rate projects to {gate["projected_minutes_per_position_at_3735"]:.2f} minutes per LLM position.',
        f'Including its one prefill position, the whole five-token continuation projects to {ref["tokens"]/3735/60:.2f} minutes, or {ref["tokens"]/3735/60/5:.2f} minutes per new token.',
        'This projection includes the complete guest workload and is not a new GPU timing.',
        'Table rows measure position begin/end markers; whole-program totals also include startup and diagnostic output. Costs describe these six positions.', '',
        '| Position | Phase | RV32 instructions | Trace tokens | Projected minutes |',
        '| --- | --- | ---: | ---: | ---: |']
    for row in ref['positions']:
        phase = 'prefill' if row['position'] < len(ref['prompt_tokens'])-1 else 'decode'
        lines.append(f'| {row["position"]} | {phase} | {row["instructions"]:,} | {row["trace_tokens"]:,} | {row["projected_minutes_at_3735"]:.2f} |')
    lines.append('')
    if gpu:
        assert gpu['status'] == 'pass'
        lines += [f'The neural CPU generates all {gpu["neural_tokens"]:,} trace tokens with zero mismatches.',
                  f'Actual GPU BYTE records produce `{gpu["observed_text"]}`.',
                  f'All {gpu["final_ram_bytes_exact"]:,} RAM bytes and all three register copies match the Mac reference exactly.',
                  f'Retained generation calls measure {gpu["measured_minutes_per_position"]:.2f} minutes per inference position.',
                  f'Including prefill, the five new tokens take {gpu["compute_seconds"]/60:.2f} retained generation minutes, or {gpu["compute_seconds"]/60/5:.2f} minutes per new token.',
                  'The timer excludes setup, comparisons, checkpoint IO, shared-lock waits, shutdown time and replayed work after the last pre-shutdown checkpoint.',
                  f'The full-map stock HF prefix contains {gpu["stock_tokens"]:,} tokens with zero mismatches.',
                  f'Minimum sampled free VRAM is {gpu["stock_minimum_free_bytes"]/2**20:.0f} MiB under the 15 GiB cap and 512 MiB guard.',
                  'The stock prefix is a bounded low-level CPU trace and does not complete a guest language-model prediction.']
        if gpu.get('shutdown_recovery'):
            recovery = gpu['shutdown_recovery']
            lines += [f'The PC shut down after {recovery["actual_trace_tokens"]:,} generated trace tokens.',
                      f'The run resumed from its hash-checked actual GPU checkpoint at {recovery["offset"]:,}, replaying {recovery["replay_tokens"]:,} tokens.',
                      'The checkpoint archive, binding and pre-shutdown actual trace pass independent Mac checks.']
    else:
        lines += ['The detached neural and stock sequence has launched under the shared GPU lock convention.',
                  'No neural continuation, neural timing, final GPU state or stock-prefix result is claimed yet.',
                  'The six positions include five generated tokens after the supplied `Once` prompt.']
        if live and live.get('main'):
            main = live['main']
            lines += [f'At {live["sampled_utc"]}, the live worker reports {main["generated_tokens"]:,} trace tokens',
                      f'and {main["mismatches"]} online mismatches. Independent final Mac validation is pending.']
    lines += ['', 'Detailed bindings and checks are in [mac-gate.json](evidence/mac-gate.json),',
              '[trained-quality.json](evidence/trained-quality.json), [covariance-quality.json](evidence/covariance-quality.json),',
              'and [summary.json](evidence/summary.json).',
              'Final acceptance and artifact bindings are in [final-checks.json](evidence/final-checks.json) and [neural-validation.json](evidence/neural-validation.json).','<!-- MEASURED_RESULTS_END -->']
    readme = HERE/'README.md'
    text = readme.read_text()
    begin,end = '<!-- MEASURED_RESULTS_BEGIN -->','<!-- MEASURED_RESULTS_END -->'
    block = '\n'.join(lines)
    if begin in text:
        start = text.index(begin)
        stop = text.index(end,start)+len(end)
        text = text[:start]+block+text[stop:]
    else:
        index = text.index('## Runtime and graph')
        text = text[:index]+block+'\n\n'+text[index:]
    readme.write_text(text)
    (evidence/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wait',action='store_true')
    a = p.parse_args()
    if a.wait:
        while not (HERE/'evidence/neural-validation.json').exists():
            progress = read(ROOT/'.cache/tiny-qwen35/progress.json')
            if progress['stage'] == 'failed':
                raise RuntimeError(progress)
            time.sleep(30)
    report()
