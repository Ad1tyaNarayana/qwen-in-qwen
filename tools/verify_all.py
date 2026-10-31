"""Run release checks in isolated tracked-file copies; never consume the caller's cache."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from release_lib import ROOT, sha, unpack_verified


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tier', type=int, choices=[0, 1, 2], default=0)
    p.add_argument('--bundle', help='Local archive path or HTTPS URL, required for tiers 1 and 2')
    p.add_argument('--json', type=Path, help='Write fresh results outside historical evidence')
    p.add_argument('--full-ir', action='store_true', help='Also run the long complete IR CPU gate at tier 2')
    a = p.parse_args()
    if sys.flags.optimize:
        p.error('Do not disable verifier assertions with -O or PYTHONOPTIMIZE')
    results = []
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='qwenputer-verify-') as temporary:
        temporary = Path(temporary)
        work = temporary / 'repo'
        work.mkdir()
        paths = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).split(b'\0')
        for raw in paths:
            if not raw:
                continue
            path = Path(os.fsdecode(raw))
            source = ROOT / path
            target = work / path
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.is_symlink():
                target.symlink_to(source.readlink())
            else:
                shutil.copyfile(source, target)
        # History resolves old source hashes explicitly. No git operation writes here.
        gitdir = subprocess.check_output(['git', 'rev-parse', '--absolute-git-dir'], cwd=ROOT, text=True).strip()
        (work / '.git').write_text('gitdir: ' + gitdir + '\n')
        upstream = work / '.upstream/qwenputer'
        shutil.copytree(work / 'vendor/qwenputer', upstream)
        (work / '.cache/neural-accelerator').mkdir(parents=True)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1',
                   TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1')
        env.pop('PYTHONOPTIMIZE', None)
        env['PYTHONPATH'] = str(upstream / 'experimental/riscv')

        def record(claim, status, reason, seconds=0):
            results.append(dict(claim=claim, status=status, reason=reason, seconds=round(seconds, 3)))
            print(f'{status:7s} | {claim:34s} | {seconds:8.2f}s | {reason}', flush=True)

        def run(claim, command, cwd=work, timeout=600):
            begin = time.perf_counter()
            try:
                completed = subprocess.run([sys.executable, '-B', *command], cwd=cwd, env=env,
                                           capture_output=True, text=True, timeout=timeout)
                if completed.returncode:
                    record(claim, 'FAIL', (completed.stderr or completed.stdout)[-1800:].strip(), time.perf_counter()-begin)
                    return False
                record(claim, 'PASS', 'Existing verifier completed; output writes isolated in temporary copy', time.perf_counter()-begin)
                return True
            except subprocess.TimeoutExpired:
                record(claim, 'FAIL', f'Exceeded {timeout}s bound', time.perf_counter()-begin)
                return False

        run('release attack controls', ['-m', 'pytest', '-q', 'tools/test_release.py'])
        run('release policy', ['tools/check_release.py', '--root', str(ROOT)], cwd=ROOT)
        # Hashes detect changes, not execution provenance. This covers every retained
        # evidence/report/data file selected in the immutable baseline manifest.
        begin = time.perf_counter()
        try:
            evidence = json.loads((ROOT / 'release/tracked-evidence.json').read_text())
            for relative, digest in evidence.items():
                if sha(ROOT / relative) != digest:
                    raise ValueError(f'Tracked evidence SHA-256 mismatch: {relative}')
            record('tracked evidence integrity', 'PASS', f'{len(evidence)} baseline files; hashes do not prove provenance', time.perf_counter()-begin)
        except (OSError, ValueError) as error:
            record('tracked evidence integrity', 'FAIL', str(error), time.perf_counter()-begin)
        run('CHIP-8 retained traces and gameplay', ['tools/check_game.py'])
        run('fused state and timing evidence', ['experiments/tiny-llm-ndot-fused/check_evidence.py'])
        run('fused executor ROM records', ['experiments/dmc-rom-fusion/check_evidence.py'])
        run('fused executor race diagnosis', ['experiments/dmc-rom-fusion/check_diagnosis.py'])
        run('fused executor feedback records', ['experiments/dmc-feedback-fusion/check_evidence.py'])
        run('checkpoint CPU corruption controls', ['experiments/checkpoint-integrity/test_restore.py'])
        run('NDOT compiler intrinsic encodings', ['experiments/neural-accelerator/test_intrinsic.py'])
        run('NDOT ISA exhaustive software gate', ['-m', 'pytest', '-q', 'experiments/neural-accelerator/test_isa.py'])
        run('pytest ISA and IR collection', ['-m', 'pytest', '--collect-only', '-q',
            'experiments/neural-accelerator/test_isa.py', 'experiments/neural-accelerator/test_ir.py'])

        ready = False
        bundle = None
        if a.tier >= 1:
            begin = time.perf_counter()
            try:
                if not a.bundle:
                    raise ValueError('--bundle is required at tier 1 or 2')
                if a.bundle.startswith('https://'):
                    bundle = temporary / 'download.tar.gz'
                    with urllib.request.urlopen(a.bundle, timeout=60) as source, bundle.open('wb') as out:
                        # Refuse an unbounded download; authenticate before extraction/use.
                        remaining = 2_000_000_001
                        while remaining:
                            block = source.read(min(1024*1024, remaining))
                            if not block:
                                break
                            out.write(block)
                            remaining -= len(block)
                        if not remaining:
                            raise ValueError('Download exceeds 2 GB')
                elif '://' in a.bundle:
                    raise ValueError('Only local paths or HTTPS URLs are supported')
                else:
                    bundle = Path(a.bundle).resolve()
                unpack_verified(bundle, work / '.cache')
                record('bundle authentication', 'PASS', 'Archive and every allowlisted file match tracked SHA-256', time.perf_counter()-begin)
                ready = run('bundle release policy', ['tools/check_release.py', '--root', str(ROOT), '--bundle', str(bundle)], cwd=ROOT)
            except Exception as error:
                record('bundle authentication', 'FAIL', str(error), time.perf_counter()-begin)
        claims = [
            ('stories260K complete retained run', 'story'),
            ('tiny Qwen complete and stock prefix', 'qwen'),
            ('showcase native and reference', 'showcase'),
            ('showcase full GPU and stock prefix', 'showcase-gpu'),
            ('explorer application data', 'explorer-showcase'),
            ('explorer graph and recorded state', 'explorer'),
            ('controlled scalar/NDOT and restore', 'controlled'),
            ('story export rejection controls', 'controls'),
        ]
        for claim, mode in claims:
            if ready:
                run(claim, ['tools/release_verify.py', mode], timeout=900)
            else:
                record(claim, 'SKIPPED', 'Requires tier 1 authenticated bundle' if a.tier == 0 else 'Bundle authentication/policy failed')
        if a.tier >= 2:
            run('bounded stock HF CPU regeneration', ['tools/hf_smoke.py'], timeout=900)
        else:
            record('bounded stock HF CPU regeneration', 'SKIPPED', 'Requires tier 2; no full guest completion claim')
        if a.tier >= 2 and a.full_ir:
            run('complete neural IR gate', ['-m', 'pytest', '-q', 'experiments/neural-accelerator/test_ir.py'], timeout=3600)
        else:
            record('complete neural IR gate', 'SKIPPED', 'Long CPU gate; tier 2 --full-ir, bounded to 3600s')
        record('floating-point quality agreement', 'SKIPPED', '291/327 is retained inspected-prompt evidence; float quality sweeps not rerun')
        record('full GPU reruns and model training', 'SKIPPED', 'Tier 3 manual Windows CUDA workflow; no GPU used')
    elapsed = time.perf_counter()-started
    counts = {status: sum(r['status'] == status for r in results) for status in ['PASS', 'FAIL', 'SKIPPED']}
    summary = dict(tier=a.tier, counts=counts, seconds=round(elapsed, 3), claims=results)
    print(f'\nTier {a.tier}: {counts}; elapsed {elapsed:.3f}s. SKIPPED claims were NOT checked and are NOT passes.')
    print('Not checked: ' + '; '.join(r['claim'] + ': ' + r['reason'] for r in results if r['status'] == 'SKIPPED'))
    if a.json:
        a.json.write_text(json.dumps(summary, indent=2)+'\n')
    return int(counts['FAIL'] > 0)


if __name__ == '__main__':
    raise SystemExit(main())
