"""Adapters for existing verifiers. Called only inside verify_all's disposable copy."""
import importlib.util
from pathlib import Path
import subprocess
import sys
from release_lib import ROOT


def load(relative):
    path = ROOT / relative
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location('release_verifier', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    mode = sys.argv[1]
    cache = ROOT / '.cache'
    if mode == 'story':
        module = load('experiments/tiny-llm-ndot-fused/verify_story.py')
        sys.argv = ['verify_story', '--inputs', str(cache/'tiny-llm/ndot-story38-20261030'),
                    '--actual', str(cache/'tiny-llm/ndot-story38-20261030-actual'),
                    '--source-revision', '017c39c', '--out', str(cache/'story-validation.json')]
        module.main()
    elif mode == 'qwen':
        module = load('experiments/tiny-qwen35/verify_gpu.py')
        module.verify(cache/'tiny-qwen35/trained-program', cache/'tiny-qwen35/actual-covariance-20261030', source_revision='017c39c')
    elif mode == 'showcase':
        # Rebuild native binaries on the verifier's platform. Never ship host executables.
        here = ROOT/'experiments/showcase'
        for demo in ['sha256', 'life', 'tictactoe', 'mnist', 'mnist-scalar']:
            source = here/('mnist' if demo == 'mnist-scalar' else demo)/'main.c'
            cmd = ['clang', '-O2', '-fwrapv', '-std=c11', '-DNATIVE', '-I'+str(here),
                   '-I'+str(cache/'showcase'/demo), '-I'+str(ROOT/'experiments/neural-accelerator')]
            if demo == 'mnist-scalar':
                cmd += ['-DSCALAR']
            subprocess.run([*cmd, str(source), '-o', str(cache/'showcase'/demo/'native')], check=True)
        load('experiments/showcase/verify_local.py').main()
    elif mode == 'showcase-gpu':
        module = load('experiments/showcase/verify_gpu.py')
        sys.path.insert(0, str(ROOT/'experiments/neural-accelerator'))
        from prepare import prepare
        prepare(cache/'showcase/release-package')
        from chipformer.rv import spec
        for demo in ['sha256', 'life', 'tictactoe', 'mnist', 'mnist-scalar']:
            module.verify(cache/'showcase/actual-20261030a', demo, spec, source_revision='017c39c')
    elif mode == 'explorer-showcase':
        load('experiments/explorer/verify_showcase.py').main()
    elif mode == 'explorer':
        load('experiments/explorer/verify_data.py').main()
    elif mode == 'controlled':
        load('experiments/checkpoint-integrity/verify.py').main()
    elif mode == 'controls':
        module = load('experiments/checkpoint-integrity/test_verifier_exports.py')
        sys.argv = ['test_verifier_exports', '--story-only', '--out', str(cache/'export-controls.json')]
        module.main()
    else:
        raise ValueError(mode)


if __name__ == '__main__':
    main()
