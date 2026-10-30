"""Versioned SHA-256 integrity for saved state, checked before GPU mutation.

This detects corruption, not a malicious writer able to replace both data and
checksum. Legacy files without integrity metadata are intentionally rejected.
"""
import hashlib
import json

VERSION = 1


class CheckpointIntegrityError(AssertionError):
    pass


def content_digest(state):
    import torch

    def describe(value):
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            return {'tensor': {'dtype': str(tensor.dtype), 'shape': list(tensor.shape),
                               'sha256': hashlib.sha256(raw).hexdigest()}}
        if isinstance(value, dict):
            if not all(isinstance(key, str) for key in value):
                raise TypeError('Checkpoint dictionary keys must be strings')
            return {'dict': {key: describe(item) for key, item in sorted(value.items())}}
        if isinstance(value, (tuple, list)):
            return {type(value).__name__: [describe(item) for item in value]}
        if value is None or type(value) in (str, int, float, bool):
            return {type(value).__name__: value}
        raise TypeError(f'Unsupported checkpoint value: {type(value).__name__}')

    payload = {key: value for key, value in state.items() if key != 'integrity'}
    raw = json.dumps(describe(payload), sort_keys=True, separators=(',', ':'),
                     allow_nan=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def seal(state):
    state['integrity'] = {'version': VERSION, 'sha256': content_digest(state)}


def verify(state):
    integrity = state.get('integrity')
    if not isinstance(integrity, dict) or integrity.get('version') != VERSION:
        raise CheckpointIntegrityError('Checkpoint integrity metadata missing or unsupported; legacy checkpoints cannot be restored. Start a fresh run.')
    if integrity.get('sha256') != content_digest(state):
        raise CheckpointIntegrityError('Checkpoint content checksum mismatch; refusing restore before copying any state to the GPU.')
