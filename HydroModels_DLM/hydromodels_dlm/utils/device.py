import warnings

import torch


def resolve_device(device_idx=0):
    """Map training_cfgs['device'] (GPU index) to torch.device."""
    idx = 0 if device_idx is None else int(device_idx)
    if idx < 0:
        raise ValueError(f'device must be >= 0, got {idx}')

    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        if idx >= n:
            raise RuntimeError(f'device {idx} out of range (available: 0..{n - 1})')
        return torch.device(f'cuda:{idx}')

    warnings.warn('CUDA unavailable; using CPU', stacklevel=2)
    return torch.device('cpu')
