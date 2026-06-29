import platform
import re

from hydromodels_dlm.utils.logging import log_detail

_FREQ_SUFFIX = re.compile(r'\s*@\s*[\d.]+\s*GHz\s*$', re.IGNORECASE)


def cpu_name():
    model = platform.processor() or 'unknown'
    try:
        with open('/proc/cpuinfo', encoding='utf-8', errors='ignore') as handle:
            for line in handle:
                if line.lower().startswith('model name'):
                    model = line.split(':', 1)[1].strip()
                    break
    except OSError:
        pass
    return _FREQ_SUFFIX.sub('', model).strip()


def resolve_cuda_index(device):
    import torch

    if device is None:
        return 0 if torch.cuda.is_available() else None
    if isinstance(device, torch.device):
        if device.type != 'cuda':
            return None
        return device.index if device.index is not None else 0
    return int(device)


def gpu_name(device=None):
    import torch

    idx = resolve_cuda_index(device)
    if idx is None or not torch.cuda.is_available() or idx >= torch.cuda.device_count():
        return None
    return torch.cuda.get_device_name(idx)


def device_summary(device=None):
    parts = [cpu_name()]
    gpu = gpu_name(device)
    if gpu:
        parts.append(gpu)
    return ', '.join(parts)


def log_runtime_environment(log, *, device=None):
    log_detail(log, 'Device: %s', device_summary(device))
