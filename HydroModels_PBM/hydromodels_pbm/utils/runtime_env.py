import platform
import random
import re

import numpy as np

from hydromodels_pbm.config.model_config import MODEL_INPUT_KEYS
from hydromodels_pbm.workflow.simulator import HydroSimulator, denormalize_param_dict


# ---------------------------------------------------------------------------
# Runtime logging
# ---------------------------------------------------------------------------


def cpu_model_name():
    model = platform.processor() or 'unknown'
    try:
        with open('/proc/cpuinfo', encoding='utf-8', errors='ignore') as handle:
            for line in handle:
                if line.lower().startswith('model name'):
                    model = line.split(':', 1)[1].strip()
                    break
    except OSError:
        pass
    return re.sub(r'\s*@\s*[\d.]+\s*GHz\s*$', '', model, flags=re.IGNORECASE).strip()


def log_runtime_environment(log):
    log.info('  Device: %s', cpu_model_name())


# ---------------------------------------------------------------------------
# Numba JIT warmup
# ---------------------------------------------------------------------------


def warmup_numba_runtime(model_name, warmup_length, random_seed=None):
    if random_seed is not None:
        np.random.seed(int(random_seed))
        random.seed(int(random_seed))

    warmed = warmup_numba_runtime.__dict__.setdefault('warmed_models', set())
    if model_name in warmed:
        return

    input_keys = MODEL_INPUT_KEYS.get(model_name)
    n_features = len(input_keys) if input_keys else 2
    wu = max(0, int(warmup_length))
    n_steps = wu + 240

    simulator = HydroSimulator(model_name)
    norm = np.full(len(simulator.param_names), 0.5, dtype=np.float64)
    params = denormalize_param_dict(norm, model_name)
    drivers = np.full((n_steps, 1, n_features), 1.0, dtype=np.float64)
    simulator.simulate(drivers, params, warmup_length=wu)
    warmed.add(model_name)
