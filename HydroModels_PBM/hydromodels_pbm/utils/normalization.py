import numpy as np

from hydromodels_pbm.config.model_config import MODEL_PARAM_DICT


# ---------------------------------------------------------------------------
# Parameter arrays
# ---------------------------------------------------------------------------


def to_array_2d(parameters):
    arr = np.asarray(parameters, dtype=np.float64)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr


def param_bounds(model_name):
    if model_name not in MODEL_PARAM_DICT:
        raise ValueError(
            f'unknown model {model_name!r}; registered: {sorted(MODEL_PARAM_DICT)}'
        )
    bounds = MODEL_PARAM_DICT[model_name].values()
    lo = np.array([float(v[0]) for v in bounds], dtype=np.float64)
    hi = np.array([float(v[1]) for v in bounds], dtype=np.float64)
    return lo, hi


def normalize(parameters, model_name):
    lo, hi = param_bounds(model_name)
    arr = to_array_2d(parameters)
    if arr.shape[1] != lo.size:
        raise ValueError(
            f'parameter columns {arr.shape[1]} != '
            f'model {model_name!r} parameter count {lo.size}'
        )
    return (arr - lo) / (hi - lo)


def denormalize(parameters, model_name):
    lo, hi = param_bounds(model_name)
    arr = to_array_2d(parameters)
    if arr.shape[1] != lo.size:
        raise ValueError(
            f'parameter columns {arr.shape[1]} != '
            f'model {model_name!r} parameter count {lo.size}'
        )
    return lo + arr * (hi - lo)
