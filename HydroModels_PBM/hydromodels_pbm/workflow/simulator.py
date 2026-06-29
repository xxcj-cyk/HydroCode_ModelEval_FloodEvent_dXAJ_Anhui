import numpy as np

from hydromodels_pbm.config.model_config import MODEL_INPUT_KEYS, MODEL_PARAM_DICT
from hydromodels_pbm.model.model_dict import MODEL_DICT
from hydromodels_pbm.utils.normalization import denormalize, normalize, to_array_2d


# ---------------------------------------------------------------------------
# Input check
# ---------------------------------------------------------------------------


def check_model_inputs(model_name, input_keys):
    required = MODEL_INPUT_KEYS.get(model_name)
    if required is None:
        return
    keys = tuple(input_keys)
    if keys != required:
        raise ValueError(
            f'model {model_name!r} requires dynamic_inputs keys {required!r} '
            f'in this order, got {keys!r}'
        )


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


def denormalize_param_dict(norm, model_name):
    names = list(MODEL_PARAM_DICT[model_name].keys())
    arr = np.asarray(norm, dtype=np.float64).reshape(1, -1)
    values = denormalize(arr, model_name).reshape(-1)
    return {name: float(value) for name, value in zip(names, values)}


def normalize_param_dict(params, model_name):
    names = list(MODEL_PARAM_DICT[model_name].keys())
    arr = np.array([[float(params[name]) for name in names]], dtype=np.float64)
    return normalize(arr, model_name).reshape(-1)


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


class HydroSimulator:
    def __init__(self, model_name):
        if model_name not in MODEL_PARAM_DICT:
            raise ValueError(
                f'unknown model {model_name!r}; registered: {sorted(MODEL_PARAM_DICT)}'
            )
        self.model_name = model_name
        self.model_fn = MODEL_DICT[model_name]
        self.param_names = list(MODEL_PARAM_DICT[model_name].keys())

    def simulate(self, p_and_e, params, warmup_length):
        if isinstance(params, dict):
            arr = np.array(
                [[float(params[name]) for name in self.param_names]],
                dtype=np.float64,
            )
        else:
            arr = to_array_2d(params)
        out = self.model_fn(p_and_e, arr, warmup_length=warmup_length)
        if isinstance(out, tuple):
            return out[0]
        return out
