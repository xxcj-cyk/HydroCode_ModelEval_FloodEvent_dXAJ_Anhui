import numpy as np
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def static_over_steps(x_norm, static):
    n_steps = x_norm.shape[0]
    return np.broadcast_to(static, (n_steps, static.shape[0]))


# ---------------------------------------------------------------------------
# dynamic_only
# ---------------------------------------------------------------------------


def series_dynamic_only(x_norm, static=None):
    return x_norm


def input_dim_dynamic_only(n_dynamic, n_static):
    return n_dynamic


# ---------------------------------------------------------------------------
# static_only
# ---------------------------------------------------------------------------


def series_static_only(x_norm, static):
    if static is None:
        raise ValueError('static_only layout requires static attributes')
    return static_over_steps(x_norm, static)


def input_dim_static_only(n_dynamic, n_static):
    return n_static


# ---------------------------------------------------------------------------
# dynamic_static
# ---------------------------------------------------------------------------


def series_dynamic_static(x_norm, static=None):
    return x_norm


def merge_dynamic_static(x_norm, static):
    if static is None:
        raise ValueError('dynamic_static layout requires static attributes')
    return np.concatenate([x_norm, static_over_steps(x_norm, static)], axis=-1)


def input_dim_dynamic_static(n_dynamic, n_static):
    return n_dynamic + n_static


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


INPUT_LAYOUT_REGISTRY = {
    'dynamic_only': {
        'series': series_dynamic_only,
        'merge_static': None,
        'input_dim': input_dim_dynamic_only,
        'uses_static': False,
    },
    'static_only': {
        'series': series_static_only,
        'merge_static': None,
        'input_dim': input_dim_static_only,
        'uses_static': True,
    },
    'dynamic_static': {
        'series': series_dynamic_static,
        'merge_static': merge_dynamic_static,
        'input_dim': input_dim_dynamic_static,
        'uses_static': True,
    },
}


def merge_static_numpy(x_norm, static, layout):
    if not layout['uses_static']:
        return np.asarray(x_norm, dtype=np.float32)
    if static is None:
        raise ValueError(f"input_layout {layout!r} requires static attributes")
    merge_fn = layout.get('merge_static')
    if merge_fn is not None:
        return np.asarray(merge_fn(x_norm, static), dtype=np.float32)
    return np.asarray(layout['series'](x_norm, static), dtype=np.float32)


def merge_model_input_torch(series, static=None):
    if static is None:
        return series
    if series.dim() == 2:
        static_steps = static.unsqueeze(0).expand(series.shape[0], -1)
        return torch.cat([series, static_steps], dim=-1)
    if series.dim() == 3:
        static_steps = static.unsqueeze(1).expand(-1, series.shape[1], -1)
        return torch.cat([series, static_steps], dim=-1)
    raise ValueError(f'series must be 2D or 3D, got shape {tuple(series.shape)}')
