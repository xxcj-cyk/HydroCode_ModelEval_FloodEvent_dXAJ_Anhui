import json
from pathlib import Path

import numpy as np

from hydromodels_dlm.dataset.normalization import NORMALIZATION_REGISTRY


# ---------------------------------------------------------------------------
# Method resolution
# ---------------------------------------------------------------------------


def get_normalization(method):
    spec = NORMALIZATION_REGISTRY.get(method)
    if spec is None:
        raise ValueError(
            f'unknown normalization method {method!r}; '
            f'choose from: {sorted(NORMALIZATION_REGISTRY)}'
        )
    return spec


def resolve_feature_methods(feature_names, scaler_params, *, strict=False):
    if not isinstance(scaler_params, dict):
        raise ValueError('scaler_params must be a dict')

    methods = {}
    assigned = {}
    undefined = []

    for method in scaler_params:
        for var in scaler_params[method]:
            if var not in feature_names:
                if strict:
                    undefined.append((method, var))
                continue
            if var in assigned:
                raise ValueError(
                    f'variable {var!r} appears in both {assigned[var]!r} and {method!r}'
                )
            assigned[var] = method
            methods[var] = method

    if undefined:
        detail = ', '.join(f'{var} ({method})' for method, var in undefined)
        raise ValueError(
            f'scaler_params lists undefined variables: {detail}; '
            f'available: {feature_names}'
        )

    missing = [name for name in feature_names if name not in methods]
    for name in missing:
        methods[name] = 'zscore'

    return [methods[name] for name in feature_names]


def validate_scaler_params(scaler_params, feature_names):
    if not isinstance(scaler_params, dict):
        raise ValueError('data_cfgs.scaler_params must be a dict')

    unknown = set(scaler_params).difference(NORMALIZATION_REGISTRY)
    if unknown:
        raise ValueError(
            f'unknown scaler_params keys {sorted(unknown)!r}; '
            f'allowed: {sorted(NORMALIZATION_REGISTRY)}'
        )

    for method, names in scaler_params.items():
        if not isinstance(names, list):
            raise ValueError(f'scaler_params[{method!r}] must be a list')

    resolve_feature_methods(feature_names, scaler_params, strict=True)


# ---------------------------------------------------------------------------
# FeatureScaler
# ---------------------------------------------------------------------------


class FeatureScaler:

    def __init__(
        self,
        feature_names,
        methods,
        mean,
        std,
        min_vals,
        max_vals,
        prcp_scale=None,
        prcp_scale_meta=None,
    ):
        self.feature_names = list(feature_names)
        self.methods = list(methods)
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.asarray(std, dtype=np.float64)
        self.min_vals = np.asarray(min_vals, dtype=np.float64)
        self.max_vals = np.asarray(max_vals, dtype=np.float64)
        self.prcp_scale = (
            None if prcp_scale is None else np.asarray(prcp_scale, dtype=np.float64)
        )
        self.prcp_scale_meta = dict(prcp_scale_meta) if prcp_scale_meta else None

    @classmethod
    def fit(cls, values, feature_names, methods, prcp_scale=None, prcp_scale_meta=None):
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != len(feature_names):
            raise ValueError('values must be 2-D with one column per feature_names entry')

        prcp_arr = None
        if prcp_scale is not None:
            prcp_arr = np.asarray(prcp_scale, dtype=np.float64)
            if prcp_arr.shape != (len(feature_names),):
                raise ValueError('prcp_scale length must match feature_names')

        n_feat = len(feature_names)
        mean = np.zeros(n_feat, dtype=np.float64)
        std = np.ones(n_feat, dtype=np.float64)
        min_vals = np.zeros(n_feat, dtype=np.float64)
        max_vals = np.zeros(n_feat, dtype=np.float64)
        for i, method in enumerate(methods):
            scale = None if prcp_arr is None else float(prcp_arr[i])
            min_val, max_val, mu, sigma = get_normalization(method)['fit'](
                array[:, i],
                prcp_scale=scale,
            )
            min_vals[i] = min_val
            max_vals[i] = max_val
            mean[i] = mu
            std[i] = sigma

        return cls(
            feature_names,
            methods,
            mean,
            std,
            min_vals,
            max_vals,
            prcp_scale=prcp_arr,
            prcp_scale_meta=prcp_scale_meta,
        )

    def prcp_scale_at(self, index):
        if self.prcp_scale is None:
            return None
        return float(self.prcp_scale[index])

    def apply_column(self, index, values, *, inverse=False):
        op = 'denormalize' if inverse else 'normalize'
        spec = get_normalization(self.methods[index])
        return spec[op](
            values,
            self.mean[index],
            self.std[index],
            prcp_scale=self.prcp_scale_at(index),
        )

    def transform(self, x):
        x = np.asarray(x, dtype=np.float64)
        out = np.empty_like(x)
        for i in range(len(self.methods)):
            out[:, i] = self.apply_column(i, x[:, i], inverse=False)
        return out

    def inverse_transform(self, x):
        x = np.asarray(x, dtype=np.float64)
        out = np.empty_like(x)
        for i in range(len(self.methods)):
            out[:, i] = self.apply_column(i, x[:, i], inverse=True)
        return out

    def to_dict(self):
        payload = {
            'feature_names': self.feature_names,
            'methods': self.methods,
            'min': self.min_vals.tolist(),
            'max': self.max_vals.tolist(),
            'mean': self.mean.tolist(),
            'std': self.std.tolist(),
        }
        if self.prcp_scale is not None:
            payload['prcp_scale'] = self.prcp_scale.tolist()
        if self.prcp_scale_meta:
            payload['prcp_scale_meta'] = self.prcp_scale_meta
        return payload

    @classmethod
    def from_dict(cls, payload):
        for key in ('feature_names', 'methods', 'min', 'max', 'mean', 'std'):
            if key not in payload:
                raise KeyError(f'scaler payload missing {key!r}')
        return cls(
            payload['feature_names'],
            payload['methods'],
            payload['mean'],
            payload['std'],
            payload['min'],
            payload['max'],
            prcp_scale=payload.get('prcp_scale'),
            prcp_scale_meta=payload.get('prcp_scale_meta'),
        )


# ---------------------------------------------------------------------------
# BasinScalerHub
# ---------------------------------------------------------------------------


class BasinScalerHub:

    def __init__(self, x_scaler, y_scaler, static_scaler=None):
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.static_scaler = static_scaler

    def transform_static(self, static_vector):
        if self.static_scaler is None:
            return np.asarray(static_vector, dtype=np.float64)
        row = np.asarray(static_vector, dtype=np.float64).reshape(1, -1)
        return self.static_scaler.transform(row)[0]

    def to_dict(self):
        payload = {
            'x': self.x_scaler.to_dict(),
            'y': self.y_scaler.to_dict(),
        }
        if self.static_scaler is not None:
            payload['static'] = self.static_scaler.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload):
        static_scaler = None
        if 'static' in payload:
            static_scaler = FeatureScaler.from_dict(payload['static'])
        return cls(
            FeatureScaler.from_dict(payload['x']),
            FeatureScaler.from_dict(payload['y']),
            static_scaler=static_scaler,
        )


# ---------------------------------------------------------------------------
# Fit and I/O
# ---------------------------------------------------------------------------


def fit_shared_static_scaler(loader, basin_ids, scaler_params):
    keys = loader.static_keys
    if not keys:
        return None
    methods = resolve_feature_methods(keys, scaler_params)
    rows = np.stack([loader.static_vector(bid, keys) for bid in basin_ids], axis=0)
    return FeatureScaler.fit(rows, keys, methods)


def fit_basin_scalers(loader, basin_ids=None, scaler_params=None):
    basin_ids = [str(bid) for bid in (basin_ids or loader.basin_ids)]
    if scaler_params is None:
        scaler_params = loader.config.get('scaler_params')
    if not isinstance(scaler_params, dict):
        raise ValueError('data_cfgs.scaler_params is required')
    x_methods = resolve_feature_methods(loader.input_keys, scaler_params)
    y_methods = resolve_feature_methods([loader.obs_key], scaler_params)

    if 'prcp_log1p_zscore' in y_methods and 'P' not in loader.input_keys:
        raise ValueError('prcp_log1p_zscore on Q requires P in dynamic_inputs')

    static_scaler = fit_shared_static_scaler(loader, basin_ids, scaler_params)
    hubs = {}
    for bid in basin_ids:
        x, y, _ = loader.arrays_for_period(bid, 'train')
        prcp_scale = None
        prcp_meta = None
        if 'prcp_log1p_zscore' in y_methods:
            prcp_scale, prcp_meta = loader.prcp_scale(bid)
            prcp_scale = [prcp_scale]
        y_scaler = FeatureScaler.fit(
            y,
            [loader.obs_key],
            y_methods,
            prcp_scale=prcp_scale,
            prcp_scale_meta=prcp_meta,
        )
        x_scaler = FeatureScaler.fit(x, loader.input_keys, x_methods)
        hubs[bid] = BasinScalerHub(x_scaler, y_scaler, static_scaler=static_scaler)
    return hubs


def save_scalers(path, scalers):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {bid: hub.to_dict() for bid, hub in scalers.items()}
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)


def load_scalers(path):
    with open(path, encoding='utf-8') as handle:
        raw = json.load(handle)
    return {bid: BasinScalerHub.from_dict(hub) for bid, hub in raw.items()}
