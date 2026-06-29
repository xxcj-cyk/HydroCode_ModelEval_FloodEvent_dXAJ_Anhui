import numpy as np


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def safe_std(std):
    value = float(std)
    return max(value, 1e-6) if np.isfinite(value) else 1e-6


def flatten_valid(values):
    arr = np.asarray(values, dtype=np.float64).ravel()
    return arr[~np.isnan(arr)]


def fit_stats(work):
    if work.size == 0:
        raise ValueError('cannot fit normalization on empty array')
    return (
        float(np.min(work)),
        float(np.max(work)),
        float(np.mean(work)),
        safe_std(np.std(work)),
    )


# ---------------------------------------------------------------------------
# zscore
# ---------------------------------------------------------------------------


def fit_zscore(values, prcp_scale=None):
    return fit_stats(flatten_valid(values))


def normalize_zscore(values, mean, std, prcp_scale=None):
    arr = np.asarray(values, dtype=np.float64)
    return (arr - mean) / safe_std(std)


def denormalize_zscore(values, mean, std, prcp_scale=None):
    arr = np.asarray(values, dtype=np.float64)
    return arr * safe_std(std) + mean


# ---------------------------------------------------------------------------
# log1p_zscore
# ---------------------------------------------------------------------------


def fit_log1p_zscore(values, prcp_scale=None):
    data = flatten_valid(values)
    if np.any(data < 0):
        raise ValueError('log1p_zscore requires non-negative values')
    return fit_stats(np.log1p(data))


def normalize_log1p_zscore(values, mean, std, prcp_scale=None):
    arr = np.asarray(values, dtype=np.float64)
    if np.any(arr < 0):
        raise ValueError('log1p_zscore requires non-negative values')
    return (np.log1p(arr) - mean) / safe_std(std)


def denormalize_log1p_zscore(values, mean, std, prcp_scale=None):
    arr = np.asarray(values, dtype=np.float64)
    work = arr * safe_std(std) + mean
    return np.expm1(work)


# ---------------------------------------------------------------------------
# prcp_log1p_zscore
# ---------------------------------------------------------------------------


def fit_prcp_log1p_zscore(values, prcp_scale=None):
    if prcp_scale is None or prcp_scale <= 0:
        raise ValueError('prcp_log1p_zscore requires positive prcp_scale')
    data = flatten_valid(values) / prcp_scale
    if np.any(data < 0):
        raise ValueError('prcp_log1p_zscore requires non-negative Q / prcp_scale')
    return fit_stats(np.log1p(data))


def normalize_prcp_log1p_zscore(values, mean, std, prcp_scale=None):
    if prcp_scale is None or prcp_scale <= 0:
        raise ValueError('prcp_log1p_zscore requires positive prcp_scale')
    arr = np.asarray(values, dtype=np.float64)
    scaled = arr / prcp_scale
    if np.any(scaled < 0):
        raise ValueError('prcp_log1p_zscore requires non-negative Q / prcp_scale')
    return (np.log1p(scaled) - mean) / safe_std(std)


def denormalize_prcp_log1p_zscore(values, mean, std, prcp_scale=None):
    if prcp_scale is None or prcp_scale <= 0:
        raise ValueError('prcp_log1p_zscore requires positive prcp_scale')
    arr = np.asarray(values, dtype=np.float64)
    work = arr * safe_std(std) + mean
    return np.expm1(work) * prcp_scale


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


NORMALIZATION_REGISTRY = {
    'zscore': {
        'fit': fit_zscore,
        'normalize': normalize_zscore,
        'denormalize': denormalize_zscore,
    },
    'log1p_zscore': {
        'fit': fit_log1p_zscore,
        'normalize': normalize_log1p_zscore,
        'denormalize': denormalize_log1p_zscore,
    },
    'prcp_log1p_zscore': {
        'fit': fit_prcp_log1p_zscore,
        'normalize': normalize_prcp_log1p_zscore,
        'denormalize': denormalize_prcp_log1p_zscore,
    },
}
