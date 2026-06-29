import numpy as np


# ---------------------------------------------------------------------------
# Valid samples
# ---------------------------------------------------------------------------


def valid_obs_sim(observed, simulated, min_samples=1):
    obs = np.asarray(observed, dtype=np.float64).ravel()
    sim = np.asarray(simulated, dtype=np.float64).ravel()
    if obs.shape != sim.shape:
        raise ValueError(
            f'observed and simulated length mismatch: {obs.shape} vs {sim.shape}'
        )
    mask = ~(np.isnan(obs) | np.isnan(sim))
    obs_v = obs[mask]
    sim_v = sim[mask]
    if obs_v.size < min_samples:
        return None, None
    return obs_v, sim_v


def safe_ratio(numerator, denominator):
    if denominator == 0 or not np.isfinite(denominator):
        return np.nan
    value = numerator / denominator
    return float(value) if np.isfinite(value) else np.nan


# ---------------------------------------------------------------------------
# Correlation and efficiency
# ---------------------------------------------------------------------------


def nash_sutcliffe_efficiency(simulated, observed):
    obs_v, sim_v = valid_obs_sim(observed, simulated, min_samples=2)
    if obs_v is None:
        return np.nan
    denom = np.sum((obs_v - np.mean(obs_v)) ** 2)
    if denom == 0:
        return np.nan
    return float(1.0 - np.sum((sim_v - obs_v) ** 2) / denom)


def kling_gupta_efficiency(simulated, observed):
    obs_v, sim_v = valid_obs_sim(observed, simulated, min_samples=2)
    if obs_v is None:
        return np.nan
    std_ratio = safe_ratio(np.std(sim_v), np.std(obs_v))
    mean_ratio = safe_ratio(np.mean(sim_v), np.mean(obs_v))
    corr_coef = np.corrcoef(obs_v, sim_v)[0, 1]
    if np.isnan(std_ratio) or np.isnan(mean_ratio) or np.isnan(corr_coef):
        return np.nan
    return float(
        1.0
        - np.sqrt(
            (corr_coef - 1) ** 2
            + (std_ratio - 1) ** 2
            + (mean_ratio - 1) ** 2
        )
    )


def pearson_correlation(simulated, observed):
    obs_v, sim_v = valid_obs_sim(observed, simulated, min_samples=2)
    if obs_v is None:
        return np.nan
    if np.std(obs_v) <= 1e-12 or np.std(sim_v) <= 1e-12:
        return np.nan
    corr = np.corrcoef(sim_v, obs_v)[0, 1]
    return float(corr) if np.isfinite(corr) else np.nan


def coefficient_of_determination(simulated, observed):
    corr = pearson_correlation(simulated, observed)
    return np.nan if np.isnan(corr) else float(corr * corr)


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------


def root_mean_squared_error(simulated, observed):
    obs_v, sim_v = valid_obs_sim(observed, simulated, min_samples=1)
    if obs_v is None:
        return np.nan
    return float(np.sqrt(np.mean((sim_v - obs_v) ** 2)))


def mean_bias_error(simulated, observed):
    obs_v, sim_v = valid_obs_sim(observed, simulated, min_samples=1)
    if obs_v is None:
        return np.nan
    return float(np.mean(sim_v - obs_v))


def high_flow_root_mean_squared_error(simulated, observed, high_ratio=0.8):
    obs_v, sim_v = valid_obs_sim(observed, simulated, min_samples=1)
    if obs_v is None:
        return np.nan
    peak_observed = np.max(obs_v)
    if peak_observed <= 0 or not np.isfinite(peak_observed):
        return np.nan
    high_flow_mask = obs_v >= peak_observed * high_ratio
    if not np.any(high_flow_mask):
        return np.nan
    mean_squared_diff = np.mean((sim_v[high_flow_mask] - obs_v[high_flow_mask]) ** 2)
    if not np.isfinite(mean_squared_diff) or mean_squared_diff < 0:
        return np.nan
    return float(np.sqrt(mean_squared_diff))


# ---------------------------------------------------------------------------
# Peak and flow volume
# ---------------------------------------------------------------------------


def peak_flow_error(simulated, observed):
    obs_v, sim_v = valid_obs_sim(observed, simulated, min_samples=1)
    if obs_v is None:
        return np.nan
    ratio = safe_ratio(np.max(sim_v) - np.max(obs_v), np.max(obs_v))
    return np.nan if np.isnan(ratio) else float(ratio * 100)


def peak_timing_error(simulated, observed, timestep=1.0):
    obs_v, sim_v = valid_obs_sim(observed, simulated, min_samples=1)
    if obs_v is None:
        return np.nan
    return float((np.argmax(sim_v) - np.argmax(obs_v)) * timestep)


def flow_high_volume_bias(simulated, observed):
    obs_v, sim_v = valid_obs_sim(observed, simulated, min_samples=5)
    if obs_v is None:
        return np.nan
    sim_sorted = np.sort(sim_v)
    obs_sorted = np.sort(obs_v)
    start = min(len(sim_sorted) - 1, int(0.98 * len(sim_sorted)))
    numerator = np.sum(sim_sorted[start:] - obs_sorted[start:])
    denominator = np.sum(obs_sorted[start:])
    ratio = safe_ratio(numerator, denominator)
    return np.nan if np.isnan(ratio) else float(ratio * 100)


def flow_low_volume_bias(simulated, observed):
    obs_v, sim_v = valid_obs_sim(observed, simulated, min_samples=5)
    if obs_v is None:
        return np.nan
    sim_sorted = np.sort(sim_v)
    obs_sorted = np.sort(obs_v)
    stop = max(1, int(0.3 * len(sim_sorted)))
    numerator = np.sum(sim_sorted[:stop] - obs_sorted[:stop])
    denominator = np.sum(obs_sorted[:stop])
    ratio = safe_ratio(numerator, denominator)
    return np.nan if np.isnan(ratio) else float(ratio * 100)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


METRIC_REGISTRY = {
    'NSE': nash_sutcliffe_efficiency,
    'KGE': kling_gupta_efficiency,
    'CORR': pearson_correlation,
    'R2': coefficient_of_determination,
    'RMSE': root_mean_squared_error,
    'MBE': mean_bias_error,
    'HIGHRMSE': high_flow_root_mean_squared_error,
    'PFE': peak_flow_error,
    'PTE': peak_timing_error,
    'FHV': flow_high_volume_bias,
    'FLV': flow_low_volume_bias,
}
