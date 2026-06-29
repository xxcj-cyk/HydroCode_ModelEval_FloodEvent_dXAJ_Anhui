import numpy as np
import pandas as pd

from hydromodels_pbm.dataset.data_source import format_timestamp
from hydromodels_pbm.utils.logging import get_logger, log_section
from hydromodels_pbm.utils.runtime_env import log_runtime_environment


# ---------------------------------------------------------------------------
# Time series
# ---------------------------------------------------------------------------


def clip_range(df, date_range):
    start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    return df.loc[start:end]


def observation_date_range(series, output_key='Q'):
    if output_key not in series.columns:
        raise ValueError(f'series missing column {output_key!r}')
    valid = series.index[~series[output_key].isna()]
    if len(valid) == 0:
        raise ValueError(f'no valid {output_key} observations')
    return format_timestamp(valid.min()), format_timestamp(valid.max())


def intersect_range(period, bounds):
    t0, t1 = pd.Timestamp(period[0]), pd.Timestamp(period[1])
    b0, b1 = pd.Timestamp(bounds[0]), pd.Timestamp(bounds[1])
    start, end = max(t0, b0), min(t1, b1)
    if start > end:
        raise ValueError(f'period {period} does not overlap streamflow bounds {bounds}')
    return [format_timestamp(start), format_timestamp(end)]


# ---------------------------------------------------------------------------
# Driver arrays
# ---------------------------------------------------------------------------


def build_driver_arrays(series, input_keys, obs_key):
    out = series.copy()
    cols = [c for c in input_keys if c in out.columns]
    if cols:
        out[cols] = out[cols].fillna(0.0)
    n_time = len(out)
    n_in = len(input_keys)
    p_and_e = np.empty((n_time, 1, n_in), dtype=np.float64)
    for i, key in enumerate(input_keys):
        p_and_e[:, 0, i] = out[key].values
    if 'PET' in input_keys:
        pet_idx = input_keys.index('PET')
        p_and_e[:, :, pet_idx] = np.abs(p_and_e[:, :, pet_idx])
    qobs = out[obs_key].values.reshape(n_time, 1, 1)
    return p_and_e, qobs


def check_arrays(p_and_e, qobs, input_keys, name='data', warmup_length=0):
    if p_and_e.ndim != 3 or qobs.ndim != 3:
        raise ValueError(
            f'{name}: expected shape [time, basin, feature], '
            f'got p_and_e{p_and_e.shape}, qobs{qobs.shape}'
        )
    n_time = p_and_e.shape[0]
    usable = n_time - warmup_length
    if usable < 1:
        raise ValueError(
            f'{name}: length {n_time}, warmup_length={warmup_length}, '
            f'only {usable} steps remain after warmup'
        )
    key_to_idx = {key: idx for idx, key in enumerate(input_keys)}
    if 'P' in key_to_idx and np.nanmin(p_and_e[:, :, key_to_idx['P']]) < 0:
        raise ValueError(f'{name}: negative precipitation values')
    if 'PET' in key_to_idx and np.nanmin(p_and_e[:, :, key_to_idx['PET']]) < 0:
        raise ValueError(f'{name}: negative PET values')
    if np.isnan(p_and_e).any():
        raise ValueError(f'{name}: missing driver values remain after zero-fill')


def missing_value_stats(sub, columns):
    n_time = len(sub)
    parts = []
    for col in columns:
        if col not in sub.columns:
            continue
        n_nan = int(sub[col].isna().sum())
        ratio_pct = 100.0 * n_nan / n_time if n_time else 0.0
        parts.append(f'{col}={n_nan}/{n_time} ({ratio_pct:.2f}%)')
    return ', '.join(parts)


def quality_period_series(series, date_range, warmup):
    if not warmup:
        return clip_range(series, date_range)
    times = series.index
    sim_times = times[warmup:]
    t0, t1 = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    return series.loc[sim_times[(sim_times >= t0) & (sim_times <= t1)]]


def log_driver_quality(series, input_keys, obs_key, log_tag, period_ranges=None):
    log = get_logger('dataset')
    columns = list(input_keys) + [obs_key]
    log_section(log, '▶ Data Quality')
    log.info('  [%s] missing values: %s', log_tag, missing_value_stats(series, columns))
    if not period_ranges:
        return
    for label, spec in period_ranges.items():
        if isinstance(spec, dict):
            date_range = spec['dates']
            warmup = int(spec.get('warmup', 0))
        else:
            date_range = spec
            warmup = 0
        sub = quality_period_series(series, date_range, warmup)
        if sub.empty:
            eff = [str(date_range[0]), str(date_range[1])]
        else:
            eff = [
                format_timestamp(sub.index.min()),
                format_timestamp(sub.index.max()),
            ]
        log.info(
            '  [%s] %s missing values: %s',
            f'{eff[0]}_{eff[1]}',
            label,
            missing_value_stats(sub, columns),
        )


# ---------------------------------------------------------------------------
# Step indices
# ---------------------------------------------------------------------------


def period_step_indices(times, warmup, date_range):
    sim_times = times[warmup:]
    t0, t1 = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    idx = np.flatnonzero((sim_times >= t0) & (sim_times <= t1))
    if idx.size == 0:
        raise ValueError(f'period {date_range} has no steps after warmup={warmup}')
    return idx


def eval_step_indices(times, warmup, date_range, series):
    period_idx = period_step_indices(times, warmup, date_range)
    if 'flood_event' not in series.columns:
        raise KeyError("series missing flood flag column 'flood_event'")
    event_mask = series['flood_event'].fillna(0).astype(int).to_numpy(dtype=bool)
    post_warmup = event_mask[warmup:]
    if len(post_warmup) != len(times) - warmup:
        raise ValueError('flood_event length does not match time axis after warmup')
    idx = period_idx[post_warmup[period_idx]]
    if idx.size == 0:
        raise ValueError(
            f'period {date_range} has no flood_event steps after warmup={warmup}'
        )
    return idx


# ---------------------------------------------------------------------------
# Period dates
# ---------------------------------------------------------------------------


def parse_period(period):
    raw = period.strip()
    if ',' in raw:
        return raw
    key = raw.lower()
    return key if key in ('calib', 'valid') else raw


def period_dates(data_cfgs, label, q_bounds=None):
    raw = label.strip()
    if ',' in raw:
        period = [p.strip() for p in raw.split(',', 1)]
        return intersect_range(period, q_bounds) if q_bounds else period

    key = raw.lower()
    if key == 'calib':
        period = list(data_cfgs['calib_period'])
    elif key == 'valid':
        period = list(data_cfgs['valid_period'])
    else:
        raise ValueError(f'unknown period {label!r}')
    return intersect_range(period, q_bounds) if q_bounds else period


# ---------------------------------------------------------------------------
# Load logging
# ---------------------------------------------------------------------------


def log_run_setup(
    root,
    basin_id,
    *,
    model=None,
    algorithm=None,
    output_dir=None,
    input_path=None,
):
    log = get_logger('dataset')
    log_section(log, '▶ Run Setup')
    log_runtime_environment(log)
    if model is not None and algorithm is not None:
        log.info(
            '  Model: %s  |  Algorithm: %s  |  Basins: %s',
            model,
            algorithm,
            [str(basin_id)],
        )
    path_label = input_path if input_path is not None else root
    if output_dir is not None:
        log.info('  Input Path: %s  |  Output Path: %s', path_label, output_dir)
    else:
        log.info('  Input Path: %s', path_label)


def log_longterm_load(
    root,
    basin_id,
    data_cfgs,
    *,
    model=None,
    algorithm=None,
    output_dir=None,
    input_path=None,
):
    log_run_setup(
        root,
        basin_id,
        model=model,
        algorithm=algorithm,
        output_dir=output_dir,
        input_path=input_path,
    )
    log = get_logger('dataset')
    log.info(
        '  Calib Period: %s ~ %s',
        data_cfgs['calib_period'][0],
        data_cfgs['calib_period'][1],
    )
    log.info(
        '  Valid Period: %s ~ %s',
        data_cfgs['valid_period'][0],
        data_cfgs['valid_period'][1],
    )


def log_flood_load(
    root,
    basin_id,
    calib_event_ids,
    valid_event_ids,
    *,
    model=None,
    algorithm=None,
    output_dir=None,
    input_path=None,
):
    log_run_setup(
        root,
        basin_id,
        model=model,
        algorithm=algorithm,
        output_dir=output_dir,
        input_path=input_path,
    )
    log = get_logger('dataset')
    log.info(
        '  Calib Events: %d [%s]',
        len(calib_event_ids),
        ', '.join(calib_event_ids),
    )
    log.info(
        '  Valid Events: %d [%s]',
        len(valid_event_ids),
        ', '.join(valid_event_ids),
    )
