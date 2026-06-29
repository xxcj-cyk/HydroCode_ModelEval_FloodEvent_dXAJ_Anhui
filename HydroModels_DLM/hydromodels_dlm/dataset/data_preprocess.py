import numpy as np
import pandas as pd

from hydromodels_dlm.dataset.data_source import format_timestamp
from hydromodels_dlm.utils.logging import get_logger, log_detail, log_section


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
# Series preparation
# ---------------------------------------------------------------------------


def clean_series(series, input_keys, obs_key):
    out = series.copy()
    cols = [c for c in input_keys if c in out.columns]
    if cols:
        out[cols] = out[cols].fillna(0.0)
    if 'PET' in input_keys and 'PET' in out.columns:
        out['PET'] = np.abs(out['PET'].to_numpy(dtype=np.float64))
    return out


def check_series(series, input_keys, name='data', warmup_length=0):
    n_time = len(series)
    usable = n_time - warmup_length
    if usable < 1:
        raise ValueError(
            f'{name}: length {n_time}, warmup_length={warmup_length}, '
            f'only {usable} steps remain after warmup'
        )
    for key in ('P', 'PET'):
        if key in input_keys and key in series.columns:
            min_val = float(series[key].min())
            if min_val < 0:
                raise ValueError(f'{name}: negative {key} values')
    driver_cols = [c for c in input_keys if c in series.columns]
    if driver_cols and series[driver_cols].isna().any().any():
        raise ValueError(f'{name}: missing driver values remain after zero-fill')


# ---------------------------------------------------------------------------
# Missing-value reporting
# ---------------------------------------------------------------------------


def format_missing_stats(sub, columns):
    n_time = len(sub)
    parts = []
    for col in columns:
        if col not in sub.columns:
            continue
        n_nan = int(sub[col].isna().sum())
        ratio_pct = 100.0 * n_nan / n_time if n_time else 0.0
        parts.append(f'{col}={n_nan}/{n_time} ({ratio_pct:.2f}%)')
    return ', '.join(parts)


def format_missing_stats_aggregate(series_list, columns):
    total_rows = 0
    nan_counts = {col: 0 for col in columns}
    for series in series_list:
        n_time = len(series)
        total_rows += n_time
        for col in columns:
            if col not in series.columns:
                continue
            nan_counts[col] += int(series[col].isna().sum())
    if total_rows == 0:
        return ', '.join(f'{col}=0/0 (0.00%)' for col in columns)
    parts = []
    for col in columns:
        n_nan = nan_counts[col]
        ratio_pct = 100.0 * n_nan / total_rows
        parts.append(f'{col}={n_nan}/{total_rows} ({ratio_pct:.2f}%)')
    return ', '.join(parts)


def period_series_for_stats(series, date_range, *, warmup=0):
    if warmup:
        times = series.index
        if len(times) <= warmup:
            raise ValueError(
                f'series length {len(times)} <= warmup={warmup}, '
                'cannot compute post-warmup period stats'
            )
        sim_times = times[warmup:]
        t0, t1 = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        sub = series.loc[sim_times[(sim_times >= t0) & (sim_times <= t1)]]
    else:
        sub = clip_range(series, date_range)
    if sub.empty:
        eff = [str(date_range[0]), str(date_range[1])]
    else:
        eff = [
            format_timestamp(sub.index.min()),
            format_timestamp(sub.index.max()),
        ]
    return sub, eff


def log_driver_quality(series, input_keys, obs_key, log_tag, period_ranges=None):
    log = get_logger('dataset')
    columns = list(input_keys) + [obs_key]
    log_section(log, '▶ Data Quality')
    log_detail(log, '[%s] missing values: %s', log_tag, format_missing_stats(series, columns))
    if not period_ranges:
        return
    for label, spec in period_ranges.items():
        if isinstance(spec, dict):
            date_range = spec['dates']
            warmup = int(spec.get('warmup', 0))
        else:
            date_range = spec
            warmup = 0
        sub, eff = period_series_for_stats(series, date_range, warmup=warmup)
        log_detail(
            log,
            '[%s] %s missing values: %s',
            f'{eff[0]}_{eff[1]}',
            label,
            format_missing_stats(sub, columns),
        )


def log_driver_quality_aggregate(series_list, columns, log_tag, period_buckets=None):
    log = get_logger('dataset')
    log_section(log, '▶ Data Quality')
    log_detail(
        log,
        '[%s] missing values: %s',
        log_tag,
        format_missing_stats_aggregate(series_list, columns),
    )
    if not period_buckets:
        return
    for label, bucket in period_buckets.items():
        log_detail(
            log,
            '[%s] %s missing values: %s',
            bucket['tag'],
            label,
            format_missing_stats_aggregate(bucket['series'], columns),
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
    return key if key in ('train', 'valid', 'test') else raw


def period_dates(data_cfgs, label, q_bounds=None):
    raw = label.strip()
    if ',' in raw:
        period = [p.strip() for p in raw.split(',', 1)]
        return intersect_range(period, q_bounds) if q_bounds else period

    key = raw.lower()
    if key == 'train':
        period = list(data_cfgs['train_period'])
    elif key == 'valid':
        period = list(data_cfgs['valid_period'])
    elif key == 'test':
        period = list(data_cfgs['test_period'])
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
    device=None,
):
    from hydromodels_dlm.utils.runtime_env import log_runtime_environment

    log = get_logger()
    log_section(log, '▶ Run Setup')
    log_runtime_environment(log, device=device)
    if model is not None and algorithm is not None:
        log_detail(
            log,
            'Model: %s  |  Algorithm: %s  |  Basins: %s',
            model,
            algorithm,
            [str(basin_id)],
        )
    path_label = input_path if input_path is not None else root
    if output_dir is not None:
        log_detail(log, 'Input Path: %s  |  Output Path: %s', path_label, output_dir)
    else:
        log_detail(log, 'Input Path: %s', path_label)
    return log


def log_longterm_periods(data_cfgs, log=None):
    log = log or get_logger('dataset')
    log_detail(
        log,
        'Train Period: %s ~ %s',
        data_cfgs['train_period'][0],
        data_cfgs['train_period'][1],
    )
    log_detail(
        log,
        'Valid Period: %s ~ %s',
        data_cfgs['valid_period'][0],
        data_cfgs['valid_period'][1],
    )
    log_detail(
        log,
        'Test Period: %s ~ %s',
        data_cfgs['test_period'][0],
        data_cfgs['test_period'][1],
    )


def log_longterm_load(
    root,
    basin_id,
    data_cfgs,
    *,
    model=None,
    algorithm=None,
    output_dir=None,
    input_path=None,
    device=None,
):
    log = log_run_setup(
        root,
        basin_id,
        model=model,
        algorithm=algorithm,
        output_dir=output_dir,
        input_path=input_path,
        device=device,
    )
    log_longterm_periods(data_cfgs, log=log)


def log_flood_load(
    root,
    basin_id,
    train_event_ids,
    valid_event_ids,
    *,
    test_event_ids=None,
    model=None,
    algorithm=None,
    output_dir=None,
    input_path=None,
    device=None,
):
    log = log_run_setup(
        root,
        basin_id,
        model=model,
        algorithm=algorithm,
        output_dir=output_dir,
        input_path=input_path,
        device=device,
    )
    log_detail(
        log,
        'Train Events: %d [%s]',
        len(train_event_ids),
        ', '.join(train_event_ids),
    )
    log_detail(
        log,
        'Valid Events: %d [%s]',
        len(valid_event_ids),
        ', '.join(valid_event_ids),
    )
    if test_event_ids is not None:
        log_detail(
            log,
            'Test Events: %d [%s]',
            len(test_event_ids),
            ', '.join(test_event_ids),
        )
