"""Metrics CSV layout and regional aggregation (flood-event vs long-term)."""

from pathlib import Path

import numpy as np
import pandas as pd

from hydromodels_pbm.utils.logging import get_logger

BEST_METRICS = 'best_metrics.csv'
EVENT_METRICS = 'event_metrics.csv'

PERIOD_ORDER = ('calib', 'valid')
PERIOD_METRICS_FILES = {
    'calib': 'best_metrics_calib.csv',
    'valid': 'best_metrics_valid.csv',
}

STAT_ORDER = ('MEDIAN', 'MEAN')
SUMMARY_LABELS = ('MEAN', 'MEDIAN')
STAT_RANK = {name: idx for idx, name in enumerate(STAT_ORDER)}
PERIOD_RANK = {name: idx for idx, name in enumerate(PERIOD_ORDER)}

FLOOD_BASIN_META = ('stat', 'period', 'basin_id')
LONGTERM_BASIN_META = ('period', 'basin_id')
FLOOD_PERIOD_META = ('stat', 'basin_id')
LONGTERM_PERIOD_META = ('basin_id',)
EVENT_META = ('period', 'basin_id', 'event_id')


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def sort_period_keys(periods):
    return sorted(
        periods,
        key=lambda name: (PERIOD_RANK.get(str(name).lower(), len(PERIOD_RANK)), name),
    )


def format_metric_value(name, value):
    prec = 2 if str(name).upper() == 'PTE' else 4
    return f'{float(value):.{prec}f}'


def metric_cols(df, meta):
    exclude = set(meta)
    return [col for col in df.columns if col not in exclude]


def format_metrics(df, meta):
    out = df.copy()
    for col in metric_cols(out, meta):
        out[col] = out[col].apply(lambda value: format_metric_value(col, value))
    return out


def save_csv(df, path, meta):
    cols = [col for col in meta if col in df.columns] + metric_cols(df, meta)
    format_metrics(df[cols], meta).to_csv(
        Path(path),
        index=False,
        encoding='utf-8-sig',
    )


def log_rows(period_metrics, basin_id, metrics_for_period):
    rows = {}
    for period in sort_period_keys(period_metrics):
        metrics = metrics_for_period(period_metrics[period])
        rows[period] = {
            'period': period,
            'basin_id': basin_id,
            **{
                name: format_metric_value(name, value)
                for name, value in metrics.items()
            },
        }
    return rows


def read_basin_metrics(exp_dir, basin_id):
    path = Path(exp_dir) / basin_id / BEST_METRICS
    if not path.is_file():
        raise FileNotFoundError(f'missing {path}')
    return pd.read_csv(path)


def is_flood_metrics_format(exp_dir, basin_ids):
    for basin_id in basin_ids:
        path = Path(exp_dir) / basin_id / BEST_METRICS
        if path.is_file() and 'stat' in pd.read_csv(path, nrows=0).columns:
            return True
    return False


# ---------------------------------------------------------------------------
# Flood basin metrics
# ---------------------------------------------------------------------------


def aggregate_event_metrics(event_rows, metric_names):
    if not event_rows:
        raise ValueError('no per-event metrics to aggregate')

    result = {}
    for stat in STAT_ORDER:
        reducer = np.nanmedian if stat == 'MEDIAN' else np.nanmean
        row = {}
        for name in metric_names:
            key = str(name).upper()
            vals = [
                float(item[key])
                for item in event_rows
                if key in item and item[key] == item[key]
            ]
            row[key] = float(reducer(vals)) if vals else np.nan
        result[stat] = row
    return result


def write_basin_metrics(basin_dir, basin_id, period_stat_metrics):
    rows = []
    for period in sort_period_keys(period_stat_metrics):
        for stat in STAT_ORDER:
            metrics = period_stat_metrics[period].get(stat)
            if metrics:
                rows.append(
                    {'stat': stat, 'period': period, 'basin_id': basin_id, **metrics}
                )

    if rows:
        df = pd.DataFrame(rows)
        df = df.assign(
            sort_stat=df['stat'].str.upper().map(lambda v: STAT_RANK.get(v, 99)),
            sort_period=df['period'].str.lower().map(lambda v: PERIOD_RANK.get(v, 99)),
        ).sort_values(['sort_stat', 'sort_period']).drop(columns=['sort_stat', 'sort_period'])
        save_csv(df, Path(basin_dir) / BEST_METRICS, FLOOD_BASIN_META)

    return log_rows(
        period_stat_metrics,
        basin_id,
        lambda period: period.get('MEAN') or {},
    )


# ---------------------------------------------------------------------------
# Long-term basin metrics
# ---------------------------------------------------------------------------


def write_basin_metrics_longterm(basin_dir, basin_id, period_metrics):
    rows = [
        {'period': period, 'basin_id': basin_id, **period_metrics[period]}
        for period in sort_period_keys(period_metrics)
    ]

    if rows:
        df = pd.DataFrame(rows)
        df = df.assign(
            sort_period=df['period'].str.lower().map(lambda v: PERIOD_RANK.get(v, 99)),
        ).sort_values('sort_period').drop(columns=['sort_period'])
        save_csv(df, Path(basin_dir) / BEST_METRICS, LONGTERM_BASIN_META)

    return log_rows(period_metrics, basin_id, lambda period: period)


# ---------------------------------------------------------------------------
# Flood experiment summary
# ---------------------------------------------------------------------------


def load_period_event_metrics(exp_dir, basin_ids, period):
    frames = []
    target = str(period).lower()
    for basin_id in basin_ids:
        path = Path(exp_dir) / basin_id / EVENT_METRICS
        if not path.is_file():
            continue
        chunk = pd.read_csv(path)
        chunk = chunk.loc[chunk['period'].astype(str).str.lower() == target]
        if not chunk.empty:
            frames.append(chunk)
    return pd.concat(frames, ignore_index=True) if frames else None


def summarize_flood_regional(exp_dir, basin_ids, period):
    events = load_period_event_metrics(exp_dir, basin_ids, period)
    if events is None:
        return None

    rows = []
    for stat in STAT_ORDER:
        reducer = np.nanmedian if stat == 'MEDIAN' else np.nanmean
        row = {'stat': stat, 'basin_id': ''}
        for col in metric_cols(events, EVENT_META):
            vals = pd.to_numeric(events[col], errors='coerce')
            row[col] = format_metric_value(col, float(reducer(vals)))
        rows.append(row)
    return rows


def group_flood_rows_by_period(exp_dir, basin_ids):
    rows_by_period = {}
    for basin_id in sorted(str(b) for b in basin_ids):
        for row in read_basin_metrics(exp_dir, basin_id).to_dict('records'):
            stat = str(row['stat']).upper()
            if stat not in STAT_ORDER:
                continue
            period = str(row['period']).lower()
            record = {
                key: value for key, value in row.items() if key not in ('period', 'stat')
            }
            record['stat'] = stat
            rows_by_period.setdefault(period, []).append(record)
    return rows_by_period


def save_flood_period_csv(path, basin_rows, regional_rows):
    df = pd.DataFrame(basin_rows)
    df = df.assign(
        sort_stat=df['stat'].str.upper().map(lambda v: STAT_RANK.get(v, 99)),
        sort_basin=df['basin_id'].astype(str),
    ).sort_values(['sort_stat', 'sort_basin']).drop(columns=['sort_stat', 'sort_basin'])
    df = format_metrics(df, FLOOD_PERIOD_META)

    regional = pd.DataFrame(regional_rows)
    parts = []
    for stat in STAT_ORDER:
        chunk = df.loc[df['stat'].str.upper() == stat]
        if not chunk.empty:
            parts.append(chunk)
        summary = regional.loc[regional['stat'].str.upper() == stat]
        if not summary.empty:
            parts.append(summary)
    out = pd.concat(parts, ignore_index=True)
    cols = [col for col in FLOOD_PERIOD_META if col in out.columns] + metric_cols(
        out, FLOOD_PERIOD_META
    )
    out[cols].to_csv(
        Path(path),
        index=False,
        encoding='utf-8-sig',
    )


def publish_flood_period_metrics(exp_dir, basin_ids):
    for period, basin_rows in group_flood_rows_by_period(exp_dir, basin_ids).items():
        filename = PERIOD_METRICS_FILES.get(period)
        if not filename or not basin_rows:
            continue
        regional_rows = summarize_flood_regional(exp_dir, basin_ids, period)
        if regional_rows is None:
            raise ValueError(f'missing event_metrics for flood period {period!r}')
        save_flood_period_csv(exp_dir / filename, basin_rows, regional_rows)


# ---------------------------------------------------------------------------
# Long-term experiment summary
# ---------------------------------------------------------------------------


def group_longterm_rows_by_period(exp_dir, basin_ids):
    rows_by_period = {}
    for basin_id in sorted(str(b) for b in basin_ids):
        for row in read_basin_metrics(exp_dir, basin_id).to_dict('records'):
            period = str(row['period']).lower()
            record = {key: value for key, value in row.items() if key != 'period'}
            rows_by_period.setdefault(period, []).append(record)
    return rows_by_period


def append_basin_equal_summary(df):
    basin_rows = df.loc[~df['basin_id'].astype(str).isin(SUMMARY_LABELS)].copy()
    summary = []
    for label, reducer in (('MEAN', 'mean'), ('MEDIAN', 'median')):
        row = {'basin_id': label}
        for col in metric_cols(basin_rows, LONGTERM_PERIOD_META):
            vals = pd.to_numeric(basin_rows[col], errors='coerce')
            row[col] = format_metric_value(col, float(getattr(vals, reducer)()))
        summary.append(row)
    out = pd.concat([basin_rows, pd.DataFrame(summary)], ignore_index=True)
    return format_metrics(out, LONGTERM_PERIOD_META)


def publish_longterm_period_metrics(exp_dir, basin_ids):
    for period, basin_rows in group_longterm_rows_by_period(exp_dir, basin_ids).items():
        filename = PERIOD_METRICS_FILES.get(period)
        if not filename or not basin_rows:
            continue
        df = append_basin_equal_summary(pd.DataFrame(basin_rows).sort_values('basin_id'))
        df.to_csv(exp_dir / filename, index=False, encoding='utf-8-sig')


def publish_experiment_period_metrics(exp_dir, basin_ids):
    log = get_logger('artifacts')
    exp_dir = Path(exp_dir)
    basin_ids = sorted(str(b) for b in basin_ids)
    if is_flood_metrics_format(exp_dir, basin_ids):
        publish_flood_period_metrics(exp_dir, basin_ids)
    else:
        publish_longterm_period_metrics(exp_dir, basin_ids)
    log.info(
        'metrics summary published: %s/{%s}',
        exp_dir,
        ', '.join(PERIOD_METRICS_FILES.values()),
    )


# ---------------------------------------------------------------------------
# Regenerate flood metrics from event_metrics.csv
# ---------------------------------------------------------------------------


def regenerate_basin_metrics(basin_dir, basin_id):
    path = Path(basin_dir) / EVENT_METRICS
    if not path.is_file():
        raise FileNotFoundError(f'missing {path}')

    df = pd.read_csv(path)
    metric_names = metric_cols(df, EVENT_META)
    period_stat_metrics = {}
    for period, group in df.groupby('period', sort=False):
        event_rows = [
            {name: float(pd.to_numeric(row[name], errors='coerce')) for name in metric_names}
            for row in group.to_dict('records')
        ]
        period_stat_metrics[str(period).lower()] = aggregate_event_metrics(
            event_rows,
            metric_names,
        )
    write_basin_metrics(basin_dir, str(basin_id), period_stat_metrics)
    return period_stat_metrics


def regenerate_experiment_metrics(exp_dir, basin_ids):
    exp_dir = Path(exp_dir)
    for basin_id in sorted(str(b) for b in basin_ids):
        regenerate_basin_metrics(exp_dir / basin_id, basin_id)
    publish_experiment_period_metrics(exp_dir, basin_ids)
