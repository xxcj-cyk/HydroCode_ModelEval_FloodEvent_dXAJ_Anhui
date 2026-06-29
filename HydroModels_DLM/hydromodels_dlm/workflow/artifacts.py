"""Metrics CSV layout and regional aggregation (flood-event vs long-term)."""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from hydromodels_dlm.utils.logging import LOG_INDENT, format_elapsed_hms, get_logger

BEST_CHECKPOINT = 'best_model.pth'
BEST_METRICS = 'best_metrics.csv'
BEST_PARAMS = 'best_params.json'
BEST_PARAMS_ALL = 'best_params_all.json'
NORMALIZATION_SCALER = 'normalization_scaler.json'
TRAINING_LOG = 'training_log.txt'
EVENT_METRICS = 'event_metrics.csv'

PERIOD_ORDER = ('train', 'valid', 'test')
PERIOD_METRICS_FILES = {
    'train': 'best_metrics_train.csv',
    'valid': 'best_metrics_valid.csv',
    'test': 'best_metrics_test.csv',
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


def metric_values_close(a, b, *, atol=1e-4):
    a = float(pd.to_numeric(a, errors='coerce'))
    b = float(pd.to_numeric(b, errors='coerce'))
    if np.isnan(a) and np.isnan(b):
        return True
    return bool(np.isclose(a, b, rtol=0, atol=atol))


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


def clear_experiment_metrics_summary(exp_dir):
    exp_dir = Path(exp_dir)
    for filename in (*PERIOD_METRICS_FILES.values(), BEST_METRICS):
        path = exp_dir / filename
        if path.is_file():
            path.unlink()


def prune_stale_basin_dirs(exp_dir, basin_ids):
    exp_dir = Path(exp_dir)
    keep = {str(b) for b in basin_ids}
    if not exp_dir.is_dir():
        return
    for child in exp_dir.iterdir():
        if not child.is_dir() or child.name in keep:
            continue
        marker = child / BEST_METRICS
        if marker.is_file() or (child / 'timeseries.csv').is_file():
            shutil.rmtree(child)


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


def read_basin_period_row(metrics_path, period, *, stat='MEAN'):
    df = pd.read_csv(metrics_path)
    if 'period' not in df.columns:
        raise ValueError(f'{metrics_path}: missing period column')
    match = df.loc[df['period'].astype(str).str.lower() == str(period).lower()]
    if 'stat' in df.columns:
        match = match.loc[match['stat'].astype(str).str.upper() == str(stat).upper()]
    elif str(stat).upper() != 'MEAN':
        return None
    if match.empty:
        return None
    row = match.iloc[0].to_dict()
    row.pop('period', None)
    row.pop('stat', None)
    return row


def read_basin_period_rows(period_csv, basin_id):
    df = pd.read_csv(period_csv)
    chunk = df.loc[
        (df['stat'].astype(str).str.upper().isin(STAT_ORDER))
        & (df['basin_id'].astype(str) == basin_id)
    ]
    return chunk.to_dict('records')


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

    return {
        period: period_stat_metrics[period].get('MEAN') or {}
        for period in sort_period_keys(period_stat_metrics)
    }


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

    return period_metrics


# ---------------------------------------------------------------------------
# Flood regional summary
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


def summarize_event_metrics(events):
    if events is None or events.empty:
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


def summarize_flood_regional(exp_dir, basin_ids, period):
    return summarize_event_metrics(load_period_event_metrics(exp_dir, basin_ids, period))


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
    out[cols].to_csv(Path(path), index=False, encoding='utf-8-sig')


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
# Multi-target ensemble
# ---------------------------------------------------------------------------


def load_target_dirs_from_summary(summary_csv, project_root):
    summary_csv = Path(summary_csv)
    if not summary_csv.is_file():
        return {}

    project_root = Path(project_root)
    test_file = PERIOD_METRICS_FILES['test']
    summary = pd.read_csv(summary_csv)
    if 'target_basin' not in summary.columns:
        raise KeyError('summary CSV missing target_basin')

    entries = {}
    for _, row in summary.iterrows():
        target = str(row['target_basin'])
        rel = Path(str(row['ensemble_dir']))
        path = rel if rel.is_absolute() else project_root / rel
        if path.is_dir() and (path / test_file).is_file():
            entries[target] = path
    return entries


def is_single_basin_ensemble(summary_csv):
    summary_csv = Path(summary_csv)
    if not summary_csv.is_file():
        return True
    values = pd.read_csv(summary_csv)['n_basins'].unique()
    return len(values) == 1 and values[0] == 1


def list_period_summary_basin_ids(target_dirs, period):
    filename = PERIOD_METRICS_FILES.get(period)
    if not filename:
        return []

    basin_ids = set()
    for subdir in target_dirs.values():
        period_csv = subdir / filename
        if not period_csv.is_file():
            continue
        df = pd.read_csv(period_csv)
        ids = df.loc[
            df['basin_id'].astype(str).str.startswith('Anhui_'),
            'basin_id',
        ]
        basin_ids.update(ids.astype(str))
    return sorted(basin_ids)


def loo_source_target(target_basins, basin_id):
    return next((name for name in target_basins if name != basin_id), None)


def load_period_events_from_path(path, period):
    if not path.is_file():
        return None
    chunk = pd.read_csv(path)
    target = str(period).lower()
    chunk = chunk.loc[chunk['period'].astype(str).str.lower() == target]
    return chunk if not chunk.empty else None


def collect_multi_target_basin_rows(target_dirs, target_basins, period, single_basin_target):
    filename = PERIOD_METRICS_FILES[period]
    basin_rows = []

    if single_basin_target:
        for target_basin in target_basins:
            period_csv = target_dirs[target_basin] / filename
            if period_csv.is_file():
                basin_rows.extend(read_basin_period_rows(period_csv, target_basin))
        return basin_rows

    for basin_id in list_period_summary_basin_ids(target_dirs, period):
        source_target = loo_source_target(target_basins, basin_id)
        if source_target is None:
            continue
        period_csv = target_dirs[source_target] / filename
        if period_csv.is_file():
            basin_rows.extend(read_basin_period_rows(period_csv, basin_id))
    return basin_rows


def load_multi_target_period_events(target_dirs, target_basins, period, single_basin_target):
    frames = []

    if single_basin_target:
        for target_basin in target_basins:
            path = target_dirs[target_basin] / target_basin / EVENT_METRICS
            chunk = load_period_events_from_path(path, period)
            if chunk is not None:
                frames.append(chunk)
    else:
        for basin_id in list_period_summary_basin_ids(target_dirs, period):
            source_target = loo_source_target(target_basins, basin_id)
            if source_target is None:
                continue
            path = target_dirs[source_target] / basin_id / EVENT_METRICS
            chunk = load_period_events_from_path(path, period)
            if chunk is not None:
                frames.append(chunk)

    return pd.concat(frames, ignore_index=True) if frames else None


def publish_multi_target_ensemble_metrics(
    ensemble_root,
    target_dirs,
    *,
    single_basin_target=True,
):
    ensemble_root = Path(ensemble_root)
    if not target_dirs:
        return

    target_basins = sorted(str(name) for name in target_dirs)
    target_dirs = {str(key): Path(path) for key, path in target_dirs.items()}

    for period in PERIOD_ORDER:
        filename = PERIOD_METRICS_FILES.get(period)
        if not filename:
            continue

        basin_rows = collect_multi_target_basin_rows(
            target_dirs,
            target_basins,
            period,
            single_basin_target,
        )
        if not basin_rows:
            continue

        events = load_multi_target_period_events(
            target_dirs,
            target_basins,
            period,
            single_basin_target,
        )
        regional_rows = summarize_event_metrics(events)
        if regional_rows is None:
            raise ValueError(
                f'missing pooled event_metrics for multi-target ensemble period {period!r}'
            )
        save_flood_period_csv(ensemble_root / filename, basin_rows, regional_rows)


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
        df = append_basin_equal_summary(
            pd.DataFrame(basin_rows).sort_values('basin_id')
        )
        df.to_csv(exp_dir / filename, index=False, encoding='utf-8-sig')


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------


def is_regional_summary_row(row):
    stat = row.get('stat', '')
    if stat is None or (isinstance(stat, float) and np.isnan(stat)):
        stat = ''
    stat = str(stat).strip().upper()
    bid = row.get('basin_id', '')
    if bid is None or (isinstance(bid, float) and np.isnan(bid)):
        bid = ''
    bid = str(bid).strip()
    if stat in STAT_ORDER and bid in ('', 'ALL'):
        return True
    return bid.upper() in SUMMARY_LABELS and stat in ('', 'ALL', 'NAN')


def verify_period_event_summary(exp_dir, basin_ids, period, filename):
    summary_path = Path(exp_dir) / filename
    if not summary_path.is_file():
        return []
    expected = summarize_flood_regional(exp_dir, basin_ids, period)
    if expected is None:
        return []

    summary = pd.read_csv(summary_path)
    regional = summary.loc[summary.apply(is_regional_summary_row, axis=1)]
    if regional.empty:
        regional = summary.loc[summary['basin_id'].isin(SUMMARY_LABELS)]
    metric_names = [
        c for c in summary.columns if c not in ('basin_id', 'period', 'stat')
    ]
    mismatches = []
    for exp_row in expected:
        label = str(exp_row['stat']).upper()
        if 'stat' in regional.columns:
            got = regional.loc[regional['stat'].astype(str).str.upper() == label]
        else:
            got = regional.loc[regional['basin_id'].astype(str).str.upper() == label]
        if got.empty:
            mismatches.append(('REGIONAL', period, label, np.nan, np.nan))
            continue
        got = got.iloc[0]
        for col in metric_names:
            if col not in exp_row or col not in got.index:
                continue
            a = float(pd.to_numeric(exp_row[col], errors='coerce'))
            b = float(pd.to_numeric(got[col], errors='coerce'))
            if not metric_values_close(a, b):
                mismatches.append(('REGIONAL', period, f'{label}/{col}', a, b))
    return mismatches


def verify_period_summary(exp_dir, basin_ids, period, filename, *, flood):
    summary_path = Path(exp_dir) / filename
    if not summary_path.is_file():
        return []
    summary = pd.read_csv(summary_path)
    if flood and 'stat' in summary.columns:
        basin_rows = summary.loc[~summary.apply(is_regional_summary_row, axis=1)]
    else:
        basin_rows = summary.loc[~summary['basin_id'].isin(SUMMARY_LABELS)]
    metric_names = [c for c in basin_rows.columns if c not in ('basin_id', 'stat')]
    stats = list(STAT_ORDER) if flood and 'stat' in basin_rows.columns else ['MEAN']
    mismatches = []
    for bid in basin_ids:
        src_path = Path(exp_dir) / bid / BEST_METRICS
        if not src_path.is_file():
            mismatches.append((bid, period, 'missing', np.nan, np.nan))
            continue
        for stat in stats:
            src = read_basin_period_row(src_path, period, stat=stat)
            if src is None:
                if stat == 'MEAN':
                    mismatches.append((bid, period, 'period', np.nan, np.nan))
                continue
            if flood and 'stat' in basin_rows.columns:
                dst = basin_rows.loc[
                    (basin_rows['basin_id'].astype(str) == bid)
                    & (basin_rows['stat'].astype(str).str.upper() == stat)
                ]
            else:
                dst = basin_rows.loc[basin_rows['basin_id'].astype(str) == bid]
            if dst.empty:
                mismatches.append((bid, period, f'{stat}/basin_id', np.nan, np.nan))
                continue
            for col in metric_names:
                if col not in src or col not in dst.columns:
                    continue
                a = float(pd.to_numeric(src[col], errors='coerce'))
                b = float(pd.to_numeric(dst.iloc[0][col], errors='coerce'))
                if not metric_values_close(a, b):
                    mismatches.append((bid, period, f'{stat}/{col}', a, b))
    return mismatches


# ---------------------------------------------------------------------------
# Publish
# ---------------------------------------------------------------------------


def publish_experiment_period_metrics(exp_dir, basin_ids):
    log = get_logger('artifacts')
    exp_dir = Path(exp_dir)
    basin_ids = sorted(str(b) for b in basin_ids)
    flood = is_flood_metrics_format(exp_dir, basin_ids)

    if flood:
        publish_flood_period_metrics(exp_dir, basin_ids)
    else:
        publish_longterm_period_metrics(exp_dir, basin_ids)

    mismatches = []
    for period in PERIOD_ORDER:
        filename = PERIOD_METRICS_FILES.get(period)
        if not filename or not (exp_dir / filename).is_file():
            continue
        mismatches.extend(
            verify_period_summary(exp_dir, basin_ids, period, filename, flood=flood)
        )
        if flood:
            mismatches.extend(
                verify_period_event_summary(exp_dir, basin_ids, period, filename)
            )

    if mismatches:
        details = '; '.join(
            (
                f'{bid}/{period}/{col}: basin={a:.4f} summary={b:.4f}'
                if col not in ('missing', 'period', 'basin_id')
                else f'{bid}/{period}/{col}'
            )
            for bid, period, col, a, b in mismatches
        )
        raise RuntimeError(f'metrics summary mismatch: {details}')

    log.info(
        'metrics summary published: %s/{%s} (%s)',
        exp_dir,
        ', '.join(PERIOD_METRICS_FILES.values()),
        'flood' if flood else 'long-term',
    )


# ---------------------------------------------------------------------------
# Regenerate
# ---------------------------------------------------------------------------


def regenerate_basin_metrics(basin_dir, basin_id):
    path = Path(basin_dir) / EVENT_METRICS
    if not path.is_file():
        raise FileNotFoundError(f'missing {path}')

    df = pd.read_csv(path)
    names = metric_cols(df, EVENT_META)
    period_stat_metrics = {}
    for period, group in df.groupby('period', sort=False):
        event_rows = [
            {name: float(pd.to_numeric(row[name], errors='coerce')) for name in names}
            for row in group.to_dict('records')
        ]
        period_stat_metrics[str(period).lower()] = aggregate_event_metrics(
            event_rows,
            names,
        )
    write_basin_metrics(basin_dir, str(basin_id), period_stat_metrics)
    return period_stat_metrics


def regenerate_experiment_metrics(exp_dir, basin_ids):
    exp_dir = Path(exp_dir)
    for basin_id in sorted(str(b) for b in basin_ids):
        regenerate_basin_metrics(exp_dir / basin_id, basin_id)
    publish_experiment_period_metrics(exp_dir, basin_ids)


# ---------------------------------------------------------------------------
# Params + logging
# ---------------------------------------------------------------------------


def publish_experiment_params_summary(exp_dir, basin_ids):
    from hydromodels_dlm.config.run_config import dump_json

    exp_dir = Path(exp_dir)
    rows = []
    for bid in sorted(str(b) for b in basin_ids):
        path = exp_dir / bid / BEST_PARAMS
        if not path.is_file():
            continue
        with path.open(encoding='utf-8') as handle:
            rows.append(json.load(handle))
    if rows:
        dump_json(rows, exp_dir / BEST_PARAMS_ALL)


def append_regional_eval_summary(
    exp_dir,
    metric_names,
    *,
    train_info=None,
    elapsed_s=None,
):
    exp_dir = Path(exp_dir)
    log_path = exp_dir / TRAINING_LOG
    if not log_path.is_file():
        return

    names = [str(m).upper() for m in metric_names]
    lines = [
        '',
        '─' * 48,
        '▶ Evaluate Results',
    ]
    if elapsed_s is not None:
        lines.append(f'{LOG_INDENT}Time {format_elapsed_hms(elapsed_s)}')
    if train_info:
        best_loss = train_info.get('best_valid_loss', float('nan'))
        best_epoch = train_info.get('best_epoch')
        lines.append(
            f'{LOG_INDENT}Best valid loss: {best_loss:.4f}  |  Best epoch: {best_epoch}'
        )

    for label in STAT_ORDER:
        for period in PERIOD_ORDER:
            summary_path = exp_dir / PERIOD_METRICS_FILES[period]
            if not summary_path.is_file():
                continue
            df = pd.read_csv(summary_path)
            if 'stat' in df.columns:
                row = df.loc[df['stat'].astype(str).str.upper() == label]
            else:
                row = df.loc[df['basin_id'].astype(str).str.upper() == label]
            if row.empty:
                continue
            row = row.iloc[0]
            parts = ', '.join(f'{name}={row[name]}' for name in names if name in df.columns)
            lines.append(f'{LOG_INDENT}[{period}] {label}: {parts}')
    with log_path.open('a', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')
