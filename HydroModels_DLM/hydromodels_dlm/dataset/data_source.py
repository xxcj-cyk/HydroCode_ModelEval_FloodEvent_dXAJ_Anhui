from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------


def format_timestamp(ts):
    ts = pd.Timestamp(ts)
    if (
        ts.hour == 0
        and ts.minute == 0
        and ts.second == 0
        and ts.microsecond == 0
    ):
        return ts.strftime('%Y-%m-%d')
    return ts.strftime('%Y-%m-%d %H:%M:%S')


# ---------------------------------------------------------------------------
# Paths and basin ids
# ---------------------------------------------------------------------------


def input_data_root(data_cfgs):
    path = data_cfgs['input_path']
    if not path:
        raise ValueError('data_cfgs.input_path is required')
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f'data directory not found: {root}')
    return root


def glob_basin_ids(root, pattern='*.csv'):
    root = Path(root)
    ids = sorted(
        p.stem
        for p in root.glob(pattern)
        if p.is_file() and not p.name.startswith('Attributes_')
    )
    if not ids:
        raise ValueError(f'no csv files match {pattern!r} under {root}')
    return ids


def resolve_basin_ids(data_cfgs, root=None):
    configured = data_cfgs['basin_ids']
    ids = [str(b) for b in configured if b] if configured else []
    root = root or input_data_root(data_cfgs)
    if not ids:
        ids = glob_basin_ids(root)
    missing = [fid for fid in ids if not (root / f'{fid}.csv').is_file()]
    if missing:
        raise FileNotFoundError(
            f'missing csv for file id(s): {", ".join(missing)} under {root}'
        )
    flood_flags = [is_flood_event_id(fid) for fid in ids]
    if any(flood_flags) and not all(flood_flags):
        raise ValueError(
            'basin_ids mixes long-term (prefix_basinid) and flood-event '
            '(prefix_basinid_eventid) file names'
        )
    return ids


# ---------------------------------------------------------------------------
# File naming
# ---------------------------------------------------------------------------


def is_flood_event_id(file_stem):
    return len(str(file_stem).split('_', 2)) >= 3


def split_file_stem(file_stem):
    parts = str(file_stem).split('_', 2)
    if len(parts) < 3:
        return str(file_stem), None
    return f'{parts[0]}_{parts[1]}', parts[2]


def event_id_from_stem(file_stem):
    _, event = split_file_stem(file_stem)
    return event


def group_file_ids_by_basin(file_ids):
    groups = {}
    for stem in file_ids:
        logical, _ = split_file_stem(stem)
        groups.setdefault(logical, []).append(str(stem))
    for key in groups:
        groups[key] = sorted(groups[key])
    return groups


# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------


def parse_variables(data_cfgs):
    raw = data_cfgs['variables']
    if not isinstance(raw, dict):
        raise ValueError('variables must contain dynamic_inputs and dynamic_outputs')
    for key in ('dynamic_inputs', 'dynamic_outputs'):
        if key not in raw:
            raise ValueError(f'variables missing {key!r}')
    din, dout = raw['dynamic_inputs'], raw['dynamic_outputs']
    if not isinstance(din, dict) or not isinstance(dout, dict):
        raise TypeError('dynamic_inputs and dynamic_outputs must be dicts')
    variables = {
        'dynamic_inputs': {str(k): str(v) for k, v in din.items() if v},
        'dynamic_outputs': {str(k): str(v) for k, v in dout.items() if v},
    }
    if not variables['dynamic_inputs']:
        raise ValueError('dynamic_inputs must not be empty')
    if not variables['dynamic_outputs']:
        raise ValueError('dynamic_outputs must not be empty')

    static_raw = raw.get('static_attributes')
    if not isinstance(static_raw, list):
        raise TypeError('variables.static_attributes must be a list of column names')
    static_keys = [str(name) for name in static_raw if name]
    if not static_keys:
        raise ValueError('variables.static_attributes must be a non-empty list')
    variables['static_attributes'] = static_keys

    input_keys = list(variables['dynamic_inputs'].keys())
    output_keys = list(variables['dynamic_outputs'].keys())
    return variables, input_keys, output_keys, static_keys


# ---------------------------------------------------------------------------
# Series I/O
# ---------------------------------------------------------------------------


def load_series(
    root,
    file_stem,
    variables,
    input_keys,
    output_keys,
    *,
    include_flood_col=False,
):
    path = Path(root) / f'{file_stem}.csv'
    if not path.is_file():
        raise FileNotFoundError(f'file not found: {path}')
    tag = f'{file_stem}.csv'

    df = pd.read_csv(path)
    if 'time' not in df.columns:
        raise KeyError(f"CSV must have column 'time'; got {list(df.columns)}")

    din = variables['dynamic_inputs']
    dout = variables['dynamic_outputs']
    col_map = {}
    out_map = {}
    missing = []
    for sym in input_keys:
        name = din[sym]
        if not name or name not in df.columns:
            missing.append((sym, name or ''))
        else:
            col_map[sym] = name
    for sym in output_keys:
        name = dout[sym]
        if not name or name not in df.columns:
            missing.append((sym, name or ''))
        else:
            out_map[sym] = name
    if missing:
        detail = ', '.join(f'{s}={n!r}' for s, n in missing)
        raise KeyError(f'{tag} missing columns: {detail}')

    file_cols = [col_map[k] for k in input_keys] + [out_map[k] for k in output_keys]
    rename = {col_map[k]: k for k in input_keys}
    rename.update({out_map[k]: k for k in output_keys})
    include_time_true = False
    if include_flood_col:
        if 'flood_event' not in df.columns:
            raise KeyError(f'{tag} missing flood flag column flood_event')
        file_cols.append('flood_event')
        include_time_true = 'time_true' in df.columns
        if include_time_true:
            file_cols.append('time_true')

    df = df.assign(time=pd.to_datetime(df['time']))
    if include_time_true:
        df['time_true'] = pd.to_datetime(df['time_true'])
    df = df.set_index('time').sort_index()
    out_cols = list(input_keys) + list(output_keys)
    if include_flood_col:
        out_cols.append('flood_event')
        if include_time_true:
            out_cols.append('time_true')
    return df[file_cols].rename(columns=rename)[out_cols]


# ---------------------------------------------------------------------------
# Attributes
# ---------------------------------------------------------------------------


def load_attributes(root, static_columns, basin_ids=None):
    root = Path(root)
    matches = sorted(root.glob('Attributes*.csv'))
    if not matches:
        raise FileNotFoundError(f'no Attributes*.csv under {root}')
    if len(matches) > 1:
        names = ', '.join(p.name for p in matches)
        raise ValueError(
            f'multiple Attributes*.csv files under {root}: {names}'
        )

    path = matches[0]
    frame = pd.read_csv(path)
    key_col = 'Basin_ID'
    if key_col not in frame.columns:
        raise KeyError(
            f"{path.name} must have column {key_col!r}; "
            f'got {list(frame.columns)}'
        )

    missing_cols = [col for col in static_columns if col not in frame.columns]
    if missing_cols:
        raise KeyError(
            f'{path.name} missing static attribute column(s): {missing_cols!r}'
        )

    subset = frame[[key_col, *static_columns]].copy()
    subset[key_col] = subset[key_col].astype(str)
    if subset[key_col].duplicated().any():
        dupes = subset.loc[subset[key_col].duplicated(), key_col].tolist()
        raise ValueError(f'duplicate {key_col} in {path.name}: {dupes[:5]}')

    out = subset.set_index(key_col)
    if basin_ids is not None:
        missing_basins = [
            str(bid) for bid in basin_ids if str(bid) not in out.index
        ]
        if missing_basins:
            preview = ', '.join(dict.fromkeys(missing_basins[:5]))
            suffix = '...' if len(missing_basins) > 5 else ''
            raise KeyError(
                f'basin(s) missing from attributes table: {preview}{suffix}'
            )
    return out


def static_attribute_value(attributes_frame, basin_id, column):
    bid = str(basin_id)
    if bid not in attributes_frame.index:
        raise KeyError(f'basin {basin_id!r} not found in attributes')
    if column not in attributes_frame.columns:
        raise KeyError(
            f'static attribute column {column!r} not in attributes frame'
        )
    return float(attributes_frame.loc[bid, column])
