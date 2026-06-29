from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader as TorchDataLoader

from hydromodels_dlm.dataset.data_preprocess import (
    check_series,
    clean_series,
    clip_range,
    eval_step_indices,
    log_driver_quality,
    log_driver_quality_aggregate,
    log_flood_load,
    log_longterm_load,
    observation_date_range,
    parse_period,
    period_dates,
    period_series_for_stats,
)
from hydromodels_dlm.dataset.data_source import (
    event_id_from_stem,
    group_file_ids_by_basin,
    input_data_root,
    is_flood_event_id,
    load_attributes,
    load_series,
    parse_variables,
    resolve_basin_ids,
    static_attribute_value,
)
from hydromodels_dlm.dataset.input_layout import INPUT_LAYOUT_REGISTRY
from hydromodels_dlm.dataset.scaler import fit_basin_scalers, load_scalers, save_scalers
from hydromodels_dlm.model.physics_guided import driver_indices, is_physics_model

DL_PERIODS = ('train', 'valid', 'test')
ROLLING_WARMUP_PERIOD = {'valid': 'train', 'test': 'valid'}


# ---------------------------------------------------------------------------
# Input layout resolution
# ---------------------------------------------------------------------------


def get_input_layout(name):
    layout = INPUT_LAYOUT_REGISTRY.get(name)
    if layout is None:
        raise ValueError(
            f'unknown input_layout {name!r}; '
            f'choose from: {sorted(INPUT_LAYOUT_REGISTRY)}'
        )
    return layout


# ---------------------------------------------------------------------------
# Windowing (train sampler + rolling eval)
# ---------------------------------------------------------------------------


def train_window_slice(start, warmup, horizon):
    end = start + horizon
    return slice(start - warmup, end), slice(start, end)


def concat_warmup_tail(x_warm, x_tgt, warmup_length):
    if x_warm is None:
        return x_tgt
    return np.vstack([x_warm[-warmup_length:], x_tgt])


def event_flood_target_mask(series):
    if 'flood_event' not in series.columns:
        raise KeyError("series missing flood flag column 'flood_event'")
    return series['flood_event'].fillna(0).astype(bool).to_numpy()


def train_window_lookup(
    x_per_basin,
    y_per_basin,
    *,
    warmup,
    horizon,
    flood_target_mask=None,
):
    lookup = []
    for basin_idx, (x, y) in enumerate(zip(x_per_basin, y_per_basin)):
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)
        n_time = y_arr.shape[0]
        flood_mask = None
        if flood_target_mask is not None:
            flood_mask = np.asarray(flood_target_mask[basin_idx], dtype=bool)
            if flood_mask.shape[0] != n_time:
                raise ValueError(
                    f'flood_target_mask length {flood_mask.shape[0]} != n_time {n_time}'
                )
        for start in range(warmup, n_time - horizon + 1):
            input_sl, target_sl = train_window_slice(start, warmup, horizon)
            if flood_mask is not None and not np.all(flood_mask[start : start + horizon]):
                continue
            if np.any(np.isnan(y_arr[target_sl])):
                continue
            if np.any(np.isnan(x_arr[input_sl])):
                continue
            lookup.append((basin_idx, start))
    return lookup


def parse_dl_period(period):
    key = parse_period(period).lower()
    if key not in DL_PERIODS:
        raise ValueError(f'unknown period {period!r}')
    return key


def norm_static_vector(loader, hub, basin_id):
    return hub.transform_static(loader.static_vector(basin_id))


def static_torch_tensor(layout, static):
    if static is not None and layout['merge_static'] is not None:
        return torch.from_numpy(static.astype(np.float32))
    return None


def scaled_rolling_sequence(loader, basin_id, period_key, scalers):
    bid = str(basin_id)
    x_tgt, y_tgt, times_tgt = loader.scaled_arrays(bid, period_key, scalers)
    warmup_period = loader.rolling_warmup_period(period_key)
    if warmup_period is None:
        return x_tgt, y_tgt, times_tgt, loader.warmup_length
    x_warm, _, _ = loader.scaled_arrays(bid, warmup_period, scalers)
    x_seq = concat_warmup_tail(x_warm, x_tgt, loader.warmup_length)
    return x_seq, y_tgt, times_tgt, loader.warmup_length


def physical_rolling_sequence(loader, basin_id, period_key, scalers):
    bid = str(basin_id)
    hub = loader.require_scalers(scalers)[bid]
    x_tgt, y_tgt, times_tgt = loader.arrays_for_period(bid, period_key)
    y_norm = hub.y_scaler.transform(y_tgt)
    warmup_period = loader.rolling_warmup_period(period_key)
    if warmup_period is None:
        return x_tgt, y_norm, times_tgt, loader.warmup_length
    x_warm, _, _ = loader.arrays_for_period(bid, warmup_period)
    x_raw = concat_warmup_tail(x_warm, x_tgt, loader.warmup_length)
    return x_raw, y_norm, times_tgt, loader.warmup_length


def prepare_period_basin(loader, basin_id, period, hub, layout, *, driver_indices=None):
    x_raw, y_raw, _ = loader.arrays_for_period(basin_id, period)
    data = {
        'x_norm': hub.x_scaler.transform(x_raw),
        'y_norm': hub.y_scaler.transform(y_raw),
        'y_raw': np.asarray(y_raw, dtype=np.float64),
        'static': None,
    }
    if driver_indices is not None:
        data['drivers'] = x_raw[:, driver_indices]
    if layout['uses_static']:
        data['static'] = norm_static_vector(loader, hub, basin_id)
    return data


def prepare_train_basin(loader, basin_id, hub, layout, *, driver_indices=None):
    return prepare_period_basin(
        loader,
        basin_id,
        'train',
        hub,
        layout,
        driver_indices=driver_indices,
    )


# ---------------------------------------------------------------------------
# Loader base
# ---------------------------------------------------------------------------


class LoaderBase:
    def __init__(self, data_cfgs, root=None, file_ids=None):
        self.config = data_cfgs
        self.root = root or input_data_root(data_cfgs)
        self.file_ids = file_ids or resolve_basin_ids(data_cfgs, root=self.root)
        self.warmup_length = int(data_cfgs['warmup_length'])
        self.forecast_length = int(data_cfgs.get('forecast_length', 1))
        self.input_layout_name = str(
            data_cfgs.get('input_layout', 'dynamic_static')
        )
        get_input_layout(self.input_layout_name)

        self.variables, self.input_keys, self.output_keys, self.static_keys = (
            parse_variables(data_cfgs)
        )
        self.obs_key = self.output_keys[0]
        self.series_cache = {}
        self.scalers = None
        self.quality_logged = set()
        self.regional_quality_logged = False

    def read_series(self, file_stem, include_flood_col=False):
        cache_key = (str(file_stem), include_flood_col)
        if cache_key in self.series_cache:
            return self.series_cache[cache_key]
        series = load_series(
            self.root,
            file_stem,
            self.variables,
            self.input_keys,
            self.output_keys,
            include_flood_col=include_flood_col,
        )
        self.series_cache[cache_key] = series
        return series

    def time_index_for_basin(self, basin_id):
        return self.read_series(str(basin_id)).index

    def static_vector(self, basin_id, static_keys=None):
        keys = static_keys or self.static_keys
        return np.asarray(
            [self.static_value(basin_id, key) for key in keys],
            dtype=np.float64,
        )

    def static_value(self, basin_id, column):
        return static_attribute_value(self.attributes_frame, basin_id, column)

    def prcp_mean_column(self):
        keys = [str(name) for name in self.static_keys]
        configured = self.config.get('prcp_mean_attribute')
        if configured:
            name = str(configured)
            if name not in keys:
                raise ValueError(
                    f'prcp_mean_attribute {name!r} not in static_attributes {keys!r}'
                )
            return name
        for name in ('Pmean_camels', 'Pmean', 'Pmean_hydroatlas'):
            if name in keys:
                return name
        raise ValueError(
            'prcp_log1p_zscore on Q requires Pmean_camels, Pmean, or '
            f'Pmean_hydroatlas in variables.static_attributes; got {keys!r}'
        )

    def prcp_scale(self, basin_id):
        pmean_yearly = self.static_value(basin_id, self.prcp_mean_column())
        value = float(pmean_yearly)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(
                f'{self.prcp_mean_column()} must be positive, got {pmean_yearly!r}'
            )
        times = self.time_index_for_basin(basin_id)
        if len(times) == 0:
            raise ValueError(f'empty time index for basin {basin_id!r}')
        daily = bool(
            (
                (times.hour == 0)
                & (times.minute == 0)
                & (times.second == 0)
                & (times.microsecond == 0)
            ).all()
        )
        if daily:
            scale, unit, step = value / 365, 'mm/d', 'daily'
        else:
            scale, unit, step = value / 8760, 'mm/h', 'hourly'
        return scale, {
            'p_mean_yearly_mm': pmean_yearly,
            'series_frequency': step,
            'prcp_scale_unit': unit,
        }

    def build_period_sampler(
        self,
        basin_ids=None,
        scalers=None,
        input_layout=None,
        *,
        model_name=None,
        period='train',
    ):
        raise NotImplementedError(
            f'{type(self).__name__} must implement build_period_sampler()'
        )

    def build_period_loader(
        self,
        basin_ids=None,
        scalers=None,
        *,
        period,
        batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        input_layout=None,
        model_name=None,
        pin_memory=False,
    ):
        from torch.utils.data import DataLoader as TorchDataLoader

        from hydromodels_dlm.model.physics_guided import is_physics_model

        sampler = self.build_period_sampler(
            basin_ids,
            scalers,
            input_layout=input_layout,
            model_name=model_name,
            period=period,
        )
        n = len(sampler)
        collate_fn = None
        if model_name and is_physics_model(model_name):
            from hydromodels_dlm.dataset.sampler import physics_guided_collate

            collate_fn = physics_guided_collate
        return TorchDataLoader(
            sampler,
            batch_size=min(int(batch_size), max(n, 1)),
            shuffle=shuffle,
            num_workers=int(num_workers),
            drop_last=drop_last and n > batch_size,
            pin_memory=bool(pin_memory),
            collate_fn=collate_fn,
        )

    def build_train_sampler(
        self,
        basin_ids=None,
        scalers=None,
        input_layout=None,
        *,
        model_name=None,
    ):
        return self.build_period_sampler(
            basin_ids,
            scalers,
            input_layout=input_layout,
            model_name=model_name,
            period='train',
        )

    def build_train_loader(
        self,
        basin_ids=None,
        scalers=None,
        *,
        batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
        input_layout=None,
        model_name=None,
        pin_memory=False,
    ):
        return self.build_period_loader(
            basin_ids,
            scalers,
            period='train',
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            input_layout=input_layout,
            model_name=model_name,
            pin_memory=pin_memory,
        )


# ---------------------------------------------------------------------------
# Long-term loader
# ---------------------------------------------------------------------------


class LongTermDataLoader(LoaderBase):
    def __init__(self, data_cfgs, root=None, file_ids=None):
        super().__init__(data_cfgs, root=root, file_ids=file_ids)
        if any(is_flood_event_id(fid) for fid in self.file_ids):
            raise ValueError(
                'LongTermDataLoader expects prefix_basinid; '
                'use FloodEventDataLoader for prefix_basinid_eventid'
            )
        self.basin_ids = list(self.file_ids)
        self.config['basin_ids'] = self.basin_ids
        self.attributes_frame = load_attributes(
            self.root, self.static_keys, self.basin_ids
        )

    def streamflow_bounds(self, basin_id):
        return observation_date_range(self.read_series(basin_id), self.obs_key)

    def date_range(self, period, basin_id):
        label = parse_period(period)
        if label.lower() not in DL_PERIODS and ',' not in label:
            raise ValueError(f'unknown period {period!r}')
        return period_dates(
            self.config, label, self.streamflow_bounds(str(basin_id))
        )

    def period_missing_ranges(self, basin_id):
        bid = str(basin_id)
        return {
            'train': {
                'dates': self.date_range('train', bid),
                'warmup': self.warmup_length,
            },
            'valid': {'dates': self.date_range('valid', bid)},
            'test': {'dates': self.date_range('test', bid)},
        }

    def log_period_missing_values(self, basin_id, *, force=False):
        bid = str(basin_id)
        if not force and bid in self.quality_logged:
            return
        ranges = self.period_missing_ranges(bid)
        span = [ranges['train']['dates'][0], ranges['test']['dates'][1]]
        series = clip_range(self.read_series(bid), span)
        log_driver_quality(
            series,
            self.input_keys,
            self.obs_key,
            log_tag=bid,
            period_ranges=ranges,
        )
        self.quality_logged.add(bid)

    def regional_span_tag(self, basin_ids):
        starts = []
        ends = []
        for bid in basin_ids:
            ranges = self.period_missing_ranges(bid)
            starts.append(ranges['train']['dates'][0])
            ends.append(ranges['test']['dates'][1])
        return f'{min(starts)}_{max(ends)}'

    def log_regional_missing_values(self, basin_ids=None):
        if self.regional_quality_logged:
            return
        basin_ids = [str(b) for b in (basin_ids or self.basin_ids)]
        columns = list(self.input_keys) + [self.obs_key]
        span_series = []
        period_buckets = {
            'train': {'tag': None, 'series': []},
            'valid': {'tag': None, 'series': []},
            'test': {'tag': None, 'series': []},
        }
        for bid in basin_ids:
            ranges = self.period_missing_ranges(bid)
            span = [ranges['train']['dates'][0], ranges['test']['dates'][1]]
            span_series.append(clip_range(self.read_series(bid), span))
            for label in period_buckets:
                spec = ranges[label]
                date_range = spec['dates']
                warmup = int(spec.get('warmup', 0))
                sub, eff_dr = period_series_for_stats(
                    span_series[-1], date_range, warmup=warmup
                )
                if period_buckets[label]['tag'] is None:
                    period_buckets[label]['tag'] = f'{eff_dr[0]}_{eff_dr[1]}'
                period_buckets[label]['series'].append(sub)

        span_tag = self.regional_span_tag(basin_ids)
        log_driver_quality_aggregate(
            span_series,
            columns,
            span_tag,
            period_buckets=period_buckets,
        )
        self.regional_quality_logged = True
        self.quality_logged.update(basin_ids)

    def log_load_context(
        self,
        basin_id,
        *,
        model=None,
        algorithm=None,
        output_dir=None,
        input_path=None,
        device=None,
    ):
        log_longterm_load(
            self.root,
            basin_id,
            self.config,
            model=model,
            algorithm=algorithm,
            output_dir=output_dir,
            input_path=input_path,
            device=device,
        )

    def period_warmup_for_check(self, period):
        key = parse_period(period).lower()
        if key == 'train':
            return self.warmup_length
        return 0

    def check_rolling_source(self, basin_id, target_period):
        source_period = self.rolling_warmup_period(target_period)
        if source_period is None:
            return
        bid = str(basin_id)
        source = self.period_series(bid, source_period)
        need = self.warmup_length
        if len(source) < need:
            raise ValueError(
                f'{bid} rolling warmup for {target_period!r} needs {need} days '
                f'from {source_period!r}, got {len(source)}'
            )

    def period_series(self, basin_id, period):
        bid = str(basin_id)
        dr = self.date_range(period, bid)
        series = clip_range(self.read_series(bid), dr)
        series = clean_series(series, self.input_keys, self.obs_key)
        check_series(
            series,
            self.input_keys,
            name=period,
            warmup_length=self.period_warmup_for_check(period),
        )
        return series

    def arrays_for_period(self, basin_id, period):
        series = self.period_series(basin_id, period)
        x = series[self.input_keys].to_numpy(dtype=np.float64)
        y = series[[self.obs_key]].to_numpy(dtype=np.float64)
        return x, y, series.index

    def fit_scalers(self, basin_ids=None):
        basin_ids = basin_ids or self.basin_ids
        self.scalers = fit_basin_scalers(self, basin_ids)
        return self.scalers

    def save_scalers(self, path, scalers=None):
        scalers = scalers if scalers is not None else self.scalers
        if scalers is None:
            raise ValueError('no scalers to save; call fit_scalers first')
        save_scalers(path, scalers)

    def load_scalers_from(self, path):
        self.scalers = load_scalers(path)
        return self.scalers

    def require_scalers(self, scalers=None):
        hubs = scalers if scalers is not None else self.scalers
        if hubs is None:
            raise ValueError('scalers required; call fit_scalers or load_scalers_from')
        return hubs

    def scaled_arrays(self, basin_id, period, scalers):
        bid = str(basin_id)
        x, y, times = self.arrays_for_period(bid, period)
        hub = self.require_scalers(scalers)[bid]
        return hub.x_scaler.transform(x), hub.y_scaler.transform(y), times

    def model_input_size(self, scalers=None, *, input_layout=None):
        layout = get_input_layout(
            input_layout or self.input_layout_name
        )
        return layout['input_dim'](len(self.input_keys), len(self.static_keys))

    def build_period_sampler(
        self,
        basin_ids=None,
        scalers=None,
        input_layout=None,
        *,
        model_name=None,
        period='train',
    ):
        basins = basin_ids or self.basin_ids
        hubs = self.require_scalers(scalers)
        layout_name = input_layout or self.input_layout_name
        period_key = parse_dl_period(period)
        from hydromodels_dlm.dataset.sampler import (
            PhysicsGuidedTrainSampler,
            SlidingWindowTrainSampler,
        )

        if model_name and is_physics_model(model_name):
            return PhysicsGuidedTrainSampler(
                self,
                basins,
                hubs,
                driver_indices=driver_indices(model_name, self.input_keys),
                period=period_key,
            )
        return SlidingWindowTrainSampler(
            self,
            basins,
            hubs,
            input_layout=layout_name,
            period=period_key,
        )

    def rolling_warmup_period(self, target_period):
        key = parse_period(target_period).lower()
        return ROLLING_WARMUP_PERIOD.get(key)

    def build_rolling_tensors(
        self, basin_id, target_period, scalers, input_layout=None
    ):
        bid = str(basin_id)
        period_key = parse_dl_period(target_period)
        self.check_rolling_source(bid, period_key)

        layout = get_input_layout(input_layout or self.input_layout_name)
        hub = self.require_scalers(scalers)[bid]
        static = norm_static_vector(self, hub, bid) if layout['uses_static'] else None

        x_seq, y_tgt, times_tgt, lead = scaled_rolling_sequence(
            self, bid, period_key, scalers
        )
        x_dyn = layout['series'](x_seq, static)
        return (
            torch.from_numpy(x_dyn.astype(np.float32)),
            static_torch_tensor(layout, static),
            torch.from_numpy(y_tgt.astype(np.float32)),
            times_tgt,
            lead,
        )

    def build_rolling_physics_tensors(
        self, basin_id, target_period, scalers, model_name, input_layout=None
    ):
        bid = str(basin_id)
        period_key = parse_dl_period(target_period)
        self.check_rolling_source(bid, period_key)

        layout = get_input_layout(input_layout or self.input_layout_name)
        hub = self.require_scalers(scalers)[bid]
        static = norm_static_vector(self, hub, bid) if layout['uses_static'] else None
        driver_idx = driver_indices(model_name, self.input_keys)

        x_raw, y_norm, times_tgt, lead = physical_rolling_sequence(
            self, bid, period_key, scalers
        )
        x_norm = hub.x_scaler.transform(x_raw)
        return (
            torch.from_numpy(x_raw[:, driver_idx].astype(np.float32)),
            torch.from_numpy(layout['series'](x_norm, static).astype(np.float32)),
            static_torch_tensor(layout, static),
            torch.from_numpy(y_norm.astype(np.float32)),
            times_tgt,
            lead,
        )


# ---------------------------------------------------------------------------
# Flood-event loader
# ---------------------------------------------------------------------------


class FloodEventDataLoader(LoaderBase):
    def __init__(self, data_cfgs, root=None, file_ids=None):
        super().__init__(data_cfgs, root=root, file_ids=file_ids)
        if not all(is_flood_event_id(fid) for fid in self.file_ids):
            raise ValueError(
                'FloodEventDataLoader expects prefix_basinid_eventid file names'
            )
        self.event_map = group_file_ids_by_basin(self.file_ids)
        self.basin_ids = sorted(self.event_map.keys())
        self.config['basin_ids'] = self.file_ids
        self.attributes_frame = load_attributes(
            self.root, self.static_keys, self.basin_ids
        )

    def event_stems(self, basin_id):
        return list(self.event_map[str(basin_id)])

    def time_index_for_basin(self, basin_id):
        stems = self.event_stems(basin_id)
        if not stems:
            raise ValueError(f'no flood events for basin {basin_id!r}')
        return self.read_series(stems[0], include_flood_col=True).index

    def date_range(self, period, basin_id):
        label = parse_period(period)
        if label.lower() not in DL_PERIODS and ',' not in label:
            raise ValueError(f'unknown period {period!r}')
        return period_dates(self.config, label, q_bounds=None)

    def build_event_bundle(self, event_stem, date_range, period):
        series = clip_range(
            self.read_series(event_stem, include_flood_col=True),
            date_range,
        )
        if series.empty:
            return None
        try:
            eval_idx = eval_step_indices(
                series.index,
                self.warmup_length,
                date_range,
                series,
            )
        except ValueError:
            return None
        series = clean_series(series, self.input_keys, self.obs_key)
        check_series(
            series,
            self.input_keys,
            name=f'{period}:{event_stem}',
            warmup_length=self.warmup_length,
        )
        return {
            'event_stem': event_stem,
            'event_id': event_id_from_stem(event_stem),
            'series': series,
            'eval_idx': eval_idx,
        }

    def iter_period_bundles(self, basin_id, period):
        bid = str(basin_id)
        date_range = self.date_range(period, bid)
        for stem in self.event_stems(bid):
            bundle = self.build_event_bundle(stem, date_range, period)
            if bundle is not None:
                yield bundle

    def list_period_event_ids(self, basin_id, period):
        return [
            bundle['event_id']
            for bundle in self.iter_period_bundles(basin_id, period)
        ]

    def load_period_events(self, basin_id, period):
        bid = str(basin_id)
        events = list(self.iter_period_bundles(bid, period))
        if not events:
            raise ValueError(
                f'no flood events with data in {period} period for basin {bid}'
            )
        return events

    def log_load_context(
        self,
        basin_id,
        *,
        model=None,
        algorithm=None,
        output_dir=None,
        input_path=None,
        device=None,
    ):
        bid = str(basin_id)
        log_flood_load(
            self.root,
            bid,
            self.list_period_event_ids(bid, 'train'),
            self.list_period_event_ids(bid, 'valid'),
            test_event_ids=self.list_period_event_ids(bid, 'test'),
            model=model,
            algorithm=algorithm,
            output_dir=output_dir,
            input_path=input_path,
            device=device,
        )

    def log_period_missing_values(self, basin_id, *, force=False):
        return

    def log_regional_missing_values(self, basin_ids=None):
        return

    def bundle_arrays(self, bundle):
        series = bundle['series']
        x = series[self.input_keys].to_numpy(dtype=np.float64)
        y = series[[self.obs_key]].to_numpy(dtype=np.float64)
        return x, y, series.index

    def arrays_for_period(self, basin_id, period):
        xs = []
        ys = []
        times = []
        for bundle in self.iter_period_bundles(basin_id, period):
            x, y, index = self.bundle_arrays(bundle)
            xs.append(x)
            ys.append(y)
            times.append(index)
        if not xs:
            raise ValueError(
                f'no flood events with data in {period} period for basin {basin_id!r}'
            )
        all_times = np.concatenate([index.to_numpy() for index in times])
        return np.vstack(xs), np.vstack(ys), pd.DatetimeIndex(all_times)

    def fit_scalers(self, basin_ids=None):
        basin_ids = basin_ids or self.basin_ids
        self.scalers = fit_basin_scalers(self, basin_ids)
        return self.scalers

    def save_scalers(self, path, scalers=None):
        scalers = scalers if scalers is not None else self.scalers
        if scalers is None:
            raise ValueError('no scalers to save; call fit_scalers first')
        save_scalers(path, scalers)

    def load_scalers_from(self, path):
        self.scalers = load_scalers(path)
        return self.scalers

    def require_scalers(self, scalers=None):
        hubs = scalers if scalers is not None else self.scalers
        if hubs is None:
            raise ValueError('scalers required; call fit_scalers or load_scalers_from')
        return hubs

    def scaled_arrays(self, basin_id, period, scalers):
        bid = str(basin_id)
        x, y, times = self.arrays_for_period(bid, period)
        hub = self.require_scalers(scalers)[bid]
        return hub.x_scaler.transform(x), hub.y_scaler.transform(y), times

    def model_input_size(self, scalers=None, *, input_layout=None):
        layout = get_input_layout(input_layout or self.input_layout_name)
        return layout['input_dim'](len(self.input_keys), len(self.static_keys))

    def build_period_sampler(
        self,
        basin_ids=None,
        scalers=None,
        input_layout=None,
        *,
        model_name=None,
        period='train',
    ):
        basins = basin_ids or self.basin_ids
        hubs = self.require_scalers(scalers)
        layout_name = input_layout or self.input_layout_name
        period_key = parse_dl_period(period)
        from hydromodels_dlm.dataset.sampler import (
            FloodEventPhysicsGuidedTrainSampler,
            FloodEventSlidingWindowTrainSampler,
        )

        if model_name and is_physics_model(model_name):
            return FloodEventPhysicsGuidedTrainSampler(
                self,
                basins,
                hubs,
                driver_indices=driver_indices(model_name, self.input_keys),
                period=period_key,
            )
        return FloodEventSlidingWindowTrainSampler(
            self,
            basins,
            hubs,
            input_layout=layout_name,
            period=period_key,
        )

    def rolling_warmup_period(self, target_period):
        return None

    def check_rolling_source(self, basin_id, target_period):
        return

    def build_event_tensors(self, basin_id, bundle, scalers, input_layout=None):
        bid = str(basin_id)
        layout = get_input_layout(input_layout or self.input_layout_name)
        hub = self.require_scalers(scalers)[bid]
        static = norm_static_vector(self, hub, bid) if layout['uses_static'] else None
        x_raw, y_raw, times = self.bundle_arrays(bundle)
        x_norm = hub.x_scaler.transform(x_raw)
        y_norm = hub.y_scaler.transform(y_raw)
        x_dyn = layout['series'](x_norm, static)
        return (
            torch.from_numpy(x_dyn.astype(np.float32)),
            static_torch_tensor(layout, static),
            torch.from_numpy(y_norm.astype(np.float32)),
            times,
            self.warmup_length,
        )

    def build_event_physics_tensors(
        self, basin_id, bundle, scalers, model_name, input_layout=None
    ):
        bid = str(basin_id)
        layout = get_input_layout(input_layout or self.input_layout_name)
        hub = self.require_scalers(scalers)[bid]
        static = norm_static_vector(self, hub, bid) if layout['uses_static'] else None
        driver_idx = driver_indices(model_name, self.input_keys)
        x_raw, y_raw, times = self.bundle_arrays(bundle)
        x_norm = hub.x_scaler.transform(x_raw)
        y_norm = hub.y_scaler.transform(y_raw)
        return (
            torch.from_numpy(x_raw[:, driver_idx].astype(np.float32)),
            torch.from_numpy(layout['series'](x_norm, static).astype(np.float32)),
            static_torch_tensor(layout, static),
            torch.from_numpy(y_norm.astype(np.float32)),
            times,
            self.warmup_length,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_data_loader(data_cfgs):
    root = input_data_root(data_cfgs)
    file_ids = resolve_basin_ids(data_cfgs, root=root)
    if all(is_flood_event_id(fid) for fid in file_ids):
        return FloodEventDataLoader(data_cfgs, root=root, file_ids=file_ids)
    return LongTermDataLoader(data_cfgs, root=root, file_ids=file_ids)


DataLoader = LongTermDataLoader
