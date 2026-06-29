from hydromodels_pbm.dataset.data_preprocess import (
    build_driver_arrays,
    check_arrays,
    clip_range,
    eval_step_indices,
    log_driver_quality,
    log_flood_load,
    log_longterm_load,
    observation_date_range,
    parse_period,
    period_dates,
    period_step_indices,
)
from hydromodels_pbm.dataset.data_source import (
    event_id_from_stem,
    group_file_ids_by_basin,
    input_data_root,
    is_flood_event_id,
    load_series,
    parse_variables,
    resolve_basin_ids,
)


# ---------------------------------------------------------------------------
# Loader base
# ---------------------------------------------------------------------------


class LoaderBase:
    def __init__(self, data_cfgs, root=None, file_ids=None):
        self.config = data_cfgs
        self.root = root or input_data_root(data_cfgs)
        self.file_ids = file_ids or resolve_basin_ids(data_cfgs, root=self.root)
        self.config['basin_ids'] = self.file_ids
        self.warmup_length = int(data_cfgs['warmup_length'])
        self.variables, self.input_keys, self.output_keys = parse_variables(data_cfgs)
        self.obs_key = self.output_keys[0]
        self.series_cache = {}

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

    def date_range(self, period, basin_id):
        q_bounds = observation_date_range(
            self.read_series(basin_id),
            self.obs_key,
        )
        return period_dates(self.config, parse_period(period), q_bounds)

    def load(self, basin_id, *, start, end):
        bid = str(basin_id)
        date_range = [start, end]
        series = clip_range(self.read_series(bid), date_range)
        p_and_e, qobs = build_driver_arrays(series, self.input_keys, self.obs_key)
        check_arrays(
            p_and_e,
            qobs,
            self.input_keys,
            name=f'{start}_{end}',
            warmup_length=self.warmup_length,
        )
        return p_and_e, qobs

    def load_calibration(self, basin_id, *, model=None, algorithm=None, output_dir=None):
        if not self.config['valid_period']:
            raise ValueError('valid_period is required in data config')
        bid = str(basin_id)
        calib_dr = self.date_range('calib', bid)
        valid_dr = self.date_range('valid', bid)
        log_longterm_load(
            self.root,
            bid,
            self.config,
            model=model,
            algorithm=algorithm,
            output_dir=output_dir,
        )
        span = [calib_dr[0], valid_dr[1]]
        series_span = clip_range(self.read_series(bid), span)
        log_driver_quality(
            series_span,
            self.input_keys,
            self.obs_key,
            log_tag=f'{span[0]}_{span[1]}',
            period_ranges={
                'calib': {'dates': calib_dr, 'warmup': self.warmup_length},
                'valid': {'dates': valid_dr},
            },
        )
        series_calib = clip_range(series_span, calib_dr)
        p_and_e, qobs = build_driver_arrays(series_calib, self.input_keys, self.obs_key)
        check_arrays(
            p_and_e,
            qobs,
            self.input_keys,
            name='calib',
            warmup_length=self.warmup_length,
        )
        calib_idx = period_step_indices(
            series_calib.index,
            self.warmup_length,
            calib_dr,
        )
        return p_and_e, qobs, calib_idx


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

    def event_stems(self, basin_id):
        return list(self.event_map[str(basin_id)])

    def date_range(self, period, basin_id):
        return period_dates(self.config, parse_period(period), q_bounds=None)

    def build_event_bundle(self, event_stem, date_range, period, *, with_arrays=True):
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
        bundle = {
            'event_stem': event_stem,
            'event_id': event_id_from_stem(event_stem),
            'series': series,
            'eval_idx': eval_idx,
        }
        if not with_arrays:
            return bundle
        p_and_e, qobs = build_driver_arrays(series, self.input_keys, self.obs_key)
        check_arrays(
            p_and_e,
            qobs,
            self.input_keys,
            name=f'{period}:{event_stem}',
            warmup_length=self.warmup_length,
        )
        bundle['p_and_e'] = p_and_e
        bundle['qobs'] = qobs
        return bundle

    def iter_period_bundles(self, basin_id, period, with_arrays=True):
        bid = str(basin_id)
        date_range = self.date_range(period, bid)
        for stem in self.event_stems(bid):
            bundle = self.build_event_bundle(
                stem, date_range, period, with_arrays=with_arrays
            )
            if bundle is not None:
                yield bundle

    def list_period_event_ids(self, basin_id, period):
        return [
            bundle['event_id']
            for bundle in self.iter_period_bundles(basin_id, period, with_arrays=False)
        ]

    def load_period_events(self, basin_id, period):
        bid = str(basin_id)
        events = list(
            self.iter_period_bundles(basin_id, period, with_arrays=True)
        )
        if not events:
            raise ValueError(
                f'no flood events with data in {period} period for basin {bid}'
            )
        return events

    def load_calibration(self, basin_id, *, model=None, algorithm=None, output_dir=None):
        if not self.config['valid_period']:
            raise ValueError('valid_period is required in data config')
        bid = str(basin_id)
        calib_events = self.load_period_events(bid, 'calib')
        log_flood_load(
            self.root,
            bid,
            [ev['event_id'] for ev in calib_events],
            self.list_period_event_ids(bid, 'valid'),
            model=model,
            algorithm=algorithm,
            output_dir=output_dir,
        )
        return calib_events


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_data_loader(data_cfgs):
    root = input_data_root(data_cfgs)
    file_ids = resolve_basin_ids(data_cfgs, root=root)
    if all(is_flood_event_id(fid) for fid in file_ids):
        return FloodEventDataLoader(data_cfgs, root=root, file_ids=file_ids)
    return LongTermDataLoader(data_cfgs, root=root, file_ids=file_ids)
