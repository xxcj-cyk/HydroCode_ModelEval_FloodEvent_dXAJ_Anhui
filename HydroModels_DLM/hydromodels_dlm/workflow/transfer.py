from pathlib import Path

import numpy as np
import torch

from hydromodels_dlm.dataset.data_loader import get_input_layout
from hydromodels_dlm.dataset.data_source import load_attributes, static_attribute_value
from hydromodels_dlm.dataset.scaler import (
    BasinScalerHub,
    FeatureScaler,
    fit_basin_scalers,
    load_scalers,
    resolve_feature_methods,
)

TRANSFER_SCALER_MODES = ('reuse', 'extend')


# ---------------------------------------------------------------------------
# Source scalers
# ---------------------------------------------------------------------------


def load_source_scalers(scaler_path):
    path = Path(scaler_path)
    if not path.is_file():
        raise FileNotFoundError(f'transfer scaler not found: {path}')
    source = load_scalers(path)
    if not source:
        raise ValueError(f'transfer scaler file is empty: {path}')
    return source


def source_static_scaler(source_hubs):
    for hub in source_hubs.values():
        if hub.static_scaler is not None:
            return hub.static_scaler
    return None


# ---------------------------------------------------------------------------
# Fit scalers
# ---------------------------------------------------------------------------


def fit_basin_xy_scaler(loader, basin_id, static_scaler):
    bid = str(basin_id)
    scaler_params = loader.config.get('scaler_params')
    if not isinstance(scaler_params, dict):
        raise ValueError('data_cfgs.scaler_params is required')

    x_methods = resolve_feature_methods(loader.input_keys, scaler_params)
    y_methods = resolve_feature_methods([loader.obs_key], scaler_params)
    if 'prcp_log1p_zscore' in y_methods and 'P' not in loader.input_keys:
        raise ValueError('prcp_log1p_zscore on Q requires P in dynamic_inputs')

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
    return BasinScalerHub(x_scaler, y_scaler, static_scaler=static_scaler)


def fit_static_scaler_for_basins(loader, basin_ids):
    scaler_params = loader.config.get('scaler_params')
    basin_ids = [str(bid) for bid in basin_ids]
    attributes = load_attributes(loader.root, loader.static_keys, basin_ids)
    methods = resolve_feature_methods(loader.static_keys, scaler_params)
    rows = np.stack(
        [
            [static_attribute_value(attributes, bid, key) for key in loader.static_keys]
            for bid in basin_ids
        ],
        axis=0,
    )
    return FeatureScaler.fit(rows, loader.static_keys, methods)


# ---------------------------------------------------------------------------
# Transfer scaler modes
# ---------------------------------------------------------------------------


def reuse_transfer_scalers(loader, source_scaler_path, basin_ids):
    source = load_source_scalers(source_scaler_path)
    layout = get_input_layout(loader.input_layout_name)
    static_scaler = source_static_scaler(source)
    if layout['uses_static'] and static_scaler is None:
        raise ValueError(
            'source scalers have no static_scaler but input_layout uses static features'
        )

    hubs = {}
    for bid in (str(b) for b in basin_ids):
        if bid in source:
            hubs[bid] = source[bid]
        else:
            hubs[bid] = fit_basin_xy_scaler(loader, bid, static_scaler)
    return hubs


def extend_transfer_scalers(loader, source_scaler_path, basin_ids):
    source = load_source_scalers(source_scaler_path)
    source_basins = sorted(source)
    targets = [str(bid) for bid in basin_ids]
    event_map = getattr(loader, 'event_map', None)
    missing_source = (
        event_map is not None
        and any(bid not in event_map for bid in source_basins)
    )

    if not missing_source:
        all_basins = list(dict.fromkeys(source_basins + targets))
        return fit_basin_scalers(loader, all_basins)

    static_scaler = fit_static_scaler_for_basins(
        loader, list(dict.fromkeys(source_basins + targets))
    )
    hubs = {}
    for bid in targets:
        if bid in source:
            hub = source[bid]
            hubs[bid] = BasinScalerHub(
                hub.x_scaler, hub.y_scaler, static_scaler=static_scaler
            )
        else:
            hubs[bid] = fit_basin_xy_scaler(loader, bid, static_scaler)
    return hubs


def build_transfer_scalers(loader, source_scaler_path, basin_ids, scaler_mode):
    mode = str(scaler_mode).lower()
    if mode not in TRANSFER_SCALER_MODES:
        raise ValueError(
            f'transfer_scaler_mode must be one of {TRANSFER_SCALER_MODES}, got {mode!r}'
        )
    if mode == 'extend':
        return extend_transfer_scalers(loader, source_scaler_path, basin_ids)
    return reuse_transfer_scalers(loader, source_scaler_path, basin_ids)


# ---------------------------------------------------------------------------
# Checkpoint and init
# ---------------------------------------------------------------------------


def check_transfer_checkpoint(loader, model_path, scalers):
    payload = torch.load(model_path, map_location='cpu', weights_only=False)
    n_inputs = int(payload['n_inputs'])
    expected = loader.model_input_size(scalers)
    if n_inputs != expected:
        raise ValueError(
            f'transfer checkpoint n_inputs={n_inputs} != layout '
            f'{loader.input_layout_name!r} ({expected} features). '
            'Use the same input_layout and model_hyperparam as the source experiment.'
        )


def load_init_weights(model, model_path, loader, scalers, device):
    check_transfer_checkpoint(loader, model_path, scalers)
    payload = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(payload['model_state'])


def build_transfer_init(loader, model_path, scaler_path, basin_ids, scaler_mode):
    scalers = build_transfer_scalers(loader, scaler_path, basin_ids, scaler_mode)
    check_transfer_checkpoint(loader, model_path, scalers)
    return model_path, scalers
