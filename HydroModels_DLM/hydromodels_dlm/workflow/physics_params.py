import numpy as np
import torch

from hydromodels_dlm.config.run_config import dump_json
from hydromodels_dlm.model.physics_guided import param_names, predict_physical_params
from hydromodels_dlm.workflow.evaluate import (
    SLIDING_BATCH_SIZE,
    merge_eval_series,
    stack_eval_windows,
    to_numpy,
    window_start_indices,
)


# ---------------------------------------------------------------------------
# Rolling parameter export
# ---------------------------------------------------------------------------


def format_param_time(value):
    if hasattr(value, 'strftime'):
        return value.strftime('%Y-%m-%d %H:%M:%S')
    text = np.datetime_as_string(np.asarray(value, dtype='datetime64[m]'), unit='m')
    if isinstance(text, np.ndarray):
        return str(text.item())
    return str(text)


def predict_physics_param_windows(
    model,
    lstm_seq,
    device,
    *,
    warmup,
    forecast=1,
    batch_size=SLIDING_BATCH_SIZE,
    starts=None,
):
    lstm_np = to_numpy(lstm_seq)
    starts = window_start_indices(lstm_np.shape[0], warmup, forecast, starts)
    names = list(param_names(model.pbm_name))
    if not starts:
        return np.zeros((0, len(names)), dtype=np.float64), names

    out = np.empty((len(starts), len(names)), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(starts), int(batch_size)):
            batch = starts[offset : offset + int(batch_size)]
            l_batch = stack_eval_windows(lstm_np, batch, warmup, forecast)
            params, _ = predict_physical_params(
                model.param_predictor,
                torch.from_numpy(l_batch.astype(np.float32)).to(device),
                model.pbm_name,
            )
            out[offset : offset + len(batch), :] = params.cpu().numpy()
    return out, names


def summarize_params_matrix(params_matrix, names, *, round_digits=3, reducer='median'):
    if params_matrix.size == 0:
        return {name: float('nan') for name in names}
    if reducer == 'mean':
        values = np.nanmean(params_matrix, axis=0)
    else:
        values = np.nanmedian(params_matrix, axis=0)
    out = {}
    for i, name in enumerate(names):
        value = float(values[i])
        if round_digits is not None:
            value = round(value, int(round_digits))
        out[name] = value
    return out


def write_best_params_json(
    path,
    *,
    basin_id,
    model_name,
    params_summary,
):
    payload = {
        'basin_id': str(basin_id),
        'model_name': str(model_name),
        'params': params_summary,
    }
    dump_json(payload, path)
    return payload


# ---------------------------------------------------------------------------
# Long-term parameter export
# ---------------------------------------------------------------------------


def collect_longterm_period_params(
    model,
    loader,
    basin_id,
    period,
    scalers,
    model_name,
    device,
):
    _drivers, lstm_dyn, x_static, _y, times, _lead = loader.build_rolling_physics_tensors(
        basin_id,
        period,
        scalers,
        model_name,
    )
    lstm_seq = merge_eval_series(lstm_dyn, x_static)
    params_matrix, param_names_out = predict_physics_param_windows(
        model,
        lstm_seq,
        device,
        warmup=loader.warmup_length,
        forecast=loader.forecast_length,
    )
    n = min(len(times), params_matrix.shape[0])
    return params_matrix[:n], param_names_out


# ---------------------------------------------------------------------------
# Flood-event parameter export
# ---------------------------------------------------------------------------


def iter_flood_event_param_windows(
    model,
    loader,
    basin_id,
    period,
    scalers,
    model_name,
    device,
):
    warmup = int(loader.warmup_length)
    forecast = int(loader.forecast_length)

    for bundle in loader.iter_period_bundles(basin_id, period):
        _drivers, lstm_dyn, x_static, _, _, _ = loader.build_event_physics_tensors(
            basin_id,
            bundle,
            scalers,
            model_name,
        )
        eval_idx = np.asarray(bundle['eval_idx'], dtype=np.int64)
        if eval_idx.size == 0:
            continue
        lstm_seq = merge_eval_series(lstm_dyn, x_static)
        params_matrix, param_names_out = predict_physics_param_windows(
            model,
            lstm_seq,
            device,
            warmup=warmup,
            forecast=forecast,
            starts=(warmup + eval_idx).tolist(),
        )
        yield {
            'event_id': str(bundle['event_id']),
            'param_names': param_names_out,
            'params_matrix': params_matrix,
            'sim_times': bundle['series'].index[warmup + eval_idx],
        }


def collect_flood_event_period_params_timeseries(
    model,
    loader,
    basin_id,
    period,
    scalers,
    model_name,
    device,
):
    rows = []
    param_names_out = list(param_names(model.pbm_name))
    bid = str(basin_id)
    period_key = str(period).lower()

    for item in iter_flood_event_param_windows(
        model, loader, basin_id, period, scalers, model_name, device
    ):
        param_names_out = item['param_names']
        for i, t in enumerate(item['sim_times']):
            row = {
                'period': period_key,
                'basin_id': bid,
                'event_id': item['event_id'],
                'time': format_param_time(t),
            }
            for j, name in enumerate(param_names_out):
                row[name] = float(item['params_matrix'][i, j])
            rows.append(row)
    return rows, param_names_out


def collect_flood_event_period_params(
    model,
    loader,
    basin_id,
    period,
    scalers,
    model_name,
    device,
):
    event_means = []
    param_names_out = list(param_names(model.pbm_name))

    for item in iter_flood_event_param_windows(
        model, loader, basin_id, period, scalers, model_name, device
    ):
        param_names_out = item['param_names']
        event_means.append(np.nanmean(item['params_matrix'], axis=0))

    if not event_means:
        empty_names = list(param_names(model.pbm_name))
        return np.zeros((0, len(empty_names)), dtype=np.float64), empty_names
    return np.vstack(event_means), param_names_out
