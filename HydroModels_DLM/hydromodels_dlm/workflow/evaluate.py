import csv
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hydromodels_dlm.dataset.input_layout import merge_model_input_torch
from hydromodels_dlm.dataset.scaler import load_scalers
from hydromodels_dlm.dataset.data_loader import FloodEventDataLoader
from hydromodels_dlm.config.run_config import eval_mode
from hydromodels_dlm.model.physics_guided import (
    PhysicsGuidedModel,
    apply_cemaneige_climatology,
    apply_pbm_grad_steps,
    predict_physical_params,
    sequential_physics_qsim,
    simulate_sliding_physics,
)
from hydromodels_dlm.utils.device import resolve_device
from hydromodels_dlm.workflow.artifacts import (
    BEST_CHECKPOINT,
    BEST_PARAMS,
    EVENT_METRICS,
    NORMALIZATION_SCALER,
    PERIOD_ORDER,
    aggregate_event_metrics,
    format_metric_value,
    write_basin_metrics,
    write_basin_metrics_longterm,
)
from hydromodels_dlm.workflow.metrics import METRIC_REGISTRY


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------


def torch_load_checkpoint(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_checkpoint(
    weight_path,
    cfg,
    n_inputs,
    device,
    *,
    forecast_length=1,
    loader=None,
    basin_id=None,
    warmup_length=365,
):
    from hydromodels_dlm.workflow.train import build_model

    payload = torch_load_checkpoint(weight_path, device)
    cfg_load = deepcopy(cfg)
    if payload.get('model_hyperparam'):
        cfg_load['model_cfgs']['model_hyperparam'] = payload['model_hyperparam']
    model = build_model(
        cfg_load,
        n_inputs=n_inputs,
        warmup_length=loader.warmup_length if loader is not None else warmup_length,
    ).to(device)
    apply_pbm_grad_steps(model, cfg_load, forecast_length)
    if loader is not None and basin_id is not None and isinstance(model, PhysicsGuidedModel):
        apply_cemaneige_climatology(model, loader, [basin_id])
    model.load_state_dict(payload['model_state'])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Eval context
# ---------------------------------------------------------------------------


def resolve_basin_dir(out_dir, basin_id, eval_output_dir):
    out_dir = Path(out_dir)
    basin_id = str(basin_id)
    if eval_output_dir is not None:
        return Path(eval_output_dir)
    if out_dir.name == basin_id:
        return out_dir
    return out_dir / basin_id


def load_eval_context(
    cfg,
    loader,
    basin_id,
    out_dir,
    device,
    *,
    weight_path=None,
    scalers=None,
    eval_output_dir=None,
):
    device = device or resolve_device(cfg['training_cfgs'].get('device'))
    basin_id = str(basin_id)
    out_dir = Path(out_dir)
    basin_dir = resolve_basin_dir(out_dir, basin_id, eval_output_dir)
    basin_dir.mkdir(parents=True, exist_ok=True)

    wp = Path(weight_path or out_dir / BEST_CHECKPOINT)
    if not wp.is_file():
        raise FileNotFoundError(f'missing checkpoint: {wp}')

    scaler_path = out_dir / NORMALIZATION_SCALER
    if scalers is None:
        if not scaler_path.is_file():
            raise FileNotFoundError(f'missing scaler: {scaler_path}')
        scalers = load_scalers(scaler_path)

    ckpt = torch_load_checkpoint(wp, device)
    model = load_checkpoint(
        wp,
        cfg,
        int(ckpt['n_inputs']),
        device,
        forecast_length=loader.forecast_length,
        loader=loader,
        basin_id=basin_id,
    )
    return device, basin_id, basin_dir, scalers, model


# ---------------------------------------------------------------------------
# Series helpers
# ---------------------------------------------------------------------------


def to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def merge_eval_series(x_dyn, x_static):
    if x_static is None:
        return x_dyn
    return merge_model_input_torch(
        x_dyn.unsqueeze(0),
        x_static.unsqueeze(0),
    ).squeeze(0)


def align_rolling_eval(qsim_n, y_n, times, lead, *, cross_period_warmup):
    qsim_n = np.asarray(qsim_n, dtype=np.float64).ravel()
    y_n = np.asarray(y_n, dtype=np.float64).ravel()
    times = np.asarray(times)
    n_sim = len(qsim_n)
    if cross_period_warmup:
        y_n = y_n[:n_sim]
        times = times[:n_sim]
    else:
        y_n = y_n[lead : lead + n_sim]
        times = times[lead : lead + n_sim]
    if len(y_n) != n_sim:
        raise ValueError(
            f'rolling eval alignment failed: qsim={n_sim}, '
            f'qobs={len(y_n)}, lead={lead}, cross_period_warmup={cross_period_warmup}'
        )
    return qsim_n, y_n, times


def window_start_indices(n_time, warmup, forecast, starts=None):
    if starts is None:
        return list(range(int(warmup), n_time - int(forecast) + 1))
    return [int(s) for s in starts]


SLIDING_BATCH_SIZE = 128


def flood_eval_starts(warmup, eval_idx):
    return [int(warmup + idx) for idx in np.asarray(eval_idx, dtype=np.int64)]


def physics_kernel_size(model):
    return int(getattr(getattr(model, 'pb_core', None), 'kernel_size', 15))


def denormalize_streamflow(hub, qsim_n, y_n):
    qobs = hub.y_scaler.inverse_transform(y_n.reshape(-1, 1))[:, 0]
    qsim = hub.y_scaler.inverse_transform(qsim_n.reshape(-1, 1))[:, 0]
    return qsim, qobs


def stack_eval_windows(series, starts, warmup, forecast):
    arr = np.asarray(series)
    starts = np.asarray(starts, dtype=np.int64)
    win_len = int(warmup) + int(forecast)
    if starts.size == 0:
        return np.zeros((0, win_len, *arr.shape[1:]), dtype=arr.dtype)
    offsets = np.arange(-int(warmup), int(forecast), dtype=np.int64)
    return arr[starts[:, None] + offsets[None, :]]


# ---------------------------------------------------------------------------
# Streamflow simulation
# ---------------------------------------------------------------------------


def simulate_lstm_qsim(
    model,
    x,
    device,
    warmup,
    forecast,
    mode,
    *,
    starts=None,
    eval_idx=None,
    batch_size=SLIDING_BATCH_SIZE,
):
    model.eval()
    with torch.inference_mode():
        if mode == 'sequential':
            x_batch = x.unsqueeze(0).to(device) if x.dim() == 2 else x.to(device)
            pred = model(x_batch)[:, int(warmup) :, :].squeeze(0).cpu().numpy()
            qsim_n = np.asarray(pred[:, 0], dtype=np.float64)
            if int(forecast) > 1:
                qsim_n = qsim_n[: max(len(qsim_n) - int(forecast) + 1, 0)]
            if eval_idx is not None:
                return qsim_n[np.asarray(eval_idx, dtype=np.int64)]
            return qsim_n

        starts = [int(s) for s in starts]
        if not starts:
            return np.zeros(0, dtype=np.float64)

        x_np = to_numpy(x)
        out_idx = int(warmup) + int(forecast) - 1
        out = np.empty(len(starts), dtype=np.float64)
        for offset in range(0, len(starts), int(batch_size)):
            batch_starts = starts[offset : offset + int(batch_size)]
            x_batch = stack_eval_windows(x_np, batch_starts, warmup, forecast)
            pred = model(torch.from_numpy(x_batch.astype(np.float32)).to(device))
            out[offset : offset + len(batch_starts)] = pred[:, out_idx, 0].cpu().numpy()
        return out


def simulate_physics_qsim(
    model,
    drivers,
    lstm_seq,
    device,
    warmup,
    forecast,
    mode,
    *,
    starts=None,
    eval_idx=None,
    batch_size=SLIDING_BATCH_SIZE,
    kernel_size=15,
):
    if mode == 'sequential':
        return sequential_physics_qsim(
            model,
            drivers,
            lstm_seq,
            device,
            warmup=warmup,
            kernel_size=int(kernel_size),
            eval_idx=eval_idx,
        )

    starts = [int(s) for s in starts]
    if not starts:
        return np.zeros(0, dtype=np.float64)

    drivers_np = to_numpy(drivers)
    lstm_np = to_numpy(lstm_seq)
    ref = next(model.parameters())
    out = np.empty(len(starts), dtype=np.float64)
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(starts), int(batch_size)):
            batch_starts = starts[offset : offset + int(batch_size)]
            d_batch = stack_eval_windows(drivers_np, batch_starts, warmup, forecast)
            l_batch = stack_eval_windows(lstm_np, batch_starts, warmup, forecast)
            params, _ = predict_physical_params(
                model.param_predictor,
                torch.from_numpy(l_batch.astype(np.float32)).to(device),
                model.pbm_name,
            )
            out[offset : offset + len(batch_starts)] = simulate_sliding_physics(
                model.pb_core,
                d_batch,
                params.detach().cpu().numpy(),
                warmup=warmup,
                forecast=forecast,
                kernel_size=int(kernel_size),
                ref=ref,
            )
    return out


# ---------------------------------------------------------------------------
# Flood-event prediction
# ---------------------------------------------------------------------------


def predict_flood_event(
    model,
    loader,
    bundle,
    scalers,
    basin_id,
    device,
    model_name,
    mode,
):
    hub = scalers[str(basin_id)]
    warmup = loader.warmup_length
    forecast = int(loader.forecast_length)
    eval_idx = np.asarray(bundle['eval_idx'], dtype=np.int64)
    sim_times = bundle['series'].index[warmup + eval_idx]
    obs_idx = warmup + eval_idx
    starts = flood_eval_starts(warmup, eval_idx)

    if isinstance(model, PhysicsGuidedModel):
        drivers, lstm_dyn, x_static, y_tensor, _, _ = loader.build_event_physics_tensors(
            basin_id,
            bundle,
            scalers,
            model_name,
        )
        qsim = simulate_physics_qsim(
            model,
            drivers,
            merge_eval_series(lstm_dyn, x_static),
            device,
            warmup,
            forecast,
            mode,
            starts=starts,
            eval_idx=eval_idx,
            kernel_size=physics_kernel_size(model),
        )
        y_n = y_tensor.numpy()[obs_idx, 0]
        qobs = hub.y_scaler.inverse_transform(y_n.reshape(-1, 1))[:, 0]
        return qsim, qobs, sim_times

    x_dyn, x_static, y_tensor, _, _ = loader.build_event_tensors(
        basin_id,
        bundle,
        scalers,
    )
    qsim_n = simulate_lstm_qsim(
        model,
        merge_eval_series(x_dyn, x_static),
        device,
        warmup,
        forecast,
        mode,
        starts=starts,
        eval_idx=eval_idx,
    )
    y_n = y_tensor.numpy()[obs_idx, 0]
    return *denormalize_streamflow(hub, qsim_n, y_n), sim_times


# ---------------------------------------------------------------------------
# Metrics CSV
# ---------------------------------------------------------------------------


def format_time_label(value):
    if hasattr(value, 'strftime'):
        return value.strftime('%Y-%m-%d')
    text = np.datetime_as_string(np.asarray(value, dtype='datetime64[D]'), unit='D')
    if isinstance(text, np.ndarray):
        return str(text.item())
    return str(text)


def append_timeseries_rows(path, times, period, loader, basin_id, qobs, qsim):
    series = loader.period_series(basin_id, period)
    columns = ['time', 'period'] + list(loader.input_keys) + ['Qobs', 'Qsim']
    write_header = not path.is_file()
    with path.open('a', newline='', encoding='utf-8-sig') as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(columns)
        for i, t in enumerate(times):
            row = [format_time_label(t), period]
            for key in loader.input_keys:
                row.append(float(series.loc[t, key]))
            row.extend([float(qobs[i]), float(qsim[i])])
            writer.writerow(row)


def append_flood_timeseries_rows(
    path,
    times,
    period,
    loader,
    basin_id,
    event_id,
    bundle,
    qobs,
    qsim,
):
    series = bundle['series']
    columns = ['time', 'period', 'event_id'] + list(loader.input_keys) + ['Qobs', 'Qsim']
    write_header = not path.is_file()
    with path.open('a', newline='', encoding='utf-8-sig') as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(columns)
        for i, t in enumerate(times):
            row = [format_time_label(t), period, event_id]
            for key in loader.input_keys:
                row.append(float(series.loc[t, key]))
            row.extend([float(qobs[i]), float(qsim[i])])
            writer.writerow(row)


def write_event_metrics_csv(path, rows, metric_names):
    if not rows:
        return
    cols = ['period', 'basin_id', 'event_id'] + [str(m).upper() for m in metric_names]
    formatted = []
    for row in rows:
        out = dict(row)
        for key, value in list(out.items()):
            if key in ('period', 'basin_id', 'event_id'):
                continue
            out[key] = format_metric_value(key, value)
        formatted.append(out)
    pd.DataFrame(formatted)[cols].to_csv(
        Path(path),
        index=False,
        encoding='utf-8-sig',
    )


# ---------------------------------------------------------------------------
# Basin evaluation
# ---------------------------------------------------------------------------


def evaluate_flood_event_basin(
    cfg,
    loader,
    basin_id,
    out_dir,
    metric_names,
    *,
    weight_path=None,
    scalers=None,
    device=None,
    eval_output_dir=None,
):
    device, basin_id, basin_dir, scalers, model = load_eval_context(
        cfg,
        loader,
        basin_id,
        out_dir,
        device,
        weight_path=weight_path,
        scalers=scalers,
        eval_output_dir=eval_output_dir,
    )
    model_name = cfg['model_cfgs']['model_name']
    mode = eval_mode(cfg)

    ts_path = basin_dir / 'timeseries.csv'
    event_metrics_path = basin_dir / EVENT_METRICS
    for path in (ts_path, event_metrics_path):
        if path.is_file():
            path.unlink()

    period_stat_metrics = {}
    event_rows = []

    for period in PERIOD_ORDER:
        event_metric_rows = []
        for bundle in loader.iter_period_bundles(basin_id, period):
            qsim, qobs, times = predict_flood_event(
                model,
                loader,
                bundle,
                scalers,
                basin_id,
                device,
                model_name,
                mode,
            )
            metrics = {
                name.upper(): float(METRIC_REGISTRY[name.upper()](qsim, qobs))
                for name in metric_names
            }
            event_metric_rows.append(metrics)
            append_flood_timeseries_rows(
                ts_path,
                times,
                period,
                loader,
                basin_id,
                bundle['event_id'],
                bundle,
                qobs,
                qsim,
            )
            event_rows.append(
                {
                    'period': period,
                    'basin_id': basin_id,
                    'event_id': bundle['event_id'],
                    **metrics,
                }
            )

        if not event_metric_rows:
            continue
        period_stat_metrics[period] = aggregate_event_metrics(
            event_metric_rows,
            metric_names,
        )

    period_metrics = write_basin_metrics(basin_dir, basin_id, period_stat_metrics)
    write_event_metrics_csv(event_metrics_path, event_rows, metric_names)

    if isinstance(model, PhysicsGuidedModel):
        from hydromodels_dlm.workflow.physics_params import (
            collect_flood_event_period_params,
            summarize_params_matrix,
            write_best_params_json,
        )

        valid_params, valid_param_names = collect_flood_event_period_params(
            model,
            loader,
            basin_id,
            'valid',
            scalers,
            model_name,
            device,
        )
        if valid_params.size:
            write_best_params_json(
                basin_dir / BEST_PARAMS,
                basin_id=basin_id,
                model_name=model_name,
                params_summary=summarize_params_matrix(
                    valid_params,
                    valid_param_names,
                    reducer='mean',
                ),
            )

    return {'basin_id': basin_id, 'metrics': period_metrics, 'basin_dir': str(basin_dir)}


def evaluate_basin(
    cfg,
    loader,
    basin_id,
    out_dir,
    metric_names,
    *,
    weight_path=None,
    scalers=None,
    device=None,
    eval_output_dir=None,
):
    if isinstance(loader, FloodEventDataLoader):
        return evaluate_flood_event_basin(
            cfg,
            loader,
            basin_id,
            out_dir,
            metric_names,
            weight_path=weight_path,
            scalers=scalers,
            device=device,
            eval_output_dir=eval_output_dir,
        )

    device, basin_id, basin_dir, scalers, model = load_eval_context(
        cfg,
        loader,
        basin_id,
        out_dir,
        device,
        weight_path=weight_path,
        scalers=scalers,
        eval_output_dir=eval_output_dir,
    )
    model_name = cfg['model_cfgs']['model_name']
    hub = scalers[basin_id]
    mode = eval_mode(cfg)
    warmup = int(loader.warmup_length)
    forecast = int(loader.forecast_length)
    kernel_size = physics_kernel_size(model)

    ts_path = basin_dir / 'timeseries.csv'
    if ts_path.is_file():
        ts_path.unlink()

    train_params_matrix = None
    train_param_names = None
    period_metrics = {}
    if isinstance(model, PhysicsGuidedModel):
        from hydromodels_dlm.workflow.physics_params import (
            collect_longterm_period_params,
            summarize_params_matrix,
            write_best_params_json,
        )

    for period in PERIOD_ORDER:
        if isinstance(model, PhysicsGuidedModel):
            drivers, lstm_dyn, x_static, y_tensor, times, lead = loader.build_rolling_physics_tensors(
                basin_id, period, scalers, model_name
            )
            cross_period = loader.rolling_warmup_period(period) is not None
            starts = window_start_indices(drivers.shape[0], warmup, forecast)
            qsim = simulate_physics_qsim(
                model,
                drivers,
                merge_eval_series(lstm_dyn, x_static),
                device,
                warmup,
                forecast,
                mode,
                starts=starts,
                kernel_size=kernel_size,
            )
            y_n = y_tensor.numpy()[:, 0]
            qsim, y_n, times = align_rolling_eval(
                qsim,
                y_n,
                times,
                lead,
                cross_period_warmup=cross_period,
            )
            qobs = hub.y_scaler.inverse_transform(y_n.reshape(-1, 1))[:, 0]

            if period == 'train':
                train_params_matrix, train_param_names = collect_longterm_period_params(
                    model,
                    loader,
                    basin_id,
                    period,
                    scalers,
                    model_name,
                    device,
                )
        else:
            x_dyn, x_static, y_tensor, times, lead = loader.build_rolling_tensors(
                basin_id, period, scalers
            )
            cross_period = loader.rolling_warmup_period(period) is not None
            x = merge_eval_series(x_dyn, x_static)
            starts = window_start_indices(x.shape[0], warmup, forecast)
            qsim_n = simulate_lstm_qsim(
                model,
                x,
                device,
                warmup,
                forecast,
                mode,
                starts=starts,
            )
            y_n = y_tensor.numpy()[:, 0]
            qsim_n, y_n, times = align_rolling_eval(
                qsim_n,
                y_n,
                times,
                lead,
                cross_period_warmup=cross_period,
            )
            qsim, qobs = denormalize_streamflow(hub, qsim_n, y_n)

        period_metrics[period] = {
            name.upper(): float(METRIC_REGISTRY[name.upper()](qsim, qobs))
            for name in metric_names
        }
        append_timeseries_rows(
            ts_path, times, period, loader, basin_id, qobs, qsim
        )

    period_metrics = write_basin_metrics_longterm(basin_dir, basin_id, period_metrics)

    if isinstance(model, PhysicsGuidedModel) and train_params_matrix is not None:
        write_best_params_json(
            basin_dir / BEST_PARAMS,
            basin_id=basin_id,
            model_name=model_name,
            params_summary=summarize_params_matrix(
                train_params_matrix,
                train_param_names,
            ),
        )

    return {'basin_id': basin_id, 'metrics': period_metrics, 'basin_dir': str(basin_dir)}
