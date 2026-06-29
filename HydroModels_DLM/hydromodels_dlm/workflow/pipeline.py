import sys
import time
import traceback
from pathlib import Path

from hydromodels_dlm import configure_runtime
from hydromodels_dlm.config.run_config import (
    data_cfgs,
    dump_json,
    evaluation_metrics,
    finalize_config,
    merge_config,
    output_dir,
    save_basin_run_configs,
    skip_train,
    training_strategy,
    transfer_finetune,
    transfer_model_path,
    transfer_scaler_mode,
    transfer_scaler_path,
    transfer_zero_shot,
    weight_path,
)
from hydromodels_dlm.dataset.data_loader import create_data_loader
from hydromodels_dlm.dataset.data_preprocess import log_longterm_load, log_longterm_periods
from hydromodels_dlm.dataset.data_source import is_flood_event_id, resolve_basin_ids
from hydromodels_dlm.dataset.scaler import load_scalers, save_scalers
from hydromodels_dlm.utils.device import resolve_device
from hydromodels_dlm.utils.runtime_env import log_runtime_environment
from hydromodels_dlm.utils.logging import (
    get_logger,
    log_detail,
    log_section,
    progress_iter,
    save_basin_log,
    setup_logging,
    start_basin_log,
    stop_basin_log,
)
from hydromodels_dlm.model.physics_guided import is_physics_model
from hydromodels_dlm.workflow.artifacts import (
    BEST_CHECKPOINT,
    BEST_METRICS,
    NORMALIZATION_SCALER,
    PERIOD_METRICS_FILES,
    TRAINING_LOG,
    append_regional_eval_summary,
    clear_experiment_metrics_summary,
    prune_stale_basin_dirs,
    publish_experiment_params_summary,
    publish_experiment_period_metrics,
)
from hydromodels_dlm.workflow.evaluate import evaluate_basin
from hydromodels_dlm.workflow.train import train_model
from hydromodels_dlm.workflow.transfer import build_transfer_init

configure_runtime()


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_eval_metrics(metrics_by_period, metric_names):
    parts = []
    for period in ('train', 'valid', 'test'):
        row = (metrics_by_period or {}).get(period)
        if not row:
            continue
        period_parts = ', '.join(
            f'{name.upper()}={row[name.upper()]:.4f}'
            for name in metric_names
            if name.upper() in row
        )
        if period_parts:
            parts.append(f'{period}: {period_parts}')
    return ' | '.join(parts) if parts else 'no metrics'


def log_output_layout(exp_dir, *, regional=False):
    log = get_logger('pipeline')
    exp_dir = Path(exp_dir)
    if regional:
        log_detail(log, 'Shared model: %s/%s', exp_dir, BEST_CHECKPOINT)
        log_detail(log, 'Shared scaler: %s/%s', exp_dir, NORMALIZATION_SCALER)
        log_detail(log, 'Training log: %s/%s', exp_dir, TRAINING_LOG)
        log_detail(log, 'Experiment config: %s/run_config.json', exp_dir)
        summary = ', '.join(PERIOD_METRICS_FILES.values())
        log_detail(log, 'Metrics summary: %s/{%s}', exp_dir, summary)
        log_detail(
            log,
            'Per basin: %s/<basin_id>/{%s, %s, timeseries.csv, run_config.json}',
            exp_dir,
            BEST_METRICS,
            TRAINING_LOG,
        )
    else:
        log_detail(log, 'Model: %s/%s', exp_dir, BEST_CHECKPOINT)
        log_detail(log, 'Scaler: %s/%s', exp_dir, NORMALIZATION_SCALER)
        log_detail(log, 'Training log: %s/%s', exp_dir, TRAINING_LOG)
        log_detail(log, 'Metrics: %s/%s', exp_dir, BEST_METRICS)


def log_basin_results(basin_id, eval_info, metric_names):
    log = get_logger('pipeline')
    log_section(log, f'▶ Basin Results  [{basin_id}]')
    metrics = (eval_info or {}).get('metrics', {})
    names = [str(m).upper() for m in metric_names]
    for period in ('train', 'valid', 'test'):
        row = metrics.get(period)
        if not row:
            continue
        parts = ', '.join(f'{name}={row[name]:.4f}' for name in names if name in row)
        if parts:
            log_detail(log, '[%s] %s', period, parts)


# ---------------------------------------------------------------------------
# Transfer
# ---------------------------------------------------------------------------


def log_transfer_source(cfg, log, label):
    log_section(log, f'▶ Transfer {label}')
    log_detail(log, 'Source model: %s', transfer_model_path(cfg))
    log_detail(log, 'Source scalers: %s', transfer_scaler_path(cfg))
    log_detail(log, 'Scaler mode: %s', transfer_scaler_mode(cfg))


def resolve_finetune_init(cfg, loader, basin_ids, exp_dir, log):
    if not transfer_finetune(cfg):
        return None, None
    log_transfer_source(cfg, log, 'Finetune')
    model_path, scalers = build_transfer_init(
        loader,
        transfer_model_path(cfg),
        transfer_scaler_path(cfg),
        basin_ids,
        transfer_scaler_mode(cfg),
    )
    save_scalers(Path(exp_dir) / NORMALIZATION_SCALER, scalers)
    return model_path, scalers


def setup_eval_context(cfg, loader, basin_ids, exp_dir, log):
    exp_dir = Path(exp_dir)
    if transfer_zero_shot(cfg):
        log_transfer_source(cfg, log, 'Zero-Shot')
        model_path, scalers = build_transfer_init(
            loader,
            transfer_model_path(cfg),
            transfer_scaler_path(cfg),
            basin_ids,
            transfer_scaler_mode(cfg),
        )
        save_scalers(exp_dir / NORMALIZATION_SCALER, scalers)
        dump_json(cfg, exp_dir / 'run_config.json')
        return model_path, scalers

    wp = weight_path(cfg) or str(exp_dir / BEST_CHECKPOINT)
    scaler_path = exp_dir / NORMALIZATION_SCALER
    scalers = load_scalers(scaler_path) if scaler_path.is_file() else None
    return wp, scalers


# ---------------------------------------------------------------------------
# Experiment entry
# ---------------------------------------------------------------------------


def log_dataset_context(loader, basin_label, cfg, output_dir, algorithm, device=None):
    input_path = data_cfgs(cfg).get('input_path') or loader.root
    loader.log_load_context(
        basin_label,
        model=cfg['model_cfgs']['model_name'],
        algorithm=algorithm,
        output_dir=str(output_dir),
        input_path=str(input_path),
        device=device,
    )


def log_regional_flood_setup(log, cfg, loader, basin_ids, exp_dir, *, algorithm, device=None):
    log_section(log, '▶ Run Setup')
    log_runtime_environment(log, device=device)
    log_detail(
        log,
        'Model: %s  |  Algorithm: %s  |  Basins: %s',
        cfg['model_cfgs']['model_name'],
        algorithm,
        [str(b) for b in basin_ids],
    )
    input_path = data_cfgs(cfg).get('input_path') or loader.root
    log_detail(log, 'Input Path: %s  |  Output Path: %s', input_path, exp_dir)
    log_longterm_periods(loader.config, log=log)


def run_script_experiment(overrides):
    setup_logging()
    cfg = merge_config(overrides)
    cfg = finalize_config(resolve_basin_ids(cfg['data_cfgs']), cfg)

    exp_dir = output_dir(cfg)
    strategy = training_strategy(cfg)
    metric_names = evaluation_metrics(cfg)
    do_train = not skip_train(cfg)
    ext_weights = weight_path(cfg)
    device = resolve_device(cfg['training_cfgs'].get('device'))

    loader = create_data_loader(data_cfgs(cfg))
    basin_ids = loader.basin_ids
    clear_experiment_metrics_summary(exp_dir)
    prune_stale_basin_dirs(exp_dir, basin_ids)

    if strategy == 'RegionalTrain':
        run_regional_train(cfg, loader, basin_ids, exp_dir, do_train, device, metric_names)
    else:
        run_local_train(
            cfg, loader, basin_ids, exp_dir, do_train, ext_weights, device, metric_names
        )

    return {'status': 'completed', 'output_dir': exp_dir}


def run_script_main(overrides, *, logger_name='script'):
    setup_logging()
    log = get_logger(logger_name)
    try:
        run_script_experiment(overrides)
    except Exception as exc:
        log.error('run failed: %s', exc)
        traceback.print_exc()
        sys.exit(1)


# ---------------------------------------------------------------------------
# LocalTrain
# ---------------------------------------------------------------------------


def run_local_train(cfg, loader, basin_ids, exp_dir, do_train, ext_weights, device, metric_names):
    log = get_logger('pipeline')
    Path(exp_dir).mkdir(parents=True, exist_ok=True)
    multi_basin = len(basin_ids) > 1

    zero_shot_wp = None
    zero_shot_scalers = None
    if transfer_zero_shot(cfg):
        zero_shot_wp, zero_shot_scalers = setup_eval_context(
            cfg, loader, basin_ids, exp_dir, log
        )

    basin_loop = progress_iter(
        basin_ids,
        desc='LocalTrain basins',
        enabled=multi_basin,
        unit='basin',
        leave=True,
    )
    for basin_id in basin_loop:
        bid = str(basin_id)
        if multi_basin:
            basin_loop.set_description(f'LocalTrain {bid}')

        basin_dir = Path(exp_dir) / bid
        start_basin_log()
        log_dataset_context(
            loader,
            bid,
            cfg,
            basin_dir,
            'Adam',
            device=device,
        )
        loader.log_period_missing_values(bid)

        init_weights, init_scalers = resolve_finetune_init(
            cfg, loader, [bid], basin_dir, log
        )
        if init_scalers is not None:
            basin_dir.mkdir(parents=True, exist_ok=True)

        train_info = None
        if do_train:
            train_info = train_model(
                cfg,
                loader,
                [bid],
                basin_dir,
                device=device,
                init_weight_path=init_weights,
                scalers=init_scalers,
            )
            loader.series_cache.pop(bid, None)

        if transfer_finetune(cfg):
            scaler_path = basin_dir / NORMALIZATION_SCALER
            wp = str(basin_dir / BEST_CHECKPOINT)
            scalers = load_scalers(scaler_path) if scaler_path.is_file() else init_scalers
        else:
            wp = zero_shot_wp or ext_weights or str(basin_dir / BEST_CHECKPOINT)
            scalers = zero_shot_scalers

        eval_info = evaluate_basin(
            cfg,
            loader,
            bid,
            basin_dir,
            metric_names,
            weight_path=wp,
            scalers=scalers,
            device=device,
        )
        log_basin_results(bid, eval_info, metric_names)
        save_basin_log(basin_dir / TRAINING_LOG)
        stop_basin_log()
        save_basin_run_configs(cfg, exp_dir, [bid])

        if multi_basin:
            metrics = (eval_info or {}).get('metrics', {})
            log_detail(log, '[%s] %s', bid, format_eval_metrics(metrics, metric_names))
        loader.series_cache.pop(bid, None)

    publish_experiment_period_metrics(exp_dir, basin_ids)
    if is_physics_model(cfg['model_cfgs']['model_name']):
        publish_experiment_params_summary(exp_dir, basin_ids)


# ---------------------------------------------------------------------------
# RegionalTrain
# ---------------------------------------------------------------------------


def run_regional_train(cfg, loader, basin_ids, exp_dir, do_train, device, metric_names):
    log = get_logger('pipeline')
    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    start_basin_log()
    regional_log_active = True
    t0_scheme = time.perf_counter()
    try:
        if is_flood_event_id(loader.file_ids[0]):
            log_regional_flood_setup(
                log,
                cfg,
                loader,
                basin_ids,
                exp_dir,
                algorithm='Adam (regional)',
                device=device,
            )
        else:
            log_longterm_load(
                loader.root,
                f'{len(basin_ids)} basins',
                loader.config,
                model=cfg['model_cfgs']['model_name'],
                algorithm='Adam (regional)',
                output_dir=str(exp_dir),
                input_path=data_cfgs(cfg).get('input_path') or loader.root,
                device=device,
            )
        loader.log_regional_missing_values(basin_ids)

        init_weights, init_scalers = resolve_finetune_init(cfg, loader, basin_ids, exp_dir, log)

        train_info = None
        if do_train:
            dump_json(cfg, exp_dir / 'run_config.json')
            train_info = train_model(
                cfg,
                loader,
                basin_ids,
                exp_dir,
                device=device,
                init_weight_path=init_weights,
                scalers=init_scalers,
            )
            save_basin_log(exp_dir / TRAINING_LOG)
            stop_basin_log()
            regional_log_active = False
            log_output_layout(exp_dir, regional=True)
        else:
            save_basin_log(exp_dir / TRAINING_LOG)
            stop_basin_log()
            regional_log_active = False

        wp, scalers = setup_eval_context(cfg, loader, basin_ids, exp_dir, log)

        log_section(log, '▶ Evaluate')
        log_output_layout(exp_dir, regional=True)

        basin_iter = progress_iter(
            basin_ids,
            desc='Evaluate basins',
            unit='basin',
            leave=True,
        )
        for basin_id in basin_iter:
            bid = str(basin_id)
            basin_iter.set_description(f'Evaluate {bid}')

            basin_dir = exp_dir / bid
            start_basin_log()
            log_dataset_context(
                loader,
                bid,
                cfg,
                basin_dir,
                'Adam (regional)',
                device=device,
            )
            loader.log_period_missing_values(bid, force=True)

            eval_info = evaluate_basin(
                cfg,
                loader,
                bid,
                exp_dir,
                metric_names,
                weight_path=wp,
                scalers=scalers,
                device=device,
                eval_output_dir=basin_dir,
            )
            log_basin_results(bid, eval_info, metric_names)
            save_basin_log(basin_dir / TRAINING_LOG)
            stop_basin_log()
            save_basin_run_configs(cfg, exp_dir, [bid])

            metrics = (eval_info or {}).get('metrics', {})
            test_nse = metrics.get('test', {}).get('NSE')
            if test_nse is not None:
                basin_iter.set_postfix(test_NSE=f'{test_nse:.4f}')
            log_detail(
                log,
                '[%s] -> %s | %s',
                bid,
                basin_dir,
                format_eval_metrics(metrics, metric_names),
            )
            loader.series_cache.pop(bid, None)

        publish_experiment_period_metrics(exp_dir, basin_ids)
        if is_physics_model(cfg['model_cfgs']['model_name']):
            publish_experiment_params_summary(exp_dir, basin_ids)
        append_regional_eval_summary(
            exp_dir,
            metric_names,
            train_info=train_info,
            elapsed_s=time.perf_counter() - t0_scheme,
        )
    finally:
        if regional_log_active:
            stop_basin_log()
