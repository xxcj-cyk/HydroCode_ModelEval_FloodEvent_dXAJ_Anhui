import csv
import json
import time
from copy import deepcopy
from pathlib import Path

from hydromodels_pbm.config.run_config import merge_config, prepare_config, save_basin_run_configs
from hydromodels_pbm.dataset.data_loader import (
    FloodEventDataLoader,
    LongTermDataLoader,
    create_data_loader,
)
from hydromodels_pbm.dataset.data_preprocess import log_flood_load, log_longterm_load
from hydromodels_pbm.dataset.data_source import glob_basin_ids, load_basin_pair_list, resolve_basin_ids
from hydromodels_pbm.utils.logging import (
    format_elapsed,
    get_logger,
    log_section,
    save_basin_log,
    start_basin_log,
    stop_basin_log,
)
from hydromodels_pbm.utils.runtime_env import warmup_numba_runtime
from hydromodels_pbm.workflow.calibrate import calibrate_basin
from hydromodels_pbm.workflow.evaluate import (
    evaluate_basin,
    evaluation_periods,
    load_basin_params,
    read_params_payload,
    resolve_params_path,
    save_experiment_params_summary,
)
from hydromodels_pbm.workflow.metrics_artifacts import (
    publish_experiment_period_metrics,
    sort_period_keys,
)
from hydromodels_pbm.workflow.simulator import (
    HydroSimulator,
    check_model_inputs,
    normalize_param_dict,
)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def log_basin_results(basin_id, cal_info, eval_info, metric_names, elapsed):
    log = get_logger('report')
    metrics = [str(m).upper() for m in metric_names]
    log_section(log, f'▶ Basin Results  [{basin_id}]')
    log.info('  Time %s', format_elapsed(elapsed))
    if cal_info:
        phys = cal_info.get('params_physical') or {}
        if phys:
            phys_fmt = {k: round(float(v), 3) for k, v in phys.items()}
            log.info('  Params: %s', json.dumps(phys_fmt, ensure_ascii=False))
        if cal_info.get('objective_value') is not None:
            log.info('  Calib objective: %.3f', cal_info['objective_value'])
    for period in sort_period_keys((eval_info or {}).get('metrics') or {}):
        row = eval_info['metrics'][period]
        parts = ', '.join(f'{name}={row[name]}' for name in metrics if name in row)
        log.info('  [%s] %s', period, parts)


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------


def run_script_experiment(overrides, save_summary=True):
    cfg = merge_config(overrides)
    cfg = prepare_config(cfg, resolve_basin_ids(cfg['data_cfgs']))

    data_cfgs = cfg['data_cfgs']
    training_cfgs = cfg['training_cfgs']
    evaluation_cfgs = cfg['evaluation_cfgs']

    calib_dir = Path(data_cfgs['output_dir'])
    calib_dir.mkdir(parents=True, exist_ok=True)

    model_name = cfg['model_cfgs']['model_name']
    metric_names = list(evaluation_cfgs['metrics'])
    params_json = evaluation_cfgs.get('params_json')
    init_params_json = evaluation_cfgs.get('init_params_json')
    source_basin_id = evaluation_cfgs.get('source_basin_id')

    loader = create_data_loader(deepcopy(data_cfgs))
    check_model_inputs(model_name, loader.input_keys)

    algo_params = training_cfgs.get('algorithm_params', {})
    warmup_numba_runtime(
        model_name,
        data_cfgs.get('warmup_length', 365),
        algo_params.get('random_seed'),
    )

    params_payload = None
    init_norm = None
    if params_json:
        params_payload = read_params_payload(params_json, source_basin_id)
    elif init_params_json:
        init_norm = normalize_param_dict(
            read_params_payload(init_params_json, source_basin_id)['params'],
            model_name,
        )

    simulator = HydroSimulator(model_name)
    cal_basins = {}
    eval_basins = {}

    for basin_id in loader.basin_ids:
        t0 = time.perf_counter()
        start_basin_log()
        key = str(basin_id)
        cal_info = None

        if params_json is None:
            cal_info = calibrate_basin(
                basin_id,
                calib_dir,
                model_name,
                training_cfgs['algorithm_name'],
                training_cfgs['objective_function'],
                algo_params,
                loader,
                init_norm=init_norm,
            )
            cal_basins[key] = cal_info
        else:
            setup_kw = {
                'model': model_name,
                'algorithm': training_cfgs['algorithm_name'],
                'output_dir': str(calib_dir),
                'input_path': str(resolve_params_path(params_json)),
            }
            if isinstance(loader, FloodEventDataLoader):
                log_flood_load(
                    loader.root,
                    basin_id,
                    loader.list_period_event_ids(basin_id, 'calib'),
                    loader.list_period_event_ids(basin_id, 'valid'),
                    **setup_kw,
                )
            elif isinstance(loader, LongTermDataLoader):
                log_longterm_load(loader.root, basin_id, data_cfgs, **setup_kw)
            else:
                raise TypeError(f'unsupported loader type: {type(loader)!r}')

        loader.series_cache.clear()
        params = load_basin_params(calib_dir, basin_id, params_payload)
        eval_info = evaluate_basin(
            basin_id,
            calib_dir,
            loader,
            evaluation_periods(loader, basin_id),
            params,
            metric_names,
            simulator,
        )
        eval_basins[key] = eval_info
        loader.series_cache.clear()

        save_basin_run_configs(cfg, calib_dir, [basin_id])
        log_basin_results(key, cal_info, eval_info, metric_names, time.perf_counter() - t0)
        save_basin_log(calib_dir / key / 'calibration_log.txt')
        stop_basin_log()

    if save_summary:
        publish_experiment_period_metrics(calib_dir, loader.basin_ids)
        save_experiment_params_summary(calib_dir, loader.basin_ids)

    return {
        'status': 'completed',
        'calib_dir': str(calib_dir),
        'calibration': {'basins': cal_basins} if cal_basins else None,
        'evaluation': {'calib_dir': str(calib_dir), 'basins': eval_basins},
    }


# ---------------------------------------------------------------------------
# Pair-list runs
# ---------------------------------------------------------------------------


def save_pair_list_csv(output_dir, pairs, param_col, param_value):
    path = output_dir / (
        'param_transplant_list.csv'
        if param_col == 'params_json'
        else 'param_recalibrate_list.csv'
    )
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['target_basin_id', 'source_basin_id', param_col])
        for target_id, source_id in pairs:
            writer.writerow([target_id, source_id, param_value])


def run_pair_list_experiment(overrides):
    cfg = merge_config(overrides)
    evaluation_cfgs = cfg['evaluation_cfgs']
    params_json = evaluation_cfgs.get('params_json')
    init_params_json = evaluation_cfgs.get('init_params_json')
    transplant_list_csv = evaluation_cfgs.get('transplant_list_csv')

    if not transplant_list_csv:
        raise ValueError('pair-list run requires evaluation_cfgs.transplant_list_csv')
    if bool(params_json) == bool(init_params_json):
        raise ValueError(
            'pair-list run requires exactly one of params_json or init_params_json'
        )

    output_dir = Path(cfg['data_cfgs']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs = load_basin_pair_list(transplant_list_csv)

    if params_json:
        save_pair_list_csv(output_dir, pairs, 'params_json', params_json)
    else:
        save_pair_list_csv(output_dir, pairs, 'init_params_json', init_params_json)

    for target_id, source_id in pairs:
        run_cfg = deepcopy(cfg)
        run_cfg['evaluation_cfgs']['source_basin_id'] = source_id
        run_cfg['data_cfgs']['basin_ids'] = glob_basin_ids(
            cfg['data_cfgs']['input_path'],
            f'{target_id}_*.csv',
        )
        run_script_experiment(run_cfg, save_summary=False)

    target_ids = [target_id for target_id, _ in pairs]
    publish_experiment_period_metrics(output_dir, target_ids)
    save_experiment_params_summary(output_dir, target_ids)

    return {
        'status': 'completed',
        'calib_dir': str(output_dir),
        'target_basin_ids': target_ids,
    }


def run_transplant_experiment(overrides):
    if not (overrides.get('evaluation_cfgs') or {}).get('params_json'):
        raise ValueError('transplant requires evaluation_cfgs.params_json')
    return run_pair_list_experiment(overrides)


def run_recalibrate_experiment(overrides):
    if not (overrides.get('evaluation_cfgs') or {}).get('init_params_json'):
        raise ValueError('recalibrate requires evaluation_cfgs.init_params_json')
    return run_pair_list_experiment(overrides)
