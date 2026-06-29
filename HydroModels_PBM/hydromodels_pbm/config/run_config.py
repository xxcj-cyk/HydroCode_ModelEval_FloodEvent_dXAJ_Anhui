from copy import deepcopy
from pathlib import Path

from hydromodels_pbm.config.model_config import MODEL_INPUT_KEYS, MODEL_PARAM_DICT
from hydromodels_pbm.utils.logging import write_json
from hydromodels_pbm.workflow.metrics import METRIC_REGISTRY
from hydromodels_pbm.workflow.objectives import LOSS_FUNCTIONS
from hydromodels_pbm.workflow.optimizers import ALGORITHM_RUNNERS


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def default_config():
    return {
        'data_cfgs': {
            'input_path': None,
            'output_dir': 'results/hydro_default',
            'basin_ids': None,
            'variables': {
                'dynamic_inputs': {
                    'P': 'precipitation',
                    'PET': 'potential evapotranspiration',
                },
                'dynamic_outputs': {'Q': 'streamflow'},
            },
            'warmup_length': 365,
            'calib_period': ['2010-01-01', '2018-12-31'],
            'valid_period': ['2019-01-01', '2022-12-31'],
        },
        'model_cfgs': {
            'model_name': 'XAJ',
        },
        'training_cfgs': {
            'algorithm_name': 'SCE-UA',
            'algorithm_params': {
                'rep': 1000,
                'ngs': 50,
                'kstop': 100,
                'peps': 0.1,
                'pcento': 0.1,
                'random_seed': 1234,
            },
            'objective_function': 'RMSE',
        },
        'evaluation_cfgs': {
            'metrics': ['NSE', 'KGE', 'RMSE', 'MBE'],
            'params_json': None,
            'init_params_json': None,
            'source_basin_id': None,
            'transplant_list_csv': None,
        },
    }


# ---------------------------------------------------------------------------
# Merge and prepare
# ---------------------------------------------------------------------------


def merge_nested_dict(base, patch):
    for key, value in patch.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            merge_nested_dict(base[key], value)
        else:
            base[key] = deepcopy(value)


def merge_config(overrides=None):
    cfg = deepcopy(default_config())
    if overrides:
        merge_nested_dict(cfg, overrides)
    return cfg


def validate_config(cfg):
    dc = cfg['data_cfgs']
    mc = cfg['model_cfgs']
    tc = cfg['training_cfgs']
    ec = cfg['evaluation_cfgs']

    if not dc.get('input_path'):
        raise ValueError('data_cfgs.input_path is required')
    if not dc.get('basin_ids'):
        raise ValueError('data_cfgs.basin_ids must not be empty')
    if not dc.get('output_dir'):
        raise ValueError('data_cfgs.output_dir is required')

    variables = dc.get('variables')
    if not isinstance(variables, dict):
        raise ValueError('data_cfgs.variables must be a dict')
    for key in ('dynamic_inputs', 'dynamic_outputs'):
        if key not in variables:
            raise ValueError(f'data_cfgs.variables missing {key!r}')

    calib = dc.get('calib_period')
    valid = dc.get('valid_period')
    if not calib or len(calib) != 2:
        raise ValueError('data_cfgs.calib_period must be [start, end]')
    if not valid or len(valid) != 2:
        raise ValueError('data_cfgs.valid_period must be [start, end]')

    model = mc.get('model_name')
    if model not in MODEL_PARAM_DICT:
        raise ValueError(
            f'unknown model {model!r}; registered: {sorted(MODEL_PARAM_DICT)}'
        )

    algo = tc.get('algorithm_name')
    if not algo:
        raise ValueError('training_cfgs.algorithm_name is required')

    obj_fn = str(tc.get('objective_function', '')).upper()
    if not obj_fn:
        raise ValueError('training_cfgs.objective_function is required')

    ap = tc.get('algorithm_params') or {}
    sceua_param_keys = ('rep', 'ngs', 'kstop', 'peps', 'pcento', 'random_seed')
    missing = [key for key in sceua_param_keys if key not in ap]
    if missing:
        raise ValueError(
            f'missing training_cfgs.algorithm_params: {", ".join(missing)}'
        )
    tc['objective_function'] = obj_fn
    algo_out = {
        'rep': int(ap['rep']),
        'ngs': int(ap['ngs']),
        'kstop': int(ap['kstop']),
        'peps': float(ap['peps']),
        'pcento': float(ap['pcento']),
        'random_seed': int(ap['random_seed']),
    }
    if 'init_perturb' in ap:
        algo_out['init_perturb'] = float(ap['init_perturb'])
    tc['algorithm_params'] = algo_out

    metrics = ec.get('metrics') or []
    if not metrics:
        raise ValueError('evaluation_cfgs.metrics must not be empty')

    if algo not in ALGORITHM_RUNNERS:
        raise ValueError(
            f'unsupported algorithm {algo!r}; available: {sorted(ALGORITHM_RUNNERS)}'
        )
    if obj_fn not in LOSS_FUNCTIONS:
        raise ValueError(
            f'unknown objective_function {obj_fn!r}; '
            f'choose from: {sorted(LOSS_FUNCTIONS)}'
        )
    bad_metrics = [
        name for name in metrics if str(name).upper() not in METRIC_REGISTRY
    ]
    if bad_metrics:
        raise ValueError(
            f'unknown evaluation metric(s) {bad_metrics!r}; '
            f'choose from: {sorted(METRIC_REGISTRY)}'
        )

    if ec.get('params_json') and ec.get('init_params_json'):
        raise ValueError(
            'evaluation_cfgs.params_json and init_params_json are mutually exclusive'
        )


def prepare_config(cfg, basin_ids):
    out = deepcopy(cfg)
    out['data_cfgs']['basin_ids'] = list(basin_ids)
    validate_config(out)
    return out


# ---------------------------------------------------------------------------
# Basin snapshot export
# ---------------------------------------------------------------------------


def basin_config_snapshot(cfg, basin_id):
    snapshot = deepcopy(cfg)
    snapshot['data_cfgs']['basin_ids'] = [str(basin_id)]
    model = snapshot['model_cfgs']['model_name']
    snapshot['model_cfgs'] = {
        **deepcopy(snapshot['model_cfgs']),
        'input_keys': list(MODEL_INPUT_KEYS[model]),
        'param_ranges': deepcopy(MODEL_PARAM_DICT[model]),
    }
    return snapshot


def save_basin_run_configs(cfg, out_dir, basin_ids):
    for basin_id in basin_ids:
        basin_dir = Path(out_dir) / str(basin_id)
        write_json(basin_config_snapshot(cfg, basin_id), basin_dir / 'run_config.json')
