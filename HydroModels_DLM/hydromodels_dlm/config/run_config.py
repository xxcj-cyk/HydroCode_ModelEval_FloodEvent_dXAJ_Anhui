import json
import os
from copy import deepcopy
from pathlib import Path

from hydromodels_dlm.dataset.scaler import validate_scaler_params
from hydromodels_dlm.dataset.data_source import (
    input_data_root,
    is_flood_event_id,
    load_attributes,
    parse_variables,
    resolve_basin_ids,
    split_file_stem,
)
from hydromodels_dlm.dataset.input_layout import INPUT_LAYOUT_REGISTRY
from hydromodels_dlm.workflow.losses import LOSS_FUNCTIONS
from hydromodels_dlm.workflow.metrics import METRIC_REGISTRY


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def default_config():
    return {
        "data_cfgs": {
            "input_path": None,
            "basin_ids": None,
            "variables": {
                "dynamic_inputs": {
                    "P": "precipitation",
                    "PET": "potential evapotranspiration",
                },
                "dynamic_outputs": {"Q": "streamflow"},
                "static_attributes": [
                    'swc_pc_syr',
                    'sgr_dk_sav',
                    'Area',
                    'ele_mt_sav',
                    'slp_dg_sav',
                    'Pmean_camels',
                    'pet_mm_syr',
                    'aet_mm_syr',
                    'tmp_dc_syr',
                    'cmi_ix_syr',
                    'snw_pc_syr',
                    'ari_ix_sav',
                    'for_pc_sse',
                    'crp_pc_sse',
                    'lit_cl_smj',
                    'soc_th_sav',
                    'kar_pc_sse',
                    'snd_pc_sav',
                    'slt_pc_sav',
                    'cly_pc_sav',
                ],
            },
            "warmup_length": 365,
            "forecast_length": 1,
            "train_period": ["1980-01-01", "2010-12-31"],
            "valid_period": ["2011-01-01", "2015-12-31"],
            "test_period": ["2016-01-01", "2020-12-31"],
            "output_dir": "results",
            "scaler_params": {
                "log1p_zscore": ["P", "PET"],
                "prcp_log1p_zscore": ["Q"],
            },
            "input_layout": "dynamic_static",
        },
        "model_cfgs": {
            "model_name": "SeqRegLSTM",
            "model_hyperparam": {
                "input_size": 22,
                "output_size": 1,
                "hidden_size": 32,
                "dropout": 0,
                "input_proj": None,
            },
            "weight_path": None,
        },
        "training_cfgs": {
            "experiment_name": "hydro_dl_default",
            "strategy": "LocalTrain",
            "loss_function": "RMSE",
            "optimizer_name": "Adam",
            "learning_rate": 0.001,
            "learning_rate_decay": 1.0,
            "epochs": 50,
            "batch_size": 256,
            "patience": 10,
            "device": 0,
            "num_workers": 0,
            "random_seed": 1111,
            "transfer_model_path": None,
            "transfer_scaler_path": None,
            "transfer_mode": "zero_shot",
            "transfer_scaler_mode": "reuse",
        },
        "evaluation_cfgs": {
            "metrics": ["NSE", "KGE", "RMSE", "CORR", "R2", "MBE"],
            "weight_path": None,
            "eval_mode": "sliding",
        },
        "run_cfgs": {
            "skip_train": False,
        },
    }


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------


def model_name(cfg):
    return cfg["model_cfgs"]["model_name"]


def data_cfgs(cfg):
    return deepcopy(cfg["data_cfgs"])


def experiment_name(cfg):
    return cfg["training_cfgs"]["experiment_name"]


TRAINING_STRATEGIES = ("LocalTrain", "RegionalTrain")


def training_strategy(cfg):
    raw = cfg["training_cfgs"].get("strategy")
    if raw is None:
        raise KeyError("training_cfgs.strategy is required")
    return str(raw)


def loss_function(cfg):
    return cfg["training_cfgs"]["loss_function"]


DEFAULT_PEAK_FOCUSED_WEIGHTS = {
    "overall_weight": 1.0,
    "peak_weight": 0.5,
    "high_weight": 1.0,
}


def peak_focused_weights(cfg):
    tc = cfg["training_cfgs"]
    raw = tc.get("peak_focused_weights") or {}
    weights = {**DEFAULT_PEAK_FOCUSED_WEIGHTS, **raw}
    out = {}
    for key, default in DEFAULT_PEAK_FOCUSED_WEIGHTS.items():
        if key not in weights:
            raise KeyError(f"peak_focused_weights missing {key!r}")
        value = float(weights[key])
        if value < 0:
            raise ValueError(f"peak_focused_weights[{key!r}] must be >= 0, got {value!r}")
        out[key] = value
    return out


def evaluation_metrics(cfg):
    return list(cfg["evaluation_cfgs"]["metrics"])


def parse_eval_mode(mode, *, default="sequential"):
    value = str(mode if mode is not None else default).lower()
    if value in ("sliding", "sequential"):
        return value
    raise ValueError(
        f"evaluation_cfgs.eval_mode must be 'sliding' or 'sequential', got {mode!r}"
    )


def eval_mode(cfg):
    return parse_eval_mode(cfg["evaluation_cfgs"].get("eval_mode"))


def transfer_model_path(cfg):
    raw = cfg["training_cfgs"].get("transfer_model_path")
    return str(raw) if raw else None


def transfer_scaler_path(cfg):
    raw = cfg["training_cfgs"].get("transfer_scaler_path")
    return str(raw) if raw else None


def transfer_scaler_mode(cfg):
    return str(cfg["training_cfgs"].get("transfer_scaler_mode", "reuse")).lower()


TRANSFER_MODES = ("zero_shot", "finetune")


def transfer_mode(cfg):
    return str(cfg["training_cfgs"].get("transfer_mode", "zero_shot")).lower()


def transfer_enabled(cfg):
    return bool(transfer_model_path(cfg) and transfer_scaler_path(cfg))


def transfer_finetune(cfg):
    return transfer_enabled(cfg) and transfer_mode(cfg) == "finetune"


def transfer_zero_shot(cfg):
    return transfer_enabled(cfg) and transfer_mode(cfg) == "zero_shot"


def weight_path(cfg):
    if transfer_zero_shot(cfg):
        return transfer_model_path(cfg)
    raw = cfg["evaluation_cfgs"].get("weight_path") or cfg["model_cfgs"].get(
        "weight_path"
    )
    return str(raw) if raw else None


def output_dir(cfg):
    return os.path.join(cfg["data_cfgs"]["output_dir"], experiment_name(cfg))


def skip_train(cfg):
    if transfer_zero_shot(cfg):
        return True
    return bool(cfg["run_cfgs"]["skip_train"])


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


def merge_dict(base, patch):
    for key, value in patch.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            merge_dict(base[key], value)
        else:
            base[key] = deepcopy(value)


def merge_config(overrides=None):
    cfg = deepcopy(default_config())
    if overrides:
        merge_dict(cfg, overrides)
    return cfg


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------


def validate_config(cfg):
    dc = cfg["data_cfgs"]
    tc = cfg["training_cfgs"]

    if not dc.get("input_path"):
        raise ValueError("data_cfgs.input_path is required")

    variables = dc.get("variables")
    if not isinstance(variables, dict):
        raise ValueError("data_cfgs.variables must be a dict")
    for key in ("dynamic_inputs", "dynamic_outputs"):
        if key not in variables:
            raise ValueError(f"data_cfgs.variables missing {key!r}")

    for key in ("train_period", "valid_period", "test_period"):
        period = dc.get(key)
        if not period or len(period) != 2:
            raise ValueError(f"data_cfgs.{key} must be [start, end]")

    if int(dc["warmup_length"]) < 0:
        raise ValueError("data_cfgs.warmup_length must be >= 0")
    if int(dc.get("forecast_length", 1)) < 1:
        raise ValueError("data_cfgs.forecast_length must be >= 1")

    scaler_params = dc.get("scaler_params") or {}
    _, input_keys, output_keys, static_keys = parse_variables(dc)
    all_symbols = list(input_keys) + list(output_keys) + list(static_keys)
    validate_scaler_params(scaler_params, all_symbols)

    layout = dc.get('input_layout', 'dynamic_static')
    if layout not in INPUT_LAYOUT_REGISTRY:
        raise ValueError(
            f'unknown data_cfgs.input_layout {layout!r}; '
            f'choose from: {sorted(INPUT_LAYOUT_REGISTRY)}'
        )

    needs_prcp = "Q" in (scaler_params.get("prcp_log1p_zscore") or [])
    if needs_prcp:
        root = dc.get("input_path")
        if root:
            load_attributes(root, static_keys)

    from hydromodels_dlm.model.model_dict import MODEL_DICT

    mname = model_name(cfg)
    if mname not in MODEL_DICT:
        raise ValueError(
            f"unknown model {mname!r}; registered: {sorted(MODEL_DICT)}"
        )

    strategy = training_strategy(cfg)
    if strategy not in TRAINING_STRATEGIES:
        raise ValueError(
            f"training_cfgs.strategy must be one of {TRAINING_STRATEGIES}, "
            f"got {strategy!r}"
        )

    loss_fn = str(loss_function(cfg)).upper()
    if loss_fn not in LOSS_FUNCTIONS:
        raise ValueError(
            f"unknown loss_function {loss_fn!r}; "
            f"choose from: {sorted(LOSS_FUNCTIONS)}"
        )
    tc["loss_function"] = loss_fn
    if loss_fn == "PEAKFOCUSED":
        tc["peak_focused_weights"] = peak_focused_weights(cfg)

    if int(tc["epochs"]) < 1:
        raise ValueError("training_cfgs.epochs must be >= 1")
    if int(tc["batch_size"]) < 1:
        raise ValueError("training_cfgs.batch_size must be >= 1")
    if int(tc["patience"]) < 1:
        raise ValueError("training_cfgs.patience must be >= 1")

    lr = float(tc["learning_rate"])
    if lr <= 0:
        raise ValueError("training_cfgs.learning_rate must be > 0")
    lr_decay = float(tc.get("learning_rate_decay", 1.0))
    if not 0 < lr_decay <= 1.0:
        raise ValueError(
            "training_cfgs.learning_rate_decay must be in (0, 1], got "
            f"{lr_decay!r}"
        )

    bad_metrics = [
        name
        for name in evaluation_metrics(cfg)
        if str(name).upper() not in METRIC_REGISTRY
    ]
    if bad_metrics:
        raise ValueError(
            f"unknown evaluation metric(s) {bad_metrics!r}; "
            f"choose from: {sorted(METRIC_REGISTRY)}"
        )

    eval_mode(cfg)

    tm = tc.get("transfer_model_path")
    ts = tc.get("transfer_scaler_path")
    if bool(tm) ^ bool(ts):
        raise ValueError(
            "training_cfgs.transfer_model_path and transfer_scaler_path "
            "must both be set for transfer"
        )
    if transfer_enabled(cfg):
        for label, path in (
            ("transfer_model_path", tm),
            ("transfer_scaler_path", ts),
        ):
            if not Path(path).is_file():
                raise ValueError(f"training_cfgs.{label} not found: {path}")
        mode = transfer_scaler_mode(cfg)
        if mode not in ("reuse", "extend"):
            raise ValueError(
                "training_cfgs.transfer_scaler_mode must be 'reuse' or 'extend', "
                f"got {mode!r}"
            )
        tmode = transfer_mode(cfg)
        if tmode not in TRANSFER_MODES:
            raise ValueError(
                f"training_cfgs.transfer_mode must be one of {TRANSFER_MODES}, "
                f"got {tmode!r}"
            )


def finalize_config(basin_ids_list, cfg):
    out = deepcopy(cfg)
    out["data_cfgs"]["basin_ids"] = list(basin_ids_list)
    if not out["data_cfgs"]["basin_ids"]:
        raise ValueError("no basin_ids resolved")
    validate_config(out)

    dc = out["data_cfgs"]
    _, _, _, static_keys = parse_variables(dc)
    root = input_data_root(dc)
    if all(is_flood_event_id(fid) for fid in basin_ids_list):
        attr_basins = sorted({split_file_stem(fid)[0] for fid in basin_ids_list})
    else:
        attr_basins = list(basin_ids_list)
    load_attributes(root, static_keys, attr_basins)
    return out


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def dump_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def run_config_for_basin(cfg, basin_id):
    snapshot = deepcopy(cfg)
    snapshot["data_cfgs"]["basin_ids"] = [str(basin_id)]
    return snapshot


def save_basin_run_configs(cfg, out_dir, basin_ids):
    for basin_id in basin_ids:
        basin_dir = Path(out_dir) / str(basin_id)
        dump_json(run_config_for_basin(cfg, basin_id), basin_dir / "run_config.json")
