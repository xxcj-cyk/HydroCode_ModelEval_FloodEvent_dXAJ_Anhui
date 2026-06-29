import csv
import multiprocessing as mp
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path

from hydromodels_dlm.config.run_config import merge_config, output_dir
from hydromodels_dlm.utils.logging import setup_logging
from hydromodels_dlm.workflow.pipeline import run_script_experiment

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PEAK_FOCUSED_WEIGHTS = {
    "overall_weight": 1.0,
    "peak_weight": 0.5,
    "high_weight": 0.5,
}

HYPERPARAMS_CSV = (
    Path(__file__).resolve().parents[2] / "Data" / "sampled_hyperparams_200.csv"
)
MAX_WORKERS = 8

BASE_CONFIG = {
    "data_cfgs": {
        "input_path": r"/home/xxcj/Dataset/Processed_Dataset/CHINA/Anhui18_663",
        "output_dir": r"HydroScirpt/Result/Sec5_Test/SeqRegLSTM_Regional",
        "basin_ids": None,
        "variables": {
            "dynamic_inputs": {"P": "p_anhui", "PET": "pet_anhui"},
            "dynamic_outputs": {"Q": "streamflow_obs_mm"},
            "static_attributes": [
                "Pmean",
                "Area",
                "swc_pc_syr",
                "sgr_dk_sav",
                "ele_mt_sav",
                "slp_dg_sav",
                "pet_mm_syr",
                "aet_mm_syr",
                "tmp_dc_syr",
                "cmi_ix_syr",
                "snw_pc_syr",
                "ari_ix_sav",
                "for_pc_sse",
                "crp_pc_sse",
                "lit_cl_smj",
                "soc_th_sav",
                "kar_pc_sse",
                "snd_pc_sav",
                "slt_pc_sav",
                "cly_pc_sav",
            ],
        },
        "scaler_params": {
            "log1p_zscore": ["P", "PET"],
            "prcp_log1p_zscore": ["Q"],
        },
        "forecast_length": 1,
        "input_layout": "dynamic_static",
        "warmup_length": 144,
        "train_period": ["2024-06-01 00:00:00", "2024-06-30 23:00:00"],
        "valid_period": ["2024-07-01 00:00:00", "2024-07-31 23:00:00"],
        "test_period": ["2024-08-01 00:00:00", "2024-08-31 23:00:00"],
    },
    "model_cfgs": {
        "model_name": "SeqRegLSTM",
        "model_hyperparam": {
            "input_size": 22,
            "output_size": 1,
            "hidden_size": 128,
            "dropout": 0.0,
            "input_proj": None,
        },
    },
    "training_cfgs": {
        "strategy": "RegionalTrain",
        "experiment_name": "SeqRegLSTM_Regional_PF_h128_dr00_wu144_b0064_lr005_seed1111",
        "loss_function": "PeakFocused",
        "peak_focused_weights": PEAK_FOCUSED_WEIGHTS,
        "optimizer_name": "Adam",
        "learning_rate": 0.05,
        "learning_rate_decay": 0.98,
        "batch_size": 64,
        "epochs": 50,
        "patience": 6,
        "random_seed": 1111,
        "device": 2,
        "num_workers": 0,
    },
    "evaluation_cfgs": {
        "metrics": ["NSE", "KGE", "RMSE", "HIGHRMSE", "PFE", "PTE"],
        "eval_mode": "sliding",
    },
    "run_cfgs": {"skip_train": False},
}

# ---------------------------------------------------------------------------
# Run naming
# ---------------------------------------------------------------------------


def format_lr_tag(lr: float) -> str:
    text = str(lr)
    if text.startswith("0."):
        text = text[2:].rstrip("0") or "0"
    return text


def build_run_name(hidden_size, dropout, warmup_length, batch_size, lr) -> str:
    seed = BASE_CONFIG["training_cfgs"]["random_seed"]
    dr_tag = f"dr{int(round(dropout * 10)):02d}"
    return (
        f"h{hidden_size:03d}_{dr_tag}"
        f"_wu{warmup_length:03d}_b{batch_size:04d}"
        f"_lr{format_lr_tag(lr)}_seed{seed}"
    )


def build_experiment_config(
    hidden_size, dropout, warmup_length, batch_size, lr
) -> dict:
    cfg = deepcopy(BASE_CONFIG)
    run_name = build_run_name(hidden_size, dropout, warmup_length, batch_size, lr)
    prefix = cfg["training_cfgs"]["experiment_name"].split("_h", 1)[0]
    cfg["data_cfgs"]["warmup_length"] = warmup_length
    cfg["model_cfgs"]["model_hyperparam"]["hidden_size"] = hidden_size
    cfg["model_cfgs"]["model_hyperparam"]["dropout"] = dropout
    cfg["training_cfgs"]["experiment_name"] = f"{prefix}_{run_name}"
    cfg["training_cfgs"]["learning_rate"] = lr
    cfg["training_cfgs"]["batch_size"] = batch_size
    return cfg


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


def load_hyperparam_sweep() -> list[tuple]:
    with open(HYPERPARAMS_CSV, encoding="utf-8-sig", newline="") as handle:
        return [
            (
                int(row["hidden_size"]),
                float(row["dropout"]),
                int(row["warmup_length"]),
                int(row["batch_size"]),
                float(row["lr"]),
            )
            for row in csv.DictReader(handle)
        ]


def run_single_scheme(args: tuple) -> bool:
    hidden_size, dropout, warmup_length, batch_size, lr = args
    run_name = build_run_name(hidden_size, dropout, warmup_length, batch_size, lr)
    try:
        overrides = build_experiment_config(
            hidden_size, dropout, warmup_length, batch_size, lr
        )
        print(f"  -> {output_dir(merge_config(overrides))}")
        run_script_experiment(overrides)
        print(f"Done: {run_name}")
        return True
    except Exception as exc:
        print(f"Failed: {run_name} -> {exc}")
        traceback.print_exc()
        return False


def run_parallel_sweep(sweep: list[tuple], max_workers: int) -> tuple[int, int]:
    worker_num = min(max_workers, len(sweep))
    ok = failed = 0
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=worker_num, mp_context=ctx) as executor:
        futures = {executor.submit(run_single_scheme, args): args for args in sweep}
        for future in as_completed(futures):
            try:
                if future.result():
                    ok += 1
                else:
                    failed += 1
            except Exception as exc:
                failed += 1
                args = futures[future]
                print(f"Error: {build_run_name(*args)} -> {exc}")
            print(f"[Progress] ok={ok}, failed={failed}, total={len(sweep)}")
    return ok, failed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    setup_logging()
    sweep = load_hyperparam_sweep()

    print("=" * 60)
    print("Sec5 — SeqRegLSTM RegionalTrain PeakFocused hyperparam sweep")
    print(f"Hyperparams CSV: {HYPERPARAMS_CSV}")
    print(f"Output dir: {BASE_CONFIG['data_cfgs']['output_dir']}")
    print(f"Loss: {BASE_CONFIG['training_cfgs']['loss_function']}")
    print(f"PeakFocused weights: {PEAK_FOCUSED_WEIGHTS}")
    print(f"Total runs: {len(sweep)}, workers: {min(MAX_WORKERS, len(sweep))}")
    print("-" * 60)

    ok, failed = run_parallel_sweep(sweep, MAX_WORKERS)

    print("=" * 60)
    print(f"Finished: ok={ok}, failed={failed}, total={len(sweep)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
