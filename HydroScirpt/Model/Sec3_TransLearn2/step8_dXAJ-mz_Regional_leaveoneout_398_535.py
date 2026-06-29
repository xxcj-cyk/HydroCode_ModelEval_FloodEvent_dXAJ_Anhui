import argparse
import csv
import multiprocessing as mp
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path

from hydromodels_dlm.config.model_config import PBM_PARAM_BOUNDS
from hydromodels_dlm.config.run_config import merge_config, output_dir
from hydromodels_dlm.dataset.data_source import (
    glob_basin_ids,
    group_file_ids_by_basin,
    split_file_stem,
)
from hydromodels_dlm.utils.logging import setup_logging
from hydromodels_dlm.workflow.pipeline import run_script_experiment

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = r"/home/xxcj/Dataset/Processed_Dataset/CHINA"
PROJECT_PREFIX = "dXAJ-mz_Train"
TOP9_CSV = (
    ROOT
    / "HydroScirpt/Result/Sec1_ModelPerf/dXAJ-mz_Regional"
    / "dXAJ-mz_Regional_Top9_ValidLoss_Hyperparams.csv"
)

STATIC_ATTRIBUTES = [
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
]

SUBSETS = {
    "398": {
        "input_path": f"{DATASET_ROOT}/Anhui18_398",
        "output_dir": r"HydroScirpt/Result/Sec3_TransLearn2/dXAJ-mz_Train_Anhui398",
        "max_workers": 5,
    },
    "535": {
        "input_path": f"{DATASET_ROOT}/Anhui18_535",
        "output_dir": r"HydroScirpt/Result/Sec3_TransLearn2/dXAJ-mz_Train_Anhui535",
        "max_workers": 5,
    },
}


# ---------------------------------------------------------------------------
# Run naming
# ---------------------------------------------------------------------------


def format_lr_tag(lr: float) -> str:
    text = str(lr)
    if text.startswith("0."):
        text = text[2:].rstrip("0") or "0"
    return text


def build_run_name(
    hidden_size: int,
    dropout: float,
    warmup_length: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> str:
    dr_tag = f"dr{int(round(dropout * 10)):02d}"
    return (
        f"h{hidden_size:03d}_{dr_tag}"
        f"_wu{warmup_length:03d}_b{batch_size:04d}"
        f"_lr{format_lr_tag(lr)}_seed{seed}"
    )


def base_config(subset: str) -> dict:
    profile = SUBSETS[subset]
    return {
        "data_cfgs": {
            "input_path": profile["input_path"],
            "output_dir": profile["output_dir"],
            "basin_ids": None,
            "variables": {
                "dynamic_inputs": {"P": "p_anhui", "PET": "pet_anhui"},
                "dynamic_outputs": {"Q": "streamflow_obs_mm"},
                "static_attributes": STATIC_ATTRIBUTES,
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
            "model_name": "dXAJ-mz",
            "model_hyperparam": {
                "input_size": 22,
                "output_size": len(PBM_PARAM_BOUNDS["dXAJ-mz"]),
                "hidden_size": 128,
                "dropout": 0.0,
                "input_proj": None,
            },
        },
        "training_cfgs": {
            "strategy": "RegionalTrain",
            "experiment_name": PROJECT_PREFIX,
            "loss_function": "RMSE",
            "optimizer_name": "Adam",
            "learning_rate": 0.001,
            "learning_rate_decay": 0.98,
            "batch_size": 32,
            "epochs": 50,
            "patience": 6,
            "pbm_grad_steps": 1,
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
# Sweep
# ---------------------------------------------------------------------------


def load_top9_hyperparams() -> list[dict]:
    with open(TOP9_CSV, encoding="utf-8-sig", newline="") as handle:
        return [
            {
                "hidden_size": int(row["hidden_size"]),
                "dropout": float(row["dropout"]),
                "warmup_length": int(row["warmup_length"]),
                "batch_size": int(row["batch_size"]),
                "lr": float(row["lr"]),
                "seed": int(row["seed"]),
            }
            for row in csv.DictReader(handle)
        ]


def build_sweep(subset: str, test_mode: bool) -> list[tuple]:
    cfg = base_config(subset)
    all_event_ids = glob_basin_ids(cfg["data_cfgs"]["input_path"])
    target_basins = sorted(group_file_ids_by_basin(all_event_ids).keys())
    top9 = load_top9_hyperparams()
    sweep = []

    for target_basin in target_basins:
        train_event_ids = sorted(
            event_id
            for event_id in all_event_ids
            if split_file_stem(event_id)[0] != target_basin
        )
        for hp in top9:
            sweep.append(
                (
                    subset,
                    target_basin,
                    train_event_ids,
                    hp["hidden_size"],
                    hp["dropout"],
                    hp["warmup_length"],
                    hp["batch_size"],
                    hp["lr"],
                    hp["seed"],
                )
            )
        if test_mode:
            return sweep[:1]
    return sweep


def build_experiment_config(
    subset: str,
    target_basin: str,
    train_event_ids: list[str],
    hidden_size: int,
    dropout: float,
    warmup_length: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> dict:
    cfg = deepcopy(base_config(subset))
    run_name = build_run_name(hidden_size, dropout, warmup_length, batch_size, lr, seed)
    cfg["data_cfgs"]["basin_ids"] = list(train_event_ids)
    cfg["data_cfgs"]["warmup_length"] = warmup_length
    cfg["model_cfgs"]["model_hyperparam"]["hidden_size"] = hidden_size
    cfg["model_cfgs"]["model_hyperparam"]["dropout"] = dropout
    cfg["training_cfgs"]["experiment_name"] = (
        f"{PROJECT_PREFIX}_{target_basin}_without_{run_name}"
    )
    cfg["training_cfgs"]["learning_rate"] = lr
    cfg["training_cfgs"]["batch_size"] = batch_size
    cfg["training_cfgs"]["random_seed"] = seed
    return cfg


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def run_single_scheme(args: tuple) -> bool:
    (
        subset,
        target_basin,
        train_event_ids,
        hidden_size,
        dropout,
        warmup_length,
        batch_size,
        lr,
        seed,
    ) = args
    run_name = build_run_name(hidden_size, dropout, warmup_length, batch_size, lr, seed)
    label = f"Anhui{subset} / {target_basin} / {run_name}"
    try:
        overrides = build_experiment_config(
            subset,
            target_basin,
            train_event_ids,
            hidden_size,
            dropout,
            warmup_length,
            batch_size,
            lr,
            seed,
        )
        print(f"  -> {output_dir(merge_config(overrides))}")
        run_script_experiment(overrides)
        print(f"Done: {label}")
        return True
    except Exception as exc:
        print(f"Failed: {label} -> {exc}")
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
                print(f"Error: Anhui{args[0]} / {args[1]} -> {exc}")
            print(f"[Progress] ok={ok}, failed={failed}, total={len(sweep)}")
    return ok, failed


def run_subset(subset: str, test_mode: bool) -> tuple[int, int]:
    sweep = build_sweep(subset, test_mode)
    workers = 1 if test_mode else SUBSETS[subset]["max_workers"]
    train_events = len(sweep[0][2]) if sweep else 0

    print("=" * 60)
    print(f"Sec3_TransLearn2 — dXAJ-mz regional leaveoneout train Top9 (Anhui{subset})")
    print(f"Top9 CSV: {TOP9_CSV}")
    print(f"Output dir: {SUBSETS[subset]['output_dir']}")
    print(f"Train events per run: {train_events}")
    print(f"Total runs: {len(sweep)}, workers: {min(workers, len(sweep))}")
    print("-" * 60)

    return run_parallel_sweep(sweep, workers)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subset",
        choices=sorted(SUBSETS),
        nargs="+",
        default=sorted(SUBSETS),
        help="Data subset(s) to run (default: 398 and 535)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run one target basin with the Top1 hyperparam config",
    )
    args = parser.parse_args()

    setup_logging()
    total_ok = total_failed = 0
    for subset in args.subset:
        ok, failed = run_subset(subset, args.test)
        total_ok += ok
        total_failed += failed

    print("=" * 60)
    print(f"All finished: ok={total_ok}, failed={total_failed}")
    print("=" * 60)


if __name__ == "__main__":
    main()
