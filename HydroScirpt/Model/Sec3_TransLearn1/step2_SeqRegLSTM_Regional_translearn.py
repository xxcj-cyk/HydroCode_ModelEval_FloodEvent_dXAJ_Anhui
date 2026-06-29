import argparse
import csv
import multiprocessing as mp
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path

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
TRAIN_PREFIX = "SeqRegLSTM_Train"
TRANSLEARN_PREFIX = "SeqRegLSTM_TransLearn"
TRAIN_DIR = ROOT / "HydroScirpt/Result/Sec2_ParamTrans/SeqRegLSTM_Train"
TOP9_CSV = (
    ROOT
    / "HydroScirpt/Result/Sec1_ModelPerf/SeqRegLSTM_Regional"
    / "SeqRegLSTM_Regional_Top9_ValidLoss_Hyperparams.csv"
)
MAX_WORKERS = 4

BASE_CONFIG = {
    "data_cfgs": {
        "input_path": r"/home/xxcj/Dataset/Processed_Dataset/CHINA/Anhui18_663",
        "output_dir": r"HydroScirpt/Result/Sec3_TransLearn1/SeqRegLSTM_TransLearn_reuse",
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
        "experiment_name": TRANSLEARN_PREFIX,
        "loss_function": "RMSE",
        "optimizer_name": "Adam",
        "learning_rate": 0.05,
        "learning_rate_decay": 0.98,
        "batch_size": 64,
        "epochs": 50,
        "patience": 6,
        "random_seed": 1111,
        "device": 2,
        "num_workers": 0,
        "transfer_mode": "finetune",
        "transfer_scaler_mode": "reuse",
        "transfer_model_path": None,
        "transfer_scaler_path": None,
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


def resolve_transfer_artifacts(target_basin: str, run_name: str) -> tuple[str, str]:
    src_dir = TRAIN_DIR / f"{TRAIN_PREFIX}_{target_basin}_without_{run_name}"
    model_path = src_dir / "best_model.pth"
    scaler_path = src_dir / "normalization_scaler.json"
    missing = [path for path in (model_path, scaler_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"missing transfer artifacts under {src_dir}: "
            + ", ".join(str(path) for path in missing)
        )
    return str(model_path), str(scaler_path)


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


def build_sweep(test_mode: bool) -> list[tuple]:
    all_event_ids = glob_basin_ids(BASE_CONFIG["data_cfgs"]["input_path"])
    target_basins = sorted(group_file_ids_by_basin(all_event_ids).keys())
    top9 = load_top9_hyperparams()
    sweep = []

    for target_basin in target_basins:
        target_event_ids = sorted(
            event_id
            for event_id in all_event_ids
            if split_file_stem(event_id)[0] == target_basin
        )
        for hp in top9:
            sweep.append(
                (
                    target_basin,
                    target_event_ids,
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
    target_basin: str,
    target_event_ids: list[str],
    hidden_size: int,
    dropout: float,
    warmup_length: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> dict:
    run_name = build_run_name(hidden_size, dropout, warmup_length, batch_size, lr, seed)
    model_path, scaler_path = resolve_transfer_artifacts(target_basin, run_name)

    cfg = deepcopy(BASE_CONFIG)
    cfg["data_cfgs"]["basin_ids"] = list(target_event_ids)
    cfg["data_cfgs"]["warmup_length"] = warmup_length
    cfg["model_cfgs"]["model_hyperparam"]["hidden_size"] = hidden_size
    cfg["model_cfgs"]["model_hyperparam"]["dropout"] = dropout
    cfg["training_cfgs"]["experiment_name"] = (
        f"{TRANSLEARN_PREFIX}_{target_basin}_{run_name}"
    )
    cfg["training_cfgs"]["learning_rate"] = lr
    cfg["training_cfgs"]["batch_size"] = batch_size
    cfg["training_cfgs"]["random_seed"] = seed
    cfg["training_cfgs"]["transfer_model_path"] = model_path
    cfg["training_cfgs"]["transfer_scaler_path"] = scaler_path
    return cfg


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def run_single_scheme(args: tuple) -> bool:
    (
        target_basin,
        target_event_ids,
        hidden_size,
        dropout,
        warmup_length,
        batch_size,
        lr,
        seed,
    ) = args
    run_name = build_run_name(hidden_size, dropout, warmup_length, batch_size, lr, seed)
    label = f"{target_basin} / {run_name}"
    try:
        overrides = build_experiment_config(
            target_basin,
            target_event_ids,
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
                print(f"Error: {futures[future][0]} -> {exc}")
            print(f"[Progress] ok={ok}, failed={failed}, total={len(sweep)}")
    return ok, failed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run one target basin with the Top1 hyperparam config",
    )
    args = parser.parse_args()

    setup_logging()
    workers = 1 if args.test else MAX_WORKERS
    sweep = build_sweep(args.test)

    print("=" * 60)
    print("Sec3 — SeqRegLSTM Regional transfer finetune (reuse)")
    print(f"Top9 CSV: {TOP9_CSV}")
    print(f"Train dir: {TRAIN_DIR}")
    print(f"Output dir: {BASE_CONFIG['data_cfgs']['output_dir']}")
    print(f"Runs: {len(sweep)}, workers: {min(workers, len(sweep))}")
    print("-" * 60)

    ok, failed = run_parallel_sweep(sweep, workers)

    print("=" * 60)
    print(f"Finished: ok={ok}, failed={failed}, total={len(sweep)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
