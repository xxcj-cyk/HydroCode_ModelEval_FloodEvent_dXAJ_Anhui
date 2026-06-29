import argparse
import csv
import multiprocessing as mp
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path

from hydromodels_dlm.config.model_config import PBM_PARAM_BOUNDS
from hydromodels_dlm.config.run_config import merge_config, output_dir
from hydromodels_dlm.dataset.data_source import glob_basin_ids, split_file_stem
from hydromodels_dlm.utils.logging import setup_logging
from hydromodels_dlm.workflow.pipeline import run_script_experiment

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[3]
MODEL_TAG = "dXAJ-mz_Local"
SCHEME_PREFIX = f"{MODEL_TAG}_"
TRANSPLANT_PREFIX = "dXAJ-mz_LocalTransplant"
LOCAL_DIR = ROOT / "HydroScirpt/Result/Sec1_ModelPerf" / MODEL_TAG
SIMILARITY_CSV = ROOT / "HydroScirpt/Data/basin_similarity_pairs.csv"
TOP9_CSV = LOCAL_DIR / f"{MODEL_TAG}_PerBasin_Top9_ValidLoss_Hyperparams.csv"
MAX_WORKERS = 16

BASE_CONFIG = {
    "data_cfgs": {
        "input_path": r"/home/xxcj/Dataset/Processed_Dataset/CHINA/Anhui18_663",
        "output_dir": r"HydroScirpt/Result/Sec2_ParamTrans/dXAJ-mz_Local_Transplant_reuse",
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
        "experiment_name": TRANSPLANT_PREFIX,
        "loss_function": "RMSE",
        "optimizer_name": "Adam",
        "learning_rate": 0.05,
        "learning_rate_decay": 0.98,
        "batch_size": 64,
        "epochs": 50,
        "patience": 6,
        "pbm_grad_steps": 1,
        "random_seed": 1111,
        "device": 2,
        "num_workers": 0,
        "transfer_mode": "zero_shot",
        "transfer_scaler_mode": "reuse",
        "transfer_model_path": None,
        "transfer_scaler_path": None,
    },
    "evaluation_cfgs": {
        "metrics": ["NSE", "KGE", "RMSE", "HIGHRMSE", "PFE", "PTE"],
        "eval_mode": "sliding",
    },
    "run_cfgs": {"skip_train": True},
}


# ---------------------------------------------------------------------------
# Similarity / Sec1 artifacts
# ---------------------------------------------------------------------------


def load_similarity_pairs() -> list[tuple[str, str]]:
    pairs = []
    with open(SIMILARITY_CSV, encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            target = str(row["Target Basin ID"]).strip()
            source = str(row["Best Source Basin ID"]).strip()
            if target and source:
                pairs.append((target, source))
    if not pairs:
        raise ValueError(f"no similarity pairs in {SIMILARITY_CSV}")
    return pairs


def load_per_basin_top9() -> dict[str, list[dict]]:
    by_basin: dict[str, list[dict]] = defaultdict(list)
    with open(TOP9_CSV, encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            by_basin[str(row["basin_id"]).strip()].append(
                {
                    "run_name": row["run_name"],
                    "hidden_size": int(row["hidden_size"]),
                    "dropout": float(row["dropout"]),
                    "warmup_length": int(row["warmup_length"]),
                    "batch_size": int(row["batch_size"]),
                    "lr": float(row["lr"]),
                    "seed": int(row["seed"]),
                }
            )
    if not by_basin:
        raise ValueError(f"no per-basin Top9 rows in {TOP9_CSV}")
    return dict(by_basin)


def local_source_dir(source_basin: str, run_name: str) -> Path:
    return LOCAL_DIR / source_basin / f"{SCHEME_PREFIX}{run_name}"


def resolve_transfer_artifacts(source_basin: str, run_name: str) -> tuple[str, str]:
    src_dir = local_source_dir(source_basin, run_name)
    model_path = src_dir / "best_model.pth"
    scaler_path = src_dir / "normalization_scaler.json"
    missing = [path for path in (model_path, scaler_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"missing Local transfer artifacts under {src_dir}: "
            + ", ".join(str(path) for path in missing)
        )
    return str(model_path), str(scaler_path)


def target_event_ids(all_event_ids: list[str], target_basin: str) -> list[str]:
    return sorted(
        event_id
        for event_id in all_event_ids
        if split_file_stem(event_id)[0] == target_basin
    )


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


def build_sweep(test_mode: bool) -> list[tuple]:
    all_event_ids = glob_basin_ids(BASE_CONFIG["data_cfgs"]["input_path"])
    similarity_pairs = load_similarity_pairs()
    top9_by_basin = load_per_basin_top9()
    sweep = []

    for target_basin, source_basin in similarity_pairs:
        source_top9 = top9_by_basin.get(source_basin)
        if not source_top9:
            raise ValueError(f"no Top9 rows for source basin {source_basin} in {TOP9_CSV}")
        events = target_event_ids(all_event_ids, target_basin)
        for hp in source_top9:
            sweep.append(
                (
                    target_basin,
                    source_basin,
                    events,
                    hp["run_name"],
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
    source_basin: str,
    target_event_ids_list: list[str],
    run_name: str,
    hidden_size: int,
    dropout: float,
    warmup_length: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> dict:
    model_path, scaler_path = resolve_transfer_artifacts(source_basin, run_name)

    cfg = deepcopy(BASE_CONFIG)
    cfg["data_cfgs"]["basin_ids"] = list(target_event_ids_list)
    cfg["data_cfgs"]["warmup_length"] = warmup_length
    cfg["model_cfgs"]["model_hyperparam"]["hidden_size"] = hidden_size
    cfg["model_cfgs"]["model_hyperparam"]["dropout"] = dropout
    cfg["training_cfgs"]["experiment_name"] = (
        f"{TRANSPLANT_PREFIX}_{target_basin}_{run_name}"
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
        source_basin,
        target_event_ids_list,
        run_name,
        hidden_size,
        dropout,
        warmup_length,
        batch_size,
        lr,
        seed,
    ) = args
    label = f"{target_basin} <- {source_basin} / {run_name}"
    try:
        overrides = build_experiment_config(
            target_basin,
            source_basin,
            target_event_ids_list,
            run_name,
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
        help="Run one target basin with the source basin Top1 config",
    )
    args = parser.parse_args()

    setup_logging()
    workers = 1 if args.test else MAX_WORKERS
    sweep = build_sweep(args.test)

    print("=" * 60)
    print("Sec2 — dXAJ-mz Local similarity transplant (reuse)")
    print(f"Similarity CSV: {SIMILARITY_CSV}")
    print(f"Local dir: {LOCAL_DIR}")
    print(f"Per-basin Top9 CSV: {TOP9_CSV}")
    print(f"Output dir: {BASE_CONFIG['data_cfgs']['output_dir']}")
    print(f"Runs: {len(sweep)}, workers: {min(workers, len(sweep))}")
    print("-" * 60)

    ok, failed = run_parallel_sweep(sweep, workers)

    print("=" * 60)
    print(f"Finished: ok={ok}, failed={failed}, total={len(sweep)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
