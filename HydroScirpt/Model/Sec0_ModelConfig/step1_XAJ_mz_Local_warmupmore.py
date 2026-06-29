from copy import deepcopy

from hydromodels_pbm.utils.logging import setup_logging
from hydromodels_pbm.workflow.pipeline import run_script_experiment

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WARMUP_LENGTHS = [168, 144, 120, 96, 72, 48, 24]

BASE_CONFIG = {
    "data_cfgs": {
        "input_path": r"/home/xxcj/Dataset/Processed_Dataset/CHINA/Anhui18_663",
        "output_dir": r"HydroScirpt/Result/Sec0_ModelConfig/XAJ_mz_Local",
        "basin_ids": None,
        "variables": {
            "dynamic_inputs": {"P": "p_anhui", "PET": "pet_anhui"},
            "dynamic_outputs": {"Q": "streamflow_obs_mm"},
        },
        "warmup_length": WARMUP_LENGTHS[0],
        "calib_period": ["2024-06-01 00:00:00", "2024-07-31 23:00:00"],
        "valid_period": ["2024-08-01 00:00:00", "2024-08-31 23:00:00"],
    },
    "model_cfgs": {"model_name": "XAJ-mz"},
    "training_cfgs": {
        "algorithm_name": "SCE-UA",
        "algorithm_params": {
            "rep": 200000,
            "ngs": 31,
            "kstop": 100,
            "peps": 0.01,
            "pcento": 0.01,
            "random_seed": 1111,
        },
        "objective_function": "RMSE",
    },
    "evaluation_cfgs": {
        "metrics": ["NSE", "KGE", "RMSE", "HighRMSE", "PFE", "PTE"],
    },
}


def build_warmup_config(warmup_length: int) -> dict:
    cfg = deepcopy(BASE_CONFIG)
    cfg["data_cfgs"]["warmup_length"] = warmup_length
    cfg["data_cfgs"]["output_dir"] = (
        f"{BASE_CONFIG['data_cfgs']['output_dir']}/XAJ_mz_Local_wu{warmup_length:03d}"
    )
    return cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    setup_logging()

    print("=" * 60)
    print("Sec0 — XAJ-mz Local warmup sweep")
    print(f"Output dir: {BASE_CONFIG['data_cfgs']['output_dir']}")
    print(f"Warmup lengths: {WARMUP_LENGTHS}")
    print("-" * 60)

    for warmup_length in WARMUP_LENGTHS:
        run_script_experiment(build_warmup_config(warmup_length))

    print("=" * 60)


if __name__ == "__main__":
    main()
