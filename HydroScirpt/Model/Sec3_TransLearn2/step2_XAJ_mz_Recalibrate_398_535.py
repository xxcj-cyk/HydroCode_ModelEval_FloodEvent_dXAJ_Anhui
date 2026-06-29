import argparse

from hydromodels_pbm.utils.logging import setup_logging
from hydromodels_pbm.workflow.pipeline import run_recalibrate_experiment

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATASET_ROOT = r"/home/xxcj/Dataset/Processed_Dataset/CHINA"

SUBSETS = {
    "398": {
        "input_path": f"{DATASET_ROOT}/Anhui18_398",
        "output_dir": r"HydroScirpt/Result/Sec3_TransLearn2/XAJ_mz_Recalibrate_Anhui398",
    },
    "535": {
        "input_path": f"{DATASET_ROOT}/Anhui18_535",
        "output_dir": r"HydroScirpt/Result/Sec3_TransLearn2/XAJ_mz_Recalibrate_Anhui535",
    },
}


def build_config(subset: str) -> dict:
    profile = SUBSETS[subset]
    return {
        "data_cfgs": {
            "input_path": profile["input_path"],
            "output_dir": profile["output_dir"],
            "basin_ids": None,
            "variables": {
                "dynamic_inputs": {"P": "p_anhui", "PET": "pet_anhui"},
                "dynamic_outputs": {"Q": "streamflow_obs_mm"},
            },
            "warmup_length": 168,
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
                "init_perturb": 0.05,
            },
            "objective_function": "RMSE",
        },
        "evaluation_cfgs": {
            "metrics": ["NSE", "KGE", "RMSE", "HighRMSE", "PFE", "PTE"],
            "init_params_json": r"HydroScirpt/Result/Sec1_ModelPerf/XAJ_mz_Local/best_params_all.json",
            "transplant_list_csv": r"HydroScirpt/Data/basin_similarity_pairs.csv",
        },
    }


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
    args = parser.parse_args()

    setup_logging()
    for subset in args.subset:
        print("=" * 60)
        print(f"Sec3_TransLearn2 — XAJ-mz Recalibrate (Anhui{subset})")
        print(f"Output dir: {SUBSETS[subset]['output_dir']}")
        print("-" * 60)
        run_recalibrate_experiment(build_config(subset))
        print("=" * 60)


if __name__ == "__main__":
    main()
