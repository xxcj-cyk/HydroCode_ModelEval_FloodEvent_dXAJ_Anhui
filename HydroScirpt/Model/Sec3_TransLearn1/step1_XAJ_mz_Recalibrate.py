from hydromodels_pbm.utils.logging import setup_logging
from hydromodels_pbm.workflow.pipeline import run_recalibrate_experiment

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_CONFIG = {
    "data_cfgs": {
        "input_path": r"/home/xxcj/Dataset/Processed_Dataset/CHINA/Anhui18_663",
        "output_dir": r"HydroScirpt/Result/Sec3_TransLearn1/XAJ_mz_Recalibrate",
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
    setup_logging()

    print("=" * 60)
    print("Sec3 — XAJ-mz parameter recalibrate")
    print(f"Output dir: {BASE_CONFIG['data_cfgs']['output_dir']}")
    print("-" * 60)

    run_recalibrate_experiment(BASE_CONFIG)

    print("=" * 60)


if __name__ == "__main__":
    main()
