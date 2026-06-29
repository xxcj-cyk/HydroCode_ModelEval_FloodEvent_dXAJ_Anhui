# HydroScirpt

Paper experiment scripts for flood-event model evaluation on the Anhui dataset. Each script is a self-contained entry point that calls [HydroModels_PBM](../HydroModels_PBM) or [HydroModels_DLM](../HydroModels_DLM) with a fixed `BASE_CONFIG`, writing results under `HydroScirpt/Result/`.

## Overview

This folder orchestrates the manuscript workflow: baseline performance, hyperparameter search, parameter transfer, transfer learning, sensitivity discussion, and peak-focused loss ablation. Scripts are grouped by paper section under `Model/`:

| Section | Purpose |
|---------|---------|
| `Sec0_ModelConfig` | Hyperparameter sweeps (warmup, regional/local LSTM and dXAJ-mz) |
| `Sec1_ModelPerf` | Baseline calibration and ensemble runs for XAJ-mz, SeqRegLSTM, dXAJ-mz |
| `Sec2_ParamTrans` | Parameter transplant and leave-one-out experiments |
| `Sec3_TransLearn1` | Transfer learning on the full Anhui663 dataset |
| `Sec3_TransLearn2` | Transfer learning with 398 train / 535 test basin split |
| `Sec4_Discussion` | Transplant vs. translearn comparisons, ensemble variants |
| `Sec5_Test` | PeakFocusedLoss ablation with reduced hyperparameter grids |

Supporting data live in `Data/` (e.g. `sampled_hyperparams_*.csv`, `basin_similarity_pairs.csv`).

## Models used

| Backend | Models | Typical use |
|---------|--------|-------------|
| PBM | `XAJ-mz` | Local calibration (SCE-UA), parameter transplant |
| DLM | `SeqRegLSTM`, `dXAJ-mz` | Regional/local training, transfer learning, peak-focused loss |

Evaluation metrics include NSE, KGE, RMSE, HighRMSE, PFE, and PTE — aligned with flood-event performance assessment.

## Data and paths

Scripts default to Anhui processed CSVs, for example:

```
/home/xxcj/Dataset/Processed_Dataset/CHINA/Anhui18_663
```

Update `data_cfgs.input_path` in each script for your environment. Flood-event runs use short calibration / validation windows (e.g. June–August 2024) with `warmup_length` typically 120–168 hours.

Hyperparameter sweep scripts read grids from `HydroScirpt/Data/sampled_hyperparams_*.csv` and run jobs in parallel via `ProcessPoolExecutor`.

## Running an experiment

From the repository root:

```bash
python HydroScirpt/Model/Sec1_ModelPerf/step1_XAJ_mz_Local.py
python HydroScirpt/Model/Sec0_ModelConfig/step2_SeqRegLSTM_Regional_hypermore.py
python HydroScirpt/Model/Sec5_Test/step2_SeqRegLSTM_Regional_PeakFocused_hypermore.py
```

Outputs are written relative to the repo root, e.g. `HydroScirpt/Result/Sec1_ModelPerf/XAJ_mz_Local/`.

Later sections often depend on earlier results (calibrated params, top hyperparameter CSVs, trained checkpoints). Check each script's `evaluation_cfgs` or path constants before running downstream steps.

## Directory layout

```
HydroScirpt/
  Data/           hyperparameter samples, basin similarity pairs
  Model/
    Sec0_ModelConfig/
    Sec1_ModelPerf/
    Sec2_ParamTrans/
    Sec3_TransLearn1/
    Sec3_TransLearn2/
    Sec4_Discussion/
    Sec5_Test/
  Result/         experiment outputs (checkpoints, metrics, logs)
```
