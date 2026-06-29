# HydroModels_PBM

Process-based hydrological modeling workflow for basin-scale calibration, evaluation, and parameter transfer. This package mirrors the layout of [HydroModels_DLM](../HydroModels_DLM) but uses classical PBM solvers and global optimization instead of deep learning.

## Overview

HydroModels_PBM provides a unified pipeline for:

- Loading daily or flood-event CSV time series with configurable variable mapping
- Calibrating PBM parameters with SCE-UA (or other configured optimizers)
- Simulating streamflow with warmup and explicit calibration / validation periods
- Evaluating NSE, KGE, RMSE, HighRMSE, PFE, PTE, and related metrics
- Transferring calibrated parameters across basins (transplant experiments)

Supported models:

| Model | Description |
|-------|-------------|
| `GR4J`, `GR5J`, `GR6J` | GR family rainfall–runoff models |
| `GR4J-CemaNeige`, `GR5J-CemaNeige`, `GR6J-CemaNeige` | GR models with CemaNeige snow module |
| `XAJ`, `XAJ-mz` | XAJ and multi-zone XAJ variants |
| `XAJ-mz-CemaNeige` | Multi-zone XAJ with snow module |

## Data format

Experiments expect preprocessed basin CSVs under `input_path`, with columns mapped through `data_cfgs.variables` (e.g. precipitation, PET, observed discharge). Periods are configured explicitly:

- `calib_period` — optimization window
- `valid_period` — hold-out evaluation window
- `warmup_length` — spin-up steps before each simulation window

Both long-term daily data and flood-event subsets are supported via the data loader.

## Workflow

Each script defines a `SCRIPT_OVERRIDES` (or `BASE_CONFIG`) dict and calls `run_script_experiment()` from `hydromodels_pbm.workflow.pipeline`.

Typical outputs under `output_dir`:

- `best_params_all.json` — calibrated parameters per basin
- Per-basin logs, metrics CSVs, and simulated time series
- Experiment-level metric summaries across train / valid / test periods

For parameter transfer, use `run_transplant_experiment()` with a donor `params_json` and a `transplant_list_csv` of basin pairs.

## Running experiments

Paper experiment entry points live in [HydroScirpt](../HydroScirpt/). Each script imports this package and calls `run_script_experiment()` or `run_transplant_experiment()` with a `BASE_CONFIG` block.

Examples:

```bash
python HydroScirpt/Model/Sec1_ModelPerf/step1_XAJ_mz_Local.py
python HydroScirpt/Model/Sec2_ParamTrans/step1_XAJ_mz_Transplant.py
```

See [HydroScirpt/README.md](../HydroScirpt/README.md) for the full Sec0–Sec5 workflow. Update `data_cfgs.input_path` in each script for your data location.

## Package layout

```
hydromodels_pbm/
  config/       run_config, model parameter bounds
  dataset/      CSV loading, preprocessing, flood-event support
  model/        GR*, XAJ*, CemaNeige implementations
  workflow/     calibrate, simulate, evaluate, pipeline
  utils/        logging, normalization, runtime helpers
```

## Dependencies

NumPy, Pandas, and Numba (optional runtime warmup). No PyTorch required.
