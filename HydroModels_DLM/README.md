# HydroModels_DLM

Deep-learning hydrological modeling workflow aligned with [HydroModels_PBM](../HydroModels_PBM). Supports pure data-driven LSTM models and physics-guided models where an LSTM predicts time-varying PBM parameters.

## Overview

HydroModels_DLM covers the full train–evaluate–transfer loop:

- **Data-driven**: `SeqRegLSTM` — LSTM maps dynamic forcings (+ optional static attributes) to streamflow
- **Physics-guided**: `dGR4J`, `dGR4J-CemaNeige`, `dXAJ`, `dXAJ-mz`, `dXAJ-mz-CemaNeige` — LSTM outputs PBM parameters; differentiable PBM stays in the autograd graph
- **Training strategies**: `RegionalTrain` (one shared model across basins) or `LocalTrain` (one model per basin)
- **Transfer learning**: zero-shot, fine-tune, and scaler reuse via `workflow/transfer.py`
- **Losses**: RMSE, NSE, HighRMSE, PFE, and composite `PeakFocusedLoss` for flood-peak emphasis

Data uses the same CSV layout as PBM, with explicit `train_period`, `valid_period`, and `test_period`. Default `warmup_length` is 365 days for long-term runs; flood-event scripts often use shorter warmup.

## Physics-guided architecture

Training follows a two-stage differentiable design:

1. LSTM reads the full warmup + forecast window (`input_layout: dynamic_static`) and maps the last hidden state to bounded PBM parameters.
2. The PBM forward pass produces simulated discharge; streamflow error backpropagates into LSTM weights.

Performance optimizations on the PBM side:

- Basin-wise vectorization (unit hydro weights, `conv1d` routing)
- Truncated backprop via `training_cfgs.pbm_grad_steps` (default = `forecast_length`; `0` = full sequence)
- Numba CPU warmup for the prefix (364 days) with PyTorch on the differentiable tail; falls back to pure PyTorch if Numba is unavailable

## Environment

Use a conda env with a **CUDA build of PyTorch** (not the default CPU wheel):

```bash
conda activate hydro
cd HydroModels_DLM
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If CUDA torch is missing:

```bash
pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

On Windows, run HydroScirpt entry scripts from the repository root; `pipeline` import handles OpenMP / MKL conflicts.

## Outputs

**RegionalTrain** — shared artifacts under `results/<experiment>/`:

- `best_model.pth`, `normalization_scaler.json`, `training_log.txt`
- Per-basin subdirs with `best_metrics.csv`, `timeseries.csv`, etc.

**LocalTrain** — per-basin dirs under `results/<experiment>/<basin_id>/` with their own checkpoint and scaler; experiment root holds aggregated metric CSVs.

## Running experiments

Paper experiment entry points live in [HydroScirpt](../HydroScirpt/). Each script imports this package and calls `run_script_experiment()` with a `BASE_CONFIG` block.

Examples:

```bash
python HydroScirpt/Model/Sec1_ModelPerf/step2_SeqRegLSTM_Regional_ensemble.py
python HydroScirpt/Model/Sec1_ModelPerf/step3_dXAJ-mz_Regional_ensemble.py
python HydroScirpt/Model/Sec5_Test/step2_SeqRegLSTM_Regional_PeakFocused_hypermore.py
```

See [HydroScirpt/README.md](../HydroScirpt/README.md) for the full Sec0–Sec5 workflow. Set `input_path`, `basin_ids`, and `model_hyperparam` (`input_size` / `output_size` must match data dimensions).

## Package layout

```
hydromodels_dlm/
  config/       run_config, model_config (PBM parameter bounds)
  dataset/      loading, normalization, sliding-window Dataset
  model/        SeqRegLSTM, dGR4J, dXAJ, physics_guided wrappers
  workflow/     train, evaluate, transfer, pipeline, losses
  utils/        logging, GPU, random seed
```
