# HydroCode_ModelEval_FloodEvent_dXAJ_Anhui

Code for "A Differentiable Xin'anjiang Model Incorporating Transfer Learning for Cross-Basin Flood Prediction under Data Scarcity"

Flood-event hydrological model evaluation for Anhui basins, comparing process-based models (PBM), deep-learning models (DLM), parameter transfer, and peak-focused training losses.

## Repository layout

| Directory | Role |
|-----------|------|
| [HydroModels_PBM](HydroModels_PBM/) | PBM calibration, simulation, and parameter transplant (GR*, XAJ*) |
| [HydroModels_DLM](HydroModels_DLM/) | LSTM and physics-guided dGR4J / dXAJ training and transfer |
| [HydroScirpt](HydroScirpt/) | Paper experiment scripts organized by section (Sec0–Sec5) |

See each subdirectory's README for setup, supported models, and run instructions.
