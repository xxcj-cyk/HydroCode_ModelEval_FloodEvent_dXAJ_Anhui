import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "HydroModels_DLM"))

from hydromodels_dlm.utils.logging import setup_logging
from hydromodels_dlm.workflow.artifacts import (
    EVENT_METRICS,
    PERIOD_METRICS_FILES,
    PERIOD_ORDER,
    STAT_ORDER,
    save_flood_period_csv,
    summarize_flood_regional,
)
from hydromodels_dlm.workflow.metrics import METRIC_REGISTRY

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_TAG = "dXAJ-mz_Local"
SCHEME_PREFIX = f"{MODEL_TAG}_"
SEC0_ROOT = ROOT / "HydroScirpt/Result/Sec0_ModelConfig" / MODEL_TAG
SEC1_ROOT = ROOT / "HydroScirpt/Result/Sec1_ModelPerf" / MODEL_TAG

TOP9_CSV = SEC1_ROOT / f"{MODEL_TAG}_PerBasin_Top9_ValidLoss_Hyperparams.csv"
MEDIAN_CONTRIB_CSV = SEC1_ROOT / f"{MODEL_TAG}_Top9_Ensemble_Median_Contribution.csv"
ENSEMBLE_DIR = SEC1_ROOT / "Ensemble_Top9"

TOP_K = 9
METRICS = ["NSE", "KGE", "RMSE", "HIGHRMSE", "PFE", "PTE"]
BASIN_ARTIFACTS = (
    "best_metrics.csv",
    "best_model.pth",
    "best_params.json",
    "event_metrics.csv",
    "normalization_scaler.json",
    "run_config.json",
    "timeseries.csv",
    "training_log.txt",
)

VALID_LOSS_PATTERN = re.compile(
    r"Valid Loss=([0-9]+(?:\.[0-9]+)?(?:e[+-]?\d+)?)",
    re.IGNORECASE,
)
RUN_NAME_PATTERN = re.compile(
    r"h(\d+)_dr(\d+)_wu(\d+)_b(\d+)_lr(\d+)_seed(\d+)",
)

TOP9_COLUMNS = [
    "basin_id",
    "rank",
    "run_name",
    "hidden_size",
    "dropout",
    "warmup_length",
    "batch_size",
    "lr",
    "seed",
    "valid_loss",
    "valid_rmse",
    "valid_nse",
    "valid_kge",
]

# ---------------------------------------------------------------------------
# Per-basin ranking (Sec0)
# ---------------------------------------------------------------------------


def list_scheme_dirs(root: Path) -> list[Path]:
    return sorted(
        path for path in root.iterdir()
        if path.is_dir() and path.name.startswith(SCHEME_PREFIX)
    )


def parse_run_name(run_name: str) -> dict:
    match = RUN_NAME_PATTERN.match(run_name)
    if not match:
        raise ValueError(f"Cannot parse run name: {run_name}")
    h, dr, wu, b, lr_tag, seed = match.groups()
    return {
        "run_name": run_name,
        "hidden_size": int(h),
        "dropout": int(dr) / 10.0,
        "warmup_length": int(wu),
        "batch_size": int(b),
        "lr": int(lr_tag) / (10 ** len(lr_tag)),
        "seed": int(seed),
    }


def read_basin_valid_loss(training_log: Path) -> float:
    matches = VALID_LOSS_PATTERN.findall(training_log.read_text(encoding="utf-8"))
    if not matches:
        raise ValueError(f"Missing valid loss in {training_log}")
    return min(float(value) for value in matches)


def read_basin_valid_metrics(metrics_csv: Path) -> dict[str, float]:
    df = pd.read_csv(metrics_csv)
    row = df[(df["period"] == "valid") & (df["stat"] == "MEDIAN")]
    if row.empty:
        raise ValueError(f"Missing valid MEDIAN row in {metrics_csv}")
    item = row.iloc[0]
    return {
        "valid_rmse": float(item["RMSE"]),
        "valid_nse": float(item["NSE"]),
        "valid_kge": float(item["KGE"]),
    }


def collect_basin_scheme_rows() -> pd.DataFrame:
    rows: list[dict] = []
    for exp_dir in list_scheme_dirs(SEC0_ROOT):
        run_name = exp_dir.name.removeprefix(SCHEME_PREFIX)
        meta = parse_run_name(run_name)
        for basin_dir in sorted(
            child for child in exp_dir.iterdir() if child.name.startswith("Anhui_")
        ):
            log_path = basin_dir / "training_log.txt"
            metrics_path = basin_dir / "best_metrics.csv"
            if not log_path.is_file() or not metrics_path.is_file():
                continue
            try:
                valid_loss = read_basin_valid_loss(log_path)
                metrics = read_basin_valid_metrics(metrics_path)
            except (ValueError, OSError) as exc:
                print(f"Skip {exp_dir.name}/{basin_dir.name}: {exc}")
                continue
            rows.append({
                "basin_id": basin_dir.name,
                "valid_loss": valid_loss,
                **meta,
                **metrics,
            })
    if not rows:
        raise RuntimeError(f"No basin-scheme rows collected under {SEC0_ROOT}")
    return pd.DataFrame(rows)


def build_per_basin_top9(all_rows: pd.DataFrame) -> pd.DataFrame:
    ranked = all_rows.sort_values(
        ["basin_id", "valid_loss", "valid_rmse", "run_name"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)
    ranked["rank"] = ranked.groupby("basin_id").cumcount() + 1
    top9 = ranked[ranked["rank"] <= TOP_K].sort_values(["basin_id", "rank"])
    return top9[TOP9_COLUMNS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Copy Top9 basin artifacts (Sec0 -> Sec1)
# ---------------------------------------------------------------------------


def scheme_dir_name(run_name: str) -> str:
    return f"{SCHEME_PREFIX}{run_name}"


def sec0_basin_dir(run_name: str, basin_id: str) -> Path:
    return SEC0_ROOT / scheme_dir_name(run_name) / basin_id


def sec1_scheme_dir(basin_id: str, run_name: str) -> Path:
    return SEC1_ROOT / basin_id / scheme_dir_name(run_name)


def clear_sec1_layout() -> None:
    SEC1_ROOT.mkdir(parents=True, exist_ok=True)
    for path in list(SEC1_ROOT.iterdir()):
        if path.is_file() and path.suffix.lower() == ".csv":
            path.unlink()
        elif path.is_dir() and (
            path.name == "Ensemble_Top9"
            or path.name.startswith("Anhui_")
            or path.name.startswith(SCHEME_PREFIX)
        ):
            shutil.rmtree(path)


def copy_basin_artifacts(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    missing = [name for name in BASIN_ARTIFACTS if not (src_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing artifacts in {src_dir}: {missing}")
    for name in BASIN_ARTIFACTS:
        shutil.copy2(src_dir / name, dst_dir / name)


def copy_per_basin_top9(top9: pd.DataFrame) -> dict[str, list[Path]]:
    basin_scheme_dirs: dict[str, list[Path]] = {}
    for basin_id, basin_rows in top9.groupby("basin_id", sort=True):
        scheme_dirs: list[Path] = []
        for row in basin_rows.itertuples(index=False):
            run_name = str(row.run_name)
            src = sec0_basin_dir(run_name, basin_id)
            dst = sec1_scheme_dir(basin_id, run_name)
            copy_basin_artifacts(src, dst)
            scheme_dirs.append(dst)
        basin_scheme_dirs[str(basin_id)] = scheme_dirs
    return basin_scheme_dirs


# ---------------------------------------------------------------------------
# Median ensemble (per basin)
# ---------------------------------------------------------------------------


def load_aligned_qsim(scheme_dirs: list[Path]) -> tuple[pd.DataFrame, np.ndarray]:
    ref = pd.read_csv(scheme_dirs[0] / "timeseries.csv")
    ref_obs = ref[["time", "period", "event_id", "Qobs"]]
    columns = [pd.to_numeric(ref["Qsim"], errors="coerce").to_numpy()]

    for scheme_dir in scheme_dirs[1:]:
        ts = pd.read_csv(scheme_dir / "timeseries.csv")
        if len(ts) != len(ref) or not ref_obs.equals(ts[ref_obs.columns]):
            raise ValueError(f"Timeseries mismatch: {scheme_dir}")
        columns.append(pd.to_numeric(ts["Qsim"], errors="coerce").to_numpy())

    return ref, np.column_stack(columns)


def build_median_timeseries(scheme_dirs: list[Path]) -> pd.DataFrame:
    ref, qsim_matrix = load_aligned_qsim(scheme_dirs)
    out = ref.copy()
    out["Qsim"] = np.median(qsim_matrix, axis=1)
    return out


def compute_event_metrics(qsim: np.ndarray, qobs: np.ndarray) -> dict[str, float]:
    return {name: float(METRIC_REGISTRY[name](qsim, qobs)) for name in METRICS}


def format_metric_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for name in METRICS:
        if name not in out.columns:
            continue
        prec = 2 if name == "PTE" else 4
        out[name] = out[name].map(lambda value: f"{float(value):.{prec}f}")
    return out


def write_basin_metrics(basin_dir: Path, basin_id: str, timeseries: pd.DataFrame) -> list[dict]:
    event_rows = []
    period_stats: dict[str, dict[str, dict[str, float]]] = {}

    for period in PERIOD_ORDER:
        period_df = timeseries.loc[timeseries["period"].astype(str).str.lower() == period]
        if period_df.empty:
            continue

        event_metric_rows = []
        for event_id, group in period_df.groupby("event_id", sort=False):
            metrics = compute_event_metrics(
                group["Qsim"].to_numpy(dtype=float),
                group["Qobs"].to_numpy(dtype=float),
            )
            event_metric_rows.append(metrics)
            event_rows.append({
                "period": period,
                "basin_id": basin_id,
                "event_id": event_id,
                **metrics,
            })

        period_stats[period] = {}
        for stat in STAT_ORDER:
            reducer = np.nanmedian if stat == "MEDIAN" else np.nanmean
            period_stats[period][stat] = {
                name: float(reducer([row[name] for row in event_metric_rows]))
                for name in METRICS
            }

    format_metric_frame(pd.DataFrame(event_rows)).to_csv(
        basin_dir / EVENT_METRICS, index=False, encoding="utf-8-sig"
    )

    basin_rows = []
    for stat in STAT_ORDER:
        for period in PERIOD_ORDER:
            metrics = period_stats.get(period, {}).get(stat)
            if metrics:
                basin_rows.append({
                    "stat": stat, "period": period, "basin_id": basin_id, **metrics
                })

    format_metric_frame(pd.DataFrame(basin_rows)).to_csv(
        basin_dir / "best_metrics.csv", index=False, encoding="utf-8-sig"
    )
    return basin_rows


def write_ensemble_outputs(basin_scheme_dirs: dict[str, list[Path]]) -> pd.DataFrame:
    ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)
    rows_by_period = {period: [] for period in PERIOD_ORDER}
    summary_rows = []

    for basin_id in sorted(basin_scheme_dirs):
        scheme_dirs = basin_scheme_dirs[basin_id]
        basin_dir = ENSEMBLE_DIR / basin_id
        basin_dir.mkdir(parents=True, exist_ok=True)

        timeseries = build_median_timeseries(scheme_dirs)
        timeseries.to_csv(basin_dir / "timeseries.csv", index=False, encoding="utf-8-sig")
        for row in write_basin_metrics(basin_dir, basin_id, timeseries):
            rows_by_period[row["period"]].append(row)

    basin_names = sorted(basin_scheme_dirs)
    for period in PERIOD_ORDER:
        basin_rows = rows_by_period[period]
        if not basin_rows:
            continue
        period_rows = [{k: v for k, v in row.items() if k != "period"} for row in basin_rows]
        regional_rows = summarize_flood_regional(ENSEMBLE_DIR, basin_names, period)
        save_flood_period_csv(
            ENSEMBLE_DIR / PERIOD_METRICS_FILES[period],
            period_rows,
            regional_rows,
        )
        for stat in STAT_ORDER:
            match = next((row for row in regional_rows if row["stat"].upper() == stat), None)
            if match:
                summary_rows.append({
                    "period": period,
                    "stat": stat,
                    **{name: match[name] for name in METRICS},
                    "basin_id": match.get("basin_id", ""),
                })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        ENSEMBLE_DIR / "ensemble_summary_metrics.csv", index=False, encoding="utf-8-sig"
    )
    return summary


def write_median_contribution(
    basin_scheme_dirs: dict[str, list[Path]],
    top9: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for basin_id, scheme_dirs in sorted(basin_scheme_dirs.items()):
        qsim_matrix = load_aligned_qsim(scheme_dirs)[1]
        median = np.median(qsim_matrix, axis=1, keepdims=True)
        hits = np.isclose(qsim_matrix, median, rtol=0, atol=1e-12).sum(axis=0).astype(int)
        total_steps = len(qsim_matrix)
        basin_top9 = top9.loc[top9["basin_id"] == basin_id].sort_values("rank")
        for index, row in enumerate(basin_top9.itertuples(index=False)):
            rows.append({
                "basin_id": basin_id,
                "rank": int(row.rank),
                "run_name": str(row.run_name),
                "median_timesteps_total": int(hits[index]),
                "median_pct_total": round(100.0 * hits[index] / total_steps, 4),
            })
    df = pd.DataFrame(rows)
    df.to_csv(MEDIAN_CONTRIB_CSV, index=False, encoding="utf-8-sig")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    setup_logging()

    print("=" * 60)
    print(f"Sec1 — {MODEL_TAG} per-basin Top9 copy / ensemble")
    print(f"Sec0 root: {SEC0_ROOT}")
    print(f"Sec1 root: {SEC1_ROOT}")
    print("-" * 60)

    all_rows = collect_basin_scheme_rows()
    top9 = build_per_basin_top9(all_rows)
    basin_count = top9["basin_id"].nunique()
    unique_schemes = top9["run_name"].nunique()

    print(f"Basins: {basin_count}")
    print(f"Top9 rows: {len(top9)}")
    print(f"Unique schemes in Top9 union: {unique_schemes}")

    clear_sec1_layout()
    top9.to_csv(TOP9_CSV, index=False, encoding="utf-8-sig")
    basin_scheme_dirs = copy_per_basin_top9(top9)
    ensemble_summary = write_ensemble_outputs(basin_scheme_dirs)
    write_median_contribution(basin_scheme_dirs, top9)

    print(f"Saved Top9 CSV: {TOP9_CSV}")
    print(f"Median contribution: {MEDIAN_CONTRIB_CSV}")
    print(f"Ensemble dir: {ENSEMBLE_DIR}")
    median_rows = ensemble_summary.loc[ensemble_summary["stat"].str.upper() == "MEDIAN"]
    for _, row in median_rows.iterrows():
        print(
            f"  {row['period']:5s}  NSE={row['NSE']}  KGE={row['KGE']}  "
            f"RMSE={row['RMSE']}  HIGHRMSE={row['HIGHRMSE']}  "
            f"PFE={row['PFE']}  PTE={row['PTE']}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
