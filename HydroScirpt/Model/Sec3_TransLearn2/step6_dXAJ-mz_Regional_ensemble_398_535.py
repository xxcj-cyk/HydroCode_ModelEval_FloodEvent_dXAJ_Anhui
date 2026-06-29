import argparse
import csv
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

MODEL_TAG = "dXAJ-mz_Regional"
SCHEME_PREFIX = f"{MODEL_TAG}_"
TOP9_CSV = (
    ROOT
    / "HydroScirpt/Result/Sec1_ModelPerf/dXAJ-mz_Regional"
    / "dXAJ-mz_Regional_Top9_ValidLoss_Hyperparams.csv"
)
RESULT_ROOT = ROOT / "HydroScirpt/Result/Sec3_TransLearn2"
ENSEMBLE_DIRNAME = "Ensemble"
METRICS = ["NSE", "KGE", "RMSE", "HIGHRMSE", "PFE", "PTE"]

SUBSETS = {
    "398": {"result_root": RESULT_ROOT / f"{MODEL_TAG}_Anhui398"},
    "535": {"result_root": RESULT_ROOT / f"{MODEL_TAG}_Anhui535"},
}


# ---------------------------------------------------------------------------
# Scheme discovery
# ---------------------------------------------------------------------------


def load_top9_run_names() -> list[str]:
    with open(TOP9_CSV, encoding="utf-8-sig", newline="") as handle:
        return [row["run_name"] for row in csv.DictReader(handle)]


def list_basin_dirs(exp_dir: Path) -> list[Path]:
    return sorted(
        path for path in exp_dir.iterdir()
        if path.is_dir() and path.name.startswith("Anhui_")
    )


def resolve_top9_dirs(result_root: Path) -> list[Path]:
    dirs = []
    for run_name in load_top9_run_names():
        scheme_dir = result_root / f"{SCHEME_PREFIX}{run_name}"
        if not scheme_dir.is_dir():
            raise FileNotFoundError(f"Missing scheme dir: {scheme_dir}")
        dirs.append(scheme_dir)
    return dirs


# ---------------------------------------------------------------------------
# Median ensemble
# ---------------------------------------------------------------------------


def load_aligned_qsim(
    top9_dirs: list[Path], basin_name: str
) -> tuple[pd.DataFrame, np.ndarray]:
    ref = pd.read_csv(top9_dirs[0] / basin_name / "timeseries.csv")
    ref_obs = ref[["time", "period", "event_id", "Qobs"]]
    columns = [pd.to_numeric(ref["Qsim"], errors="coerce").to_numpy()]

    for exp_dir in top9_dirs[1:]:
        ts = pd.read_csv(exp_dir / basin_name / "timeseries.csv")
        if len(ts) != len(ref) or not ref_obs.equals(ts[ref_obs.columns]):
            raise ValueError(f"Timeseries mismatch: {exp_dir.name}/{basin_name}")
        columns.append(pd.to_numeric(ts["Qsim"], errors="coerce").to_numpy())

    return ref, np.column_stack(columns)


def build_median_timeseries(top9_dirs: list[Path], basin_name: str) -> pd.DataFrame:
    ref, qsim_matrix = load_aligned_qsim(top9_dirs, basin_name)
    out = ref.copy()
    out["Qsim"] = np.median(qsim_matrix, axis=1)
    return out


def compute_event_metrics(qsim: np.ndarray, qobs: np.ndarray) -> dict[str, float]:
    return {name: float(METRIC_REGISTRY[name](qsim, qobs)) for name in METRICS}


def format_metric_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for name in METRICS:
        if name not in out.columns:
            continue
        prec = 2 if name == "PTE" else 4
        out[name] = out[name].map(lambda value: f"{float(value):.{prec}f}")
    return out


def save_basin_metrics(
    basin_dir: Path, basin_id: str, timeseries: pd.DataFrame
) -> list[dict]:
    event_rows = []
    period_stats: dict[str, dict[str, dict[str, float]]] = {}

    for period in PERIOD_ORDER:
        period_df = timeseries.loc[
            timeseries["period"].astype(str).str.lower() == period
        ]
        if period_df.empty:
            continue

        event_metric_rows = []
        for event_id, group in period_df.groupby("event_id", sort=False):
            metrics = compute_event_metrics(
                group["Qsim"].to_numpy(dtype=float),
                group["Qobs"].to_numpy(dtype=float),
            )
            event_metric_rows.append(metrics)
            event_rows.append(
                {"period": period, "basin_id": basin_id, "event_id": event_id, **metrics}
            )

        period_stats[period] = {}
        for stat in STAT_ORDER:
            reducer = np.nanmedian if stat == "MEDIAN" else np.nanmean
            period_stats[period][stat] = {
                name: float(reducer([row[name] for row in event_metric_rows]))
                for name in METRICS
            }

    format_metric_table(pd.DataFrame(event_rows)).to_csv(
        basin_dir / EVENT_METRICS, index=False, encoding="utf-8-sig"
    )

    basin_rows = []
    for stat in STAT_ORDER:
        for period in PERIOD_ORDER:
            metrics = period_stats.get(period, {}).get(stat)
            if metrics:
                basin_rows.append(
                    {"stat": stat, "period": period, "basin_id": basin_id, **metrics}
                )

    format_metric_table(pd.DataFrame(basin_rows)).to_csv(
        basin_dir / "best_metrics.csv", index=False, encoding="utf-8-sig"
    )
    return basin_rows


def write_median_contribution(
    top9_dirs: list[Path], run_names: list[str], contrib_csv: Path
) -> None:
    counts = np.zeros(len(top9_dirs), dtype=np.int64)
    total_steps = 0

    for basin_dir in list_basin_dirs(top9_dirs[0]):
        qsim_matrix = load_aligned_qsim(top9_dirs, basin_dir.name)[1]
        median = np.median(qsim_matrix, axis=1, keepdims=True)
        counts += (
            np.isclose(qsim_matrix, median, rtol=0, atol=1e-12)
            .sum(axis=0)
            .astype(np.int64)
        )
        total_steps += len(qsim_matrix)

    rows = [
        {
            "rank": rank,
            "run_name": run_name,
            "median_timesteps_total": int(counts[rank - 1]),
            "median_pct_total": round(100.0 * counts[rank - 1] / total_steps, 4),
        }
        for rank, run_name in enumerate(run_names, start=1)
    ]
    pd.DataFrame(rows).to_csv(contrib_csv, index=False, encoding="utf-8-sig")


def run_subset(subset: str) -> None:
    result_root = SUBSETS[subset]["result_root"]
    ensemble_dir = result_root / ENSEMBLE_DIRNAME
    contrib_csv = (
        ensemble_dir / f"{MODEL_TAG}_Anhui{subset}_Top9_Ensemble_Median_Contribution.csv"
    )
    run_names = load_top9_run_names()
    top9_dirs = resolve_top9_dirs(result_root)

    print("=" * 60)
    print(f"Sec3_TransLearn2 — {MODEL_TAG} Top9 median ensemble (Anhui{subset})")
    print(f"Top9 CSV: {TOP9_CSV}")
    print(f"Result root: {result_root}")
    print(f"Schemes: {len(top9_dirs)}")
    print("-" * 60)

    ensemble_dir.mkdir(parents=True, exist_ok=True)
    basin_names = [path.name for path in list_basin_dirs(top9_dirs[0])]
    rows_by_period = {period: [] for period in PERIOD_ORDER}
    summary_rows = []

    for basin_name in basin_names:
        basin_dir = ensemble_dir / basin_name
        basin_dir.mkdir(parents=True, exist_ok=True)
        timeseries = build_median_timeseries(top9_dirs, basin_name)
        timeseries.to_csv(
            basin_dir / "timeseries.csv", index=False, encoding="utf-8-sig"
        )
        for row in save_basin_metrics(basin_dir, basin_name, timeseries):
            rows_by_period[row["period"]].append(row)

    for period in PERIOD_ORDER:
        basin_rows = rows_by_period[period]
        if not basin_rows:
            continue
        period_rows = [
            {k: v for k, v in row.items() if k != "period"} for row in basin_rows
        ]
        regional_rows = summarize_flood_regional(ensemble_dir, basin_names, period)
        save_flood_period_csv(
            ensemble_dir / PERIOD_METRICS_FILES[period], period_rows, regional_rows
        )
        for stat in STAT_ORDER:
            match = next(
                (row for row in regional_rows if row["stat"].upper() == stat), None
            )
            if match:
                summary_rows.append(
                    {
                        "period": period,
                        "stat": stat,
                        **{name: match[name] for name in METRICS},
                        "basin_id": match.get("basin_id", ""),
                    }
                )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        ensemble_dir / "ensemble_summary_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_median_contribution(top9_dirs, run_names, contrib_csv)

    print(f"Ensemble dir: {ensemble_dir}")
    print(f"Median contribution: {contrib_csv}")
    median_rows = summary.loc[summary["stat"].str.upper() == "MEDIAN"]
    for _, row in median_rows.iterrows():
        print(
            f"  {row['period']:5s}  NSE={row['NSE']}  KGE={row['KGE']}  "
            f"RMSE={row['RMSE']}  HIGHRMSE={row['HIGHRMSE']}  "
            f"PFE={row['PFE']}  PTE={row['PTE']}"
        )
    print("=" * 60)


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
        run_subset(subset)


if __name__ == "__main__":
    main()
