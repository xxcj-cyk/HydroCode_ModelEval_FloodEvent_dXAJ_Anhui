import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
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
    is_single_basin_ensemble,
    load_target_dirs_from_summary,
    publish_multi_target_ensemble_metrics,
    save_flood_period_csv,
    summarize_flood_regional,
)
from hydromodels_dlm.workflow.metrics import METRIC_REGISTRY

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_TAG = "SeqRegLSTM"
RESULT_ROOT = ROOT / "HydroScirpt/Result/Sec3_TransLearn2"
ENSEMBLE_DIRNAME = "Ensemble"
SCHEME_PREFIX = "SeqRegLSTM_TransLearn_"
METRICS = ["NSE", "KGE", "RMSE", "HIGHRMSE", "PFE", "PTE"]

SUBSETS = ("398", "535")

SUMMARY_COLUMNS = [
    "target_basin",
    "ensemble_dir",
    "n_schemes",
    "n_basins",
    *[f"{period}_{metric}" for period in PERIOD_ORDER for metric in METRICS],
]

CONTRIB_COLUMNS = [
    "target_basin",
    "run_name",
    "median_timesteps_total",
    "median_pct_total",
]


@dataclass(frozen=True)
class EnsembleJob:
    mode: str
    result_root: Path
    scheme_pattern: re.Pattern[str]
    summary_csv_name: str
    contrib_csv_name: str

    @property
    def ensemble_root(self) -> Path:
        return self.result_root / ENSEMBLE_DIRNAME

    @property
    def summary_csv(self) -> Path:
        return self.ensemble_root / self.summary_csv_name

    @property
    def contrib_csv(self) -> Path:
        return self.ensemble_root / self.contrib_csv_name

    def ensemble_dir_for(self, target_basin: str) -> Path:
        return self.ensemble_root / f"{SCHEME_PREFIX}{target_basin}"


def build_ensemble_job(subset: str) -> EnsembleJob:
    pattern = re.compile(rf"^{re.escape(SCHEME_PREFIX)}(Anhui_\d+)_(h\d+_.+)$")
    subset_tag = f"Anhui{subset}"
    prefix = f"SeqRegLSTM_TransLearn_reuse_{subset_tag}"
    return EnsembleJob(
        mode="reuse",
        result_root=RESULT_ROOT / prefix,
        scheme_pattern=pattern,
        summary_csv_name=f"{prefix}_Ensemble_Summary.csv",
        contrib_csv_name=f"{prefix}_Ensemble_Median_Contribution.csv",
    )


# ---------------------------------------------------------------------------
# Scheme grouping
# ---------------------------------------------------------------------------


def list_basin_dirs(exp_dir: Path) -> list[Path]:
    return sorted(
        path for path in exp_dir.iterdir()
        if path.is_dir() and path.name.startswith("Anhui_")
    )


def group_schemes_by_target(job: EnsembleJob) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(job.result_root.iterdir()):
        if not path.is_dir() or path.name == ENSEMBLE_DIRNAME:
            continue
        match = job.scheme_pattern.match(path.name)
        if match:
            groups[match.group(1)].append(path)
    return dict(groups)


# ---------------------------------------------------------------------------
# Median ensemble
# ---------------------------------------------------------------------------


def load_aligned_qsim_matrix(scheme_dirs: list[Path], basin_name: str) -> np.ndarray:
    ref = pd.read_csv(scheme_dirs[0] / basin_name / "timeseries.csv")
    ref_obs = ref[["time", "period", "event_id", "Qobs"]]
    columns = [pd.to_numeric(ref["Qsim"], errors="coerce").to_numpy()]

    for exp_dir in scheme_dirs[1:]:
        ts = pd.read_csv(exp_dir / basin_name / "timeseries.csv")
        if len(ts) != len(ref) or not ref_obs.equals(ts[ref_obs.columns]):
            raise ValueError(f"Timeseries mismatch: {exp_dir.name}/{basin_name}")
        columns.append(pd.to_numeric(ts["Qsim"], errors="coerce").to_numpy())

    return np.column_stack(columns)


def build_median_timeseries(scheme_dirs: list[Path], basin_name: str) -> pd.DataFrame:
    ref = pd.read_csv(scheme_dirs[0] / basin_name / "timeseries.csv")
    out = ref.copy()
    out["Qsim"] = np.median(load_aligned_qsim_matrix(scheme_dirs, basin_name), axis=1)
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


def build_target_ensemble(
    job: EnsembleJob, target_basin: str, scheme_dirs: list[Path]
) -> tuple[pd.DataFrame, Path]:
    out_dir = job.ensemble_dir_for(target_basin)
    out_dir.mkdir(parents=True, exist_ok=True)

    basin_names = [path.name for path in list_basin_dirs(scheme_dirs[0])]
    rows_by_period = {period: [] for period in PERIOD_ORDER}
    summary_rows = []

    for basin_name in basin_names:
        basin_dir = out_dir / basin_name
        basin_dir.mkdir(parents=True, exist_ok=True)
        timeseries = build_median_timeseries(scheme_dirs, basin_name)
        timeseries.to_csv(basin_dir / "timeseries.csv", index=False, encoding="utf-8-sig")
        for row in save_basin_metrics(basin_dir, basin_name, timeseries):
            rows_by_period[row["period"]].append(row)

    for period in PERIOD_ORDER:
        basin_rows = rows_by_period[period]
        if not basin_rows:
            continue
        period_rows = [
            {key: value for key, value in row.items() if key != "period"}
            for row in basin_rows
        ]
        regional_rows = summarize_flood_regional(out_dir, basin_names, period)
        save_flood_period_csv(
            out_dir / PERIOD_METRICS_FILES[period], period_rows, regional_rows
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
        out_dir / "ensemble_summary_metrics.csv", index=False, encoding="utf-8-sig"
    )
    return summary, out_dir


def collect_median_contribution(
    job: EnsembleJob, target_basin: str, scheme_dirs: list[Path]
) -> list[dict]:
    counts = np.zeros(len(scheme_dirs), dtype=np.int64)
    total_steps = 0

    for basin_dir in list_basin_dirs(scheme_dirs[0]):
        qsim = load_aligned_qsim_matrix(scheme_dirs, basin_dir.name)
        median = np.median(qsim, axis=1, keepdims=True)
        counts += np.isclose(qsim, median, rtol=0, atol=1e-12).sum(axis=0).astype(np.int64)
        total_steps += len(qsim)

    rows = []
    for exp_dir, count in zip(scheme_dirs, counts):
        run_name = job.scheme_pattern.match(exp_dir.name).group(2)
        rows.append(
            {
                "target_basin": target_basin,
                "run_name": run_name,
                "median_timesteps_total": int(count),
                "median_pct_total": round(100.0 * count / total_steps, 4)
                if total_steps
                else 0.0,
            }
        )
    return rows


def build_summary_row(
    target_basin: str,
    out_dir: Path,
    n_schemes: int,
    n_basins: int,
    summary: pd.DataFrame,
) -> dict:
    row = {
        "target_basin": target_basin,
        "ensemble_dir": str(out_dir.relative_to(ROOT)),
        "n_schemes": n_schemes,
        "n_basins": n_basins,
    }
    median_rows = summary.loc[summary["stat"].astype(str).str.upper() == "MEDIAN"]
    for _, item in median_rows.iterrows():
        period = str(item["period"]).lower()
        for metric in METRICS:
            row[f"{period}_{metric}"] = float(item[metric])
    return row


def save_summary_tables(
    job: EnsembleJob,
    summary_rows: list[dict],
    contrib_rows: list[dict],
    replace_basin: str | None,
) -> None:
    summary_df = pd.DataFrame(summary_rows)[SUMMARY_COLUMNS]
    contrib_df = pd.DataFrame(contrib_rows)[CONTRIB_COLUMNS]

    if replace_basin is not None:
        if job.summary_csv.exists():
            kept = pd.read_csv(job.summary_csv)[SUMMARY_COLUMNS]
            summary_df = pd.concat(
                [kept.loc[kept["target_basin"] != replace_basin], summary_df],
                ignore_index=True,
            )
        if job.contrib_csv.exists():
            kept = pd.read_csv(job.contrib_csv)[CONTRIB_COLUMNS]
            contrib_df = pd.concat(
                [kept.loc[kept["target_basin"] != replace_basin], contrib_df],
                ignore_index=True,
            )

    summary_df.sort_values("target_basin").to_csv(
        job.summary_csv, index=False, encoding="utf-8-sig"
    )
    contrib_df.sort_values(["target_basin", "run_name"]).to_csv(
        job.contrib_csv, index=False, encoding="utf-8-sig"
    )


def run_median_ensemble(job: EnsembleJob, replace_basin: str | None = None) -> None:
    groups = group_schemes_by_target(job)
    if not groups:
        raise FileNotFoundError(f"No scheme runs under {job.result_root}")
    if replace_basin is not None:
        if replace_basin not in groups:
            raise ValueError(f"Unknown target basin for {job.mode}: {replace_basin}")
        groups = {replace_basin: groups[replace_basin]}

    print("=" * 72)
    print(f"Sec3_TransLearn2 — {MODEL_TAG} translearn median ensemble [{job.mode}]")
    print(f"Result root: {job.result_root}")
    print(f"Ensemble root: {job.ensemble_root}")
    print(f"Target groups: {len(groups)}")
    print("-" * 72)

    job.ensemble_root.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    contrib_rows = []

    for target_basin in sorted(groups):
        scheme_dirs = groups[target_basin]
        n_basins = len(list_basin_dirs(scheme_dirs[0]))
        print(
            f"[{target_basin}] schemes={len(scheme_dirs)} basins={n_basins} "
            f"-> {job.ensemble_dir_for(target_basin).name}"
        )

        summary, out_dir = build_target_ensemble(job, target_basin, scheme_dirs)
        summary_rows.append(
            build_summary_row(target_basin, out_dir, len(scheme_dirs), n_basins, summary)
        )
        contrib_rows.extend(collect_median_contribution(job, target_basin, scheme_dirs))

        for _, row in summary.loc[summary["stat"].astype(str).str.upper() == "MEDIAN"].iterrows():
            print(
                f"  {row['period']:5s}  NSE={row['NSE']}  KGE={row['KGE']}  "
                f"RMSE={row['RMSE']}  HIGHRMSE={row['HIGHRMSE']}  "
                f"PFE={row['PFE']}  PTE={row['PTE']}"
            )

    save_summary_tables(job, summary_rows, contrib_rows, replace_basin)

    target_dirs = load_target_dirs_from_summary(job.summary_csv, ROOT)
    if target_dirs:
        publish_multi_target_ensemble_metrics(
            job.ensemble_root,
            target_dirs,
            single_basin_target=is_single_basin_ensemble(job.summary_csv),
        )

    print("-" * 72)
    print(f"Ensemble summary: {job.summary_csv}")
    print(f"Median contribution: {job.contrib_csv}")
    if target_dirs:
        print(f"Root period metrics: {job.ensemble_root}")
    print(f"Finished groups: {len(summary_rows)}")
    print("=" * 72)


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
    parser.add_argument(
        "--test",
        metavar="BASIN",
        nargs="?",
        const="Anhui_50406910",
        help="Build ensemble for one target basin (default: Anhui_50406910)",
    )
    args = parser.parse_args()

    setup_logging()
    for subset in args.subset:
        run_median_ensemble(build_ensemble_job(subset), replace_basin=args.test)


if __name__ == "__main__":
    main()
