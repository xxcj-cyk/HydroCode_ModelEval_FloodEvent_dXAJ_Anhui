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

MODEL_TAG = "SeqRegLSTM_Regional"
SEC0_ROOT = ROOT / "HydroScirpt/Result/Sec0_ModelConfig" / MODEL_TAG
SEC1_ROOT = ROOT / "HydroScirpt/Result/Sec1_ModelPerf" / MODEL_TAG
SCHEME_PREFIX = f"{MODEL_TAG}_"

RANKED_CSV = SEC1_ROOT / f"{MODEL_TAG}_All_Schemes_Ranked_by_ValidLoss.csv"
TOP9_CSV = SEC1_ROOT / f"{MODEL_TAG}_Top9_ValidLoss_Hyperparams.csv"
MEDIAN_CONTRIB_CSV = SEC1_ROOT / f"{MODEL_TAG}_Top9_Ensemble_Median_Contribution.csv"
ENSEMBLE_DIR = SEC1_ROOT / "Ensemble_Top9"

METRICS = ["NSE", "KGE", "RMSE", "HIGHRMSE", "PFE", "PTE"]
TOP_K = 9
VALID_LOSS_PATTERN = re.compile(
    r"Best valid loss:\s*([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)
RUN_NAME_PATTERN = re.compile(r"h(\d+)_dr(\d+)_wu(\d+)_b(\d+)_lr(\d+)_seed(\d+)")

RANKED_COLUMNS = [
    "period",
    "rank",
    "run_name",
    "hidden_size",
    "dropout",
    "warmup_length",
    "batch_size",
    "lr",
    "seed",
    "valid_loss",
    "valid_rmse_median",
    *METRICS,
]
TOP9_COLUMNS = [
    "rank",
    "run_name",
    "hidden_size",
    "dropout",
    "batch_size",
    "warmup_length",
    "lr",
    "seed",
    "valid_loss",
    "valid_rmse_median",
]

# ---------------------------------------------------------------------------
# Scheme discovery
# ---------------------------------------------------------------------------


def list_scheme_dirs(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and path.name.startswith(SCHEME_PREFIX)
    )


def list_basin_dirs(exp_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in exp_dir.iterdir()
        if path.is_dir() and path.name.startswith("Anhui_")
    )


def parse_run_name(scheme_dir_name: str) -> dict:
    name = scheme_dir_name.removeprefix(SCHEME_PREFIX)
    match = RUN_NAME_PATTERN.match(name)
    if not match:
        raise ValueError(f"Cannot parse scheme name: {scheme_dir_name}")
    h, dr, wu, b, lr_tag, seed = match.groups()
    return {
        "run_name": name,
        "hidden_size": int(h),
        "dropout": int(dr) / 10.0,
        "warmup_length": int(wu),
        "batch_size": int(b),
        "lr": int(lr_tag) / (10 ** len(lr_tag)),
        "seed": int(seed),
    }


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def read_best_valid_loss(exp_dir: Path) -> float:
    log_path = exp_dir / "training_log.txt"
    if not log_path.is_file():
        raise FileNotFoundError(f"Missing training log: {log_path}")
    matches = VALID_LOSS_PATTERN.findall(log_path.read_text(encoding="utf-8"))
    if not matches:
        raise ValueError(f"Missing best valid loss in training log: {log_path}")
    return float(matches[-1])


def read_regional_median_metrics(exp_dir: Path, period: str) -> dict[str, float]:
    df = pd.read_csv(exp_dir / PERIOD_METRICS_FILES[period])
    mask = (df["stat"].astype(str).str.upper() == "MEDIAN") & (
        df["basin_id"].isna() | (df["basin_id"].astype(str).str.strip() == "")
    )
    row = df.loc[mask]
    if row.empty:
        raise ValueError(f"Missing regional MEDIAN row: {exp_dir.name} / {period}")
    return {name: float(row.iloc[0][name]) for name in METRICS}


def collect_scheme_summary(exp_dir: Path) -> dict:
    meta = parse_run_name(exp_dir.name)
    period_metrics = {
        period: read_regional_median_metrics(exp_dir, period) for period in PERIOD_ORDER
    }
    return {
        "scheme": exp_dir.name,
        "valid_loss": read_best_valid_loss(exp_dir),
        "valid_rmse_median": period_metrics["valid"]["RMSE"],
        "period_metrics": period_metrics,
        **meta,
    }


def sort_schemes_by_valid_loss(summaries: list[dict]) -> list[dict]:
    return sorted(
        summaries,
        key=lambda row: (row["valid_loss"], row["valid_rmse_median"], row["run_name"]),
    )


def write_ranked_table(sorted_schemes: list[dict]) -> None:
    rows = []
    for rank, item in enumerate(sorted_schemes, start=1):
        base = {
            "rank": rank,
            "run_name": item["run_name"],
            "hidden_size": item["hidden_size"],
            "dropout": item["dropout"],
            "warmup_length": item["warmup_length"],
            "batch_size": item["batch_size"],
            "lr": item["lr"],
            "seed": item["seed"],
            "valid_loss": item["valid_loss"],
            "valid_rmse_median": item["valid_rmse_median"],
        }
        for period in PERIOD_ORDER:
            rows.append({"period": period, **base, **item["period_metrics"][period]})
    pd.DataFrame(rows)[RANKED_COLUMNS].to_csv(
        RANKED_CSV, index=False, encoding="utf-8-sig"
    )


def write_top9_table(top9: list[dict]) -> None:
    rows = [
        {
            "rank": rank,
            "run_name": item["run_name"],
            "hidden_size": item["hidden_size"],
            "dropout": item["dropout"],
            "batch_size": item["batch_size"],
            "warmup_length": item["warmup_length"],
            "lr": item["lr"],
            "seed": item["seed"],
            "valid_loss": item["valid_loss"],
            "valid_rmse_median": item["valid_rmse_median"],
        }
        for rank, item in enumerate(top9, start=1)
    ]
    pd.DataFrame(rows)[TOP9_COLUMNS].to_csv(TOP9_CSV, index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# Copy Top9
# ---------------------------------------------------------------------------


def copy_top9_to_sec1(top9: list[dict]) -> list[Path]:
    SEC1_ROOT.mkdir(parents=True, exist_ok=True)
    keep = {item["scheme"] for item in top9}

    for path in SEC1_ROOT.iterdir():
        if (
            path.is_dir()
            and path.name.startswith(SCHEME_PREFIX)
            and path.name not in keep
        ):
            shutil.rmtree(path)

    copied = []
    for item in top9:
        src = SEC0_ROOT / item["scheme"]
        dst = SEC1_ROOT / item["scheme"]
        if not src.is_dir():
            raise FileNotFoundError(f"Missing Sec0 scheme dir: {src}")
        shutil.copytree(src, dst, dirs_exist_ok=True)
        copied.append(dst)
    return copied


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


def format_metric_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for name in METRICS:
        if name not in out.columns:
            continue
        prec = 2 if name == "PTE" else 4
        out[name] = out[name].map(lambda value: f"{float(value):.{prec}f}")
    return out


def write_basin_metrics(
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
                {
                    "period": period,
                    "basin_id": basin_id,
                    "event_id": event_id,
                    **metrics,
                }
            )

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
                basin_rows.append(
                    {"stat": stat, "period": period, "basin_id": basin_id, **metrics}
                )

    format_metric_frame(pd.DataFrame(basin_rows)).to_csv(
        basin_dir / "best_metrics.csv", index=False, encoding="utf-8-sig"
    )
    return basin_rows


def write_ensemble_outputs(top9_dirs: list[Path]) -> pd.DataFrame:
    ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)
    basin_names = [path.name for path in list_basin_dirs(top9_dirs[0])]
    rows_by_period = {period: [] for period in PERIOD_ORDER}
    summary_rows = []

    for basin_name in basin_names:
        basin_dir = ENSEMBLE_DIR / basin_name
        basin_dir.mkdir(parents=True, exist_ok=True)
        timeseries = build_median_timeseries(top9_dirs, basin_name)
        timeseries.to_csv(
            basin_dir / "timeseries.csv", index=False, encoding="utf-8-sig"
        )
        for row in write_basin_metrics(basin_dir, basin_name, timeseries):
            rows_by_period[row["period"]].append(row)

    for period in PERIOD_ORDER:
        basin_rows = rows_by_period[period]
        if not basin_rows:
            continue
        period_rows = [
            {k: v for k, v in row.items() if k != "period"} for row in basin_rows
        ]
        regional_rows = summarize_flood_regional(ENSEMBLE_DIR, basin_names, period)
        save_flood_period_csv(
            ENSEMBLE_DIR / PERIOD_METRICS_FILES[period], period_rows, regional_rows
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
        ENSEMBLE_DIR / "ensemble_summary_metrics.csv", index=False, encoding="utf-8-sig"
    )
    return summary


def write_median_contribution(top9: list[dict], top9_dirs: list[Path]) -> pd.DataFrame:
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
            "run_name": item["run_name"],
            "median_timesteps_total": int(counts[rank - 1]),
            "median_pct_total": round(100.0 * counts[rank - 1] / total_steps, 4),
        }
        for rank, item in enumerate(top9, start=1)
    ]
    df = pd.DataFrame(rows)
    df.to_csv(MEDIAN_CONTRIB_CSV, index=False, encoding="utf-8-sig")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    setup_logging()

    scheme_dirs = list_scheme_dirs(SEC0_ROOT)
    if not scheme_dirs:
        raise FileNotFoundError(f"No Sec0 results under {SEC0_ROOT}")

    print("=" * 60)
    print(f"Sec1 — {MODEL_TAG} rank / Top9 / ensemble")
    print(f"Sec0 root: {SEC0_ROOT}")
    print(f"Sec1 root: {SEC1_ROOT}")
    print(f"Schemes scanned: {len(scheme_dirs)}")
    print("-" * 60)

    sorted_schemes = sort_schemes_by_valid_loss(
        [collect_scheme_summary(path) for path in scheme_dirs]
    )
    top9 = sorted_schemes[:TOP_K]

    SEC1_ROOT.mkdir(parents=True, exist_ok=True)
    write_ranked_table(sorted_schemes)
    write_top9_table(top9)
    top9_dirs = copy_top9_to_sec1(top9)
    ensemble_summary = write_ensemble_outputs(top9_dirs)
    write_median_contribution(top9, top9_dirs)

    print(f"Ranked metrics: {RANKED_CSV}")
    print(f"Top{TOP_K} hyperparams: {TOP9_CSV}")
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
