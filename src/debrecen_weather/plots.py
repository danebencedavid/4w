"""Visualization utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import ensure_dir, resolve_path
from .modeling import quantile_name


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_raw_bias_heatmap(features: pd.DataFrame, target: str, figures_dir: Path) -> Path:
    work = features.copy()
    y_col = f"observed_{target}"
    raw_col = f"forecast_{target}"
    work["raw_error"] = work[raw_col] - work[y_col]
    table = work.pivot_table(index="hour", columns="lead_hours", values="raw_error", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(table.to_numpy(), aspect="auto", origin="lower", cmap="coolwarm")
    ax.set_title("Raw Forecast Bias by Local Hour and Lead")
    ax.set_xlabel("Lead hours")
    ax.set_ylabel("Local hour")
    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels(table.columns)
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels(table.index)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Forecast - observed")
    return _save(fig, figures_dir / "raw_bias_heatmap.png")


def plot_mae_by_lead(predictions: pd.DataFrame, target: str, figures_dir: Path) -> Path:
    y_col = f"observed_{target}"
    raw_col = f"forecast_{target}"
    median_col = quantile_name(0.5)
    work = predictions[predictions["split"] == "test"].copy()
    rows = []
    for lead, group in work.groupby("lead_hours"):
        rows.append(
            {
                "lead_hours": lead,
                "raw": np.mean(np.abs(group[raw_col] - group[y_col])),
                "calibrated": np.mean(np.abs(group[median_col] - group[y_col])),
            }
        )
    table = pd.DataFrame(rows).sort_values("lead_hours")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(table["lead_hours"], table["raw"], marker="o", label="Raw")
    ax.plot(table["lead_hours"], table["calibrated"], marker="o", label="Calibrated median")
    ax.set_title("Test MAE by Lead Time")
    ax.set_xlabel("Lead hours")
    ax.set_ylabel("MAE")
    ax.grid(alpha=0.25)
    ax.legend()
    return _save(fig, figures_dir / "mae_by_lead.png")


def plot_interval_case_study(
    predictions: pd.DataFrame,
    target: str,
    interval_low: float,
    interval_high: float,
    figures_dir: Path,
) -> Path:
    low_col = f"{quantile_name(interval_low)}_conformal"
    high_col = f"{quantile_name(interval_high)}_conformal"
    median_col = quantile_name(0.5)
    y_col = f"observed_{target}"
    raw_col = f"forecast_{target}"

    work = predictions[predictions["split"] == "test"].copy()
    preferred_model = "ecmwf_ifs025" if "ecmwf_ifs025" in set(work["model"]) else work["model"].iloc[0]
    preferred_lead = 24 if 24 in set(work["lead_hours"]) else sorted(work["lead_hours"].unique())[0]
    case = work[(work["model"] == preferred_model) & (work["lead_hours"] == preferred_lead)].copy()
    case["valid_time_utc"] = pd.to_datetime(case["valid_time_utc"], utc=True)
    case = case.sort_values("valid_time_utc").tail(24 * 14)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.fill_between(case["valid_time_utc"], case[low_col], case[high_col], alpha=0.20, label="Conformal interval")
    ax.plot(case["valid_time_utc"], case[y_col], color="black", linewidth=1.6, label="Observed")
    ax.plot(case["valid_time_utc"], case[raw_col], color="#9a3412", linewidth=1.1, label="Raw forecast")
    ax.plot(case["valid_time_utc"], case[median_col], color="#2563eb", linewidth=1.3, label="Calibrated median")
    ax.set_title(f"Case Study: {preferred_model}, {preferred_lead}h Lead")
    ax.set_xlabel("Valid time UTC")
    ax.set_ylabel(target)
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    return _save(fig, figures_dir / "interval_case_study.png")


def plot_coverage_by_lead(
    predictions: pd.DataFrame,
    target: str,
    interval_low: float,
    interval_high: float,
    figures_dir: Path,
) -> Path:
    y_col = f"observed_{target}"
    low_col = f"{quantile_name(interval_low)}_conformal"
    high_col = f"{quantile_name(interval_high)}_conformal"
    nominal = interval_high - interval_low
    work = predictions[predictions["split"] == "test"].copy()
    rows = []
    for lead_bucket, group in work.groupby("lead_bucket"):
        rows.append(
            {
                "lead_bucket": lead_bucket,
                "coverage": float(((group[y_col] >= group[low_col]) & (group[y_col] <= group[high_col])).mean()),
                "width": float((group[high_col] - group[low_col]).mean()),
            }
        )
    table = pd.DataFrame(rows).sort_values("lead_bucket")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(table["lead_bucket"], table["coverage"], color="#2563eb", alpha=0.8)
    ax.axhline(nominal, color="black", linestyle="--", linewidth=1.0, label=f"Nominal {nominal:.0%}")
    ax.set_ylim(0, 1)
    ax.set_title("Empirical Test Coverage by Lead Bucket")
    ax.set_xlabel("Lead bucket")
    ax.set_ylabel("Coverage")
    ax.legend()
    return _save(fig, figures_dir / "coverage_by_lead_bucket.png")


def plot_regime_skill(predictions: pd.DataFrame, target: str, figures_dir: Path) -> Path:
    y_col = f"observed_{target}"
    raw_col = f"forecast_{target}"
    median_col = quantile_name(0.5)
    work = predictions[predictions["split"] == "test"].copy()
    rows = []
    for regime, group in work.groupby("regime"):
        raw_mae = np.mean(np.abs(group[raw_col] - group[y_col]))
        calibrated_mae = np.mean(np.abs(group[median_col] - group[y_col]))
        rows.append(
            {
                "regime": regime,
                "skill": 1 - calibrated_mae / raw_mae if raw_mae else np.nan,
                "n": len(group),
            }
        )
    table = pd.DataFrame(rows).sort_values("skill", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(table["regime"], 100 * table["skill"], color="#0f766e", alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Calibrated MAE Skill vs Raw Forecast by Regime")
    ax.set_xlabel("Regime")
    ax.set_ylabel("Skill (%)")
    ax.tick_params(axis="x", rotation=30)
    return _save(fig, figures_dir / "regime_skill.png")


def plot_reliability_temperature_event(
    predictions: pd.DataFrame,
    target: str,
    threshold: float,
    quantiles: list[float],
    figures_dir: Path,
) -> Path:
    from .evaluate import probability_exceedance_from_quantiles

    y_col = f"observed_{target}"
    work = predictions[predictions["split"] == "test"].copy()
    work["probability"] = probability_exceedance_from_quantiles(work, threshold, quantiles)
    work["observed_event"] = work[y_col] >= threshold
    work["bin"] = pd.cut(work["probability"], bins=np.linspace(0, 1, 11), include_lowest=True)
    table = work.groupby("bin", observed=False).agg(
        mean_probability=("probability", "mean"),
        observed_frequency=("observed_event", "mean"),
        n=("observed_event", "size"),
    )
    table = table.dropna(subset=["mean_probability"])

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
    ax.plot(table["mean_probability"], table["observed_frequency"], marker="o", color="#7c3aed")
    ax.set_title(f"Reliability: P({target} >= {threshold:g})")
    ax.set_xlabel("Forecast probability")
    ax.set_ylabel("Observed frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    return _save(fig, figures_dir / "reliability_warm_day.png")


def plot_from_config(cfg: dict, target: str | None = None) -> list[Path]:
    target = target or cfg["training"]["target"]
    features = pd.read_csv(resolve_path(cfg, "features"))
    predictions = pd.read_csv(resolve_path(cfg, "predictions"))
    figures_dir = resolve_path(cfg, "figures_dir")
    ensure_dir(figures_dir)
    interval_low = float(cfg["training"]["interval_low"])
    interval_high = float(cfg["training"]["interval_high"])
    quantiles = [float(q) for q in cfg["training"]["quantiles"]]

    paths = [
        plot_raw_bias_heatmap(features, target, figures_dir),
        plot_mae_by_lead(predictions, target, figures_dir),
        plot_interval_case_study(predictions, target, interval_low, interval_high, figures_dir),
        plot_coverage_by_lead(predictions, target, interval_low, interval_high, figures_dir),
        plot_regime_skill(predictions, target, figures_dir),
    ]
    if target == "temperature_2m":
        paths.append(
            plot_reliability_temperature_event(
                predictions,
                target,
                threshold=float(cfg["events"]["warm_day_c"]),
                quantiles=quantiles,
                figures_dir=figures_dir,
            )
        )
    return paths
