"""Forecast evaluation metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import ensure_parent, resolve_path
from .modeling import quantile_name


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def bias(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.mean(y_pred - y_true))


def pinball_loss(y_true: pd.Series, y_pred: pd.Series, quantile: float) -> float:
    error = y_true - y_pred
    return float(np.mean(np.maximum(quantile * error, (quantile - 1) * error)))


def interval_coverage(y_true: pd.Series, lower: pd.Series, upper: pd.Series) -> float:
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def interval_width(lower: pd.Series, upper: pd.Series) -> float:
    return float(np.mean(upper - lower))


def brier_score(y_true_event: pd.Series, probability: pd.Series) -> float:
    return float(np.mean((probability - y_true_event.astype(float)) ** 2))


def probability_exceedance_from_quantiles(
    df: pd.DataFrame,
    threshold: float,
    quantiles: list[float],
) -> pd.Series:
    """Approximate P(y >= threshold) from predicted quantiles by interpolation."""
    q_cols = [quantile_name(q) for q in quantiles if quantile_name(q) in df]
    q_values = [q for q in quantiles if quantile_name(q) in df]
    if not q_cols:
        return pd.Series(np.nan, index=df.index)

    values = df[q_cols].to_numpy(dtype=float)
    probs = []
    for row in values:
        order = np.argsort(row)
        sorted_values = row[order]
        sorted_probs = np.asarray(q_values)[order]
        cdf = np.interp(threshold, sorted_values, sorted_probs, left=0.0, right=1.0)
        probs.append(1.0 - cdf)
    return pd.Series(probs, index=df.index)


def _metric_rows_for_group(
    frame: pd.DataFrame,
    group_label: dict[str, object],
    target: str,
    interval_low: float,
    interval_high: float,
    quantiles: list[float],
) -> list[dict[str, object]]:
    y_col = f"observed_{target}"
    raw_col = f"forecast_{target}"
    median_col = quantile_name(0.5)
    low_col = quantile_name(interval_low)
    high_col = quantile_name(interval_high)
    low_conf = f"{low_col}_conformal"
    high_conf = f"{high_col}_conformal"

    rows: list[dict[str, object]] = []
    base = {"target": target, "n": len(frame), **group_label}
    if raw_col in frame:
        rows.append(
            {
                **base,
                "forecast": "raw",
                "mae": mae(frame[y_col], frame[raw_col]),
                "rmse": rmse(frame[y_col], frame[raw_col]),
                "bias": bias(frame[y_col], frame[raw_col]),
            }
        )
    if median_col in frame:
        calibrated = {
            **base,
            "forecast": "calibrated_median",
            "mae": mae(frame[y_col], frame[median_col]),
            "rmse": rmse(frame[y_col], frame[median_col]),
            "bias": bias(frame[y_col], frame[median_col]),
        }
        if raw_col in frame:
            raw_mae = mae(frame[y_col], frame[raw_col])
            calibrated["mae_skill_vs_raw"] = 1 - calibrated["mae"] / raw_mae if raw_mae else np.nan
        rows.append(calibrated)
    if low_col in frame and high_col in frame:
        rows.append(
            {
                **base,
                "forecast": f"q{int(interval_low * 100)}_q{int(interval_high * 100)}",
                "coverage": interval_coverage(frame[y_col], frame[low_col], frame[high_col]),
                "interval_width": interval_width(frame[low_col], frame[high_col]),
            }
        )
    if low_conf in frame and high_conf in frame:
        rows.append(
            {
                **base,
                "forecast": "conformal_interval",
                "coverage": interval_coverage(frame[y_col], frame[low_conf], frame[high_conf]),
                "interval_width": interval_width(frame[low_conf], frame[high_conf]),
                "mean_conformal_adjustment": float(frame.get("conformal_adjustment", pd.Series(np.nan)).mean()),
            }
        )
    for quantile in quantiles:
        col = quantile_name(quantile)
        if col in frame:
            rows.append(
                {
                    **base,
                    "forecast": col,
                    "pinball_loss": pinball_loss(frame[y_col], frame[col], quantile),
                }
            )
    return rows


def evaluate_predictions(
    predictions: pd.DataFrame,
    target: str,
    interval_low: float,
    interval_high: float,
    quantiles: list[float],
) -> pd.DataFrame:
    """Evaluate predictions overall and by operational groups."""
    work = predictions.dropna(subset=[f"observed_{target}"]).copy()
    rows = _metric_rows_for_group(
        work,
        {"split": "all", "model": "all", "lead_bucket": "all", "regime": "all"},
        target,
        interval_low,
        interval_high,
        quantiles,
    )
    for split, group in work.groupby("split"):
        rows.extend(
            _metric_rows_for_group(
                group,
                {"split": split, "model": "all", "lead_bucket": "all", "regime": "all"},
                target,
                interval_low,
                interval_high,
                quantiles,
            )
        )
    for columns in [("split", "model"), ("split", "lead_bucket"), ("split", "regime")]:
        for key, group in work.groupby(list(columns), dropna=False):
            if not isinstance(key, tuple):
                key = (key,)
            label = {"split": "all", "model": "all", "lead_bucket": "all", "regime": "all"}
            label.update(dict(zip(columns, key)))
            rows.extend(
                _metric_rows_for_group(
                    group,
                    label,
                    target,
                    interval_low,
                    interval_high,
                    quantiles,
                )
            )

    return pd.DataFrame(rows)


def evaluate_events(
    predictions: pd.DataFrame,
    target: str,
    quantiles: list[float],
    event_thresholds: dict[str, float],
) -> pd.DataFrame:
    """Evaluate threshold exceedance events for the configured target."""
    y_col = f"observed_{target}"
    rows = []
    for event_name, threshold in event_thresholds.items():
        probability = probability_exceedance_from_quantiles(predictions, threshold, quantiles)
        observed_event = predictions[y_col] >= threshold
        frame = pd.DataFrame(
            {
                "split": predictions["split"],
                "event": event_name,
                "observed_event": observed_event,
                "probability": probability,
            }
        ).dropna()
        for split, group in frame.groupby("split"):
            rows.append(
                {
                    "split": split,
                    "event": event_name,
                    "threshold": threshold,
                    "n": len(group),
                    "base_rate": float(group["observed_event"].mean()),
                    "brier_score": brier_score(group["observed_event"], group["probability"]),
                    "mean_probability": float(group["probability"].mean()),
                }
            )
    return pd.DataFrame(rows)


def evaluate_from_config(cfg: dict, target: str | None = None) -> pd.DataFrame:
    target = target or cfg["training"]["target"]
    predictions = pd.read_csv(resolve_path(cfg, "predictions"))
    metrics = evaluate_predictions(
        predictions,
        target=target,
        interval_low=float(cfg["training"]["interval_low"]),
        interval_high=float(cfg["training"]["interval_high"]),
        quantiles=[float(q) for q in cfg["training"]["quantiles"]],
    )

    if target == "temperature_2m":
        event_thresholds = {
            "frost": float(cfg["events"]["frost_c"]),
            "warm_day": float(cfg["events"]["warm_day_c"]),
            "hot_day": float(cfg["events"]["hot_day_c"]),
        }
        event_metrics = evaluate_events(
            predictions,
            target=target,
            quantiles=[float(q) for q in cfg["training"]["quantiles"]],
            event_thresholds=event_thresholds,
        )
        if not event_metrics.empty:
            event_metrics["forecast"] = "event_probability"
            metrics = pd.concat([metrics, event_metrics], ignore_index=True, sort=False)

    output = resolve_path(cfg, "metrics")
    ensure_parent(output)
    metrics.to_csv(output, index=False)
    return metrics
