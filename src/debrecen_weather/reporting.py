"""Static HTML report generation."""

from __future__ import annotations

from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ensure_parent, resolve_path


def _relative_image(path: Path, report_path: Path) -> str:
    return path.resolve().relative_to(report_path.parent.resolve()).as_posix()


def _clean_metric_table(df: pd.DataFrame, columns: list[str]) -> str:
    """Render a compact metric table without noisy NaN cells."""
    if df.empty:
        return "<p>No rows available.</p>"
    keep = [column for column in columns if column in df.columns]
    display = df[keep].copy()
    for column in display.select_dtypes(include=[np.number]).columns:
        display[column] = display[column].round(3)
    display = display.replace({np.nan: "", "NaN": ""})
    return display.to_html(index=False, classes="metrics", border=0)


def _metric_section_html(metrics: pd.DataFrame) -> str:
    """Create user-facing metric tables from the long metrics CSV."""
    if metrics.empty:
        return "<p>No metrics were generated.</p>"

    headline = metrics[
        (metrics.get("model", "") == "all")
        & (metrics.get("lead_bucket", "") == "all")
        & (metrics.get("regime", "") == "all")
        & metrics.get("split", "").isin(["validation", "test"])
        & metrics.get("forecast", "").isin(["raw", "calibrated_median", "q10_q90", "conformal_interval"])
    ].copy()
    headline_html = _clean_metric_table(
        headline,
        [
            "split",
            "forecast",
            "n",
            "mae",
            "rmse",
            "bias",
            "mae_skill_vs_raw",
            "coverage",
            "interval_width",
            "mean_conformal_adjustment",
        ],
    )

    model_rows = metrics[
        (metrics.get("split", "") == "test")
        & (metrics.get("model", "") != "all")
        & (metrics.get("lead_bucket", "") == "all")
        & (metrics.get("regime", "") == "all")
        & metrics.get("forecast", "").isin(["raw", "calibrated_median"])
    ].copy()
    model_html = _clean_metric_table(
        model_rows,
        ["model", "forecast", "n", "mae", "rmse", "bias", "mae_skill_vs_raw"],
    )

    event_rows = metrics[metrics.get("forecast", "") == "event_probability"].copy()
    event_html = _clean_metric_table(
        event_rows,
        ["split", "event", "threshold", "n", "base_rate", "brier_score", "mean_probability"],
    )

    return f"""
      <h2>Headline Metrics</h2>
      {headline_html}
      <h2>Test Metrics By Model</h2>
      {model_html}
      <h2>Event Probabilities</h2>
      {event_html}
    """


def _metric_cards(metrics: pd.DataFrame) -> str:
    if metrics.empty:
        return ""
    selector = (
        (metrics.get("split", "") == "test")
        & (metrics.get("model", "") == "all")
        & (metrics.get("lead_bucket", "") == "all")
        & (metrics.get("regime", "") == "all")
    )
    test = metrics[selector]

    def value_for(forecast: str, column: str) -> float | None:
        row = test[test.get("forecast", "") == forecast]
        if row.empty or column not in row:
            return None
        value = row.iloc[0][column]
        return None if pd.isna(value) else float(value)

    cards = {
        "Raw Test MAE": value_for("raw", "mae"),
        "Calibrated Test MAE": value_for("calibrated_median", "mae"),
        "MAE Skill vs Raw": value_for("calibrated_median", "mae_skill_vs_raw"),
        "Conformal Coverage": value_for("conformal_interval", "coverage"),
        "Conformal Width": value_for("conformal_interval", "interval_width"),
    }
    card_html = []
    for label, value in cards.items():
        shown = "" if value is None else f"{value:.3f}"
        if label == "MAE Skill vs Raw" and value is not None:
            shown = f"{100 * value:.1f}%"
        if label == "Conformal Coverage" and value is not None:
            shown = f"{100 * value:.1f}%"
        card_html.append(f'<div class="card"><span>{escape(label)}</span><strong>{escape(shown)}</strong></div>')
    return f'<div class="cards">{"".join(card_html)}</div>'


def write_html_report(cfg: dict, figure_paths: list[Path] | None = None) -> Path:
    """Write a lightweight static dashboard/report."""
    report_path = resolve_path(cfg, "report_html")
    metrics_path = resolve_path(cfg, "metrics")
    predictions_path = resolve_path(cfg, "predictions")
    figures_dir = resolve_path(cfg, "figures_dir")
    if figure_paths is None:
        figure_paths = sorted(figures_dir.glob("*.png"))

    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    predictions = pd.read_csv(predictions_path) if predictions_path.exists() else pd.DataFrame()
    station = cfg["station"]

    summary = {}
    if not predictions.empty:
        summary = {
            "Prediction rows": f"{len(predictions):,}",
            "Models": ", ".join(sorted(predictions["model"].dropna().unique())),
            "Lead hours": ", ".join(str(x) for x in sorted(predictions["lead_hours"].dropna().unique())),
            "Test start": str(pd.to_datetime(predictions.loc[predictions["split"] == "test", "valid_time_utc"]).min()),
            "Test end": str(pd.to_datetime(predictions.loc[predictions["split"] == "test", "valid_time_utc"]).max()),
        }

    metric_cards = _metric_cards(metrics)
    metric_sections = _metric_section_html(metrics)
    image_html = "\n".join(
        f'<section class="figure-section"><h2>{escape(path.stem.replace("_", " ").title())}</h2>'
        f'<img src="{escape(_relative_image(path, report_path))}" alt="{escape(path.stem)}"></section>'
        for path in figure_paths
    )
    summary_rows = "\n".join(f"<tr><th>{escape(k)}</th><td>{escape(v)}</td></tr>" for k, v in summary.items())

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Debrecen Weather Calibration Report</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 0; color: #172026; background: #f7f8fa; }}
    header {{ padding: 28px 36px; background: #143642; color: white; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px 24px 48px; }}
    h1 {{ margin: 0 0 8px; font-size: 30px; }}
    h2 {{ margin: 0 0 14px; font-size: 20px; }}
    section {{ background: white; border: 1px solid #d9e1e7; border-radius: 6px; padding: 18px; margin: 0 0 18px; }}
    img {{ max-width: 100%; height: auto; display: block; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #e5ebef; padding: 8px 9px; text-align: left; }}
    th {{ color: #344955; }}
    .figures {{ display: block; }}
    .figure-section {{ width: 100%; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin-bottom: 18px; }}
    .card {{ border: 1px solid #dce5ea; border-radius: 6px; padding: 12px; background: #fbfcfd; }}
    .card span {{ display: block; color: #5d6b74; font-size: 12px; margin-bottom: 5px; }}
    .card strong {{ display: block; color: #143642; font-size: 22px; }}
    .muted {{ color: #d6e4ea; }}
  </style>
</head>
<body>
  <header>
    <h1>Debrecen Weather Calibration Report</h1>
    <div class="muted">Station {escape(station["icao"])} / WMO {escape(str(station["wmo"]))}, {station["latitude"]}, {station["longitude"]}, {station["elevation_m"]} m</div>
  </header>
  <main>
    <section>
      <h2>Run Summary</h2>
      <table>{summary_rows}</table>
    </section>
    <section>
      <h2>Metric Preview</h2>
      {metric_cards}
      {metric_sections}
    </section>
    <div class="figures">
      {image_html}
    </div>
  </main>
</body>
</html>
"""
    ensure_parent(report_path)
    report_path.write_text(html, encoding="utf-8")
    return report_path
