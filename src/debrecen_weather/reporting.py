"""Static HTML report generation."""

from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd

from .config import ensure_parent, resolve_path


def _relative_image(path: Path, report_path: Path) -> str:
    return path.resolve().relative_to(report_path.parent.resolve()).as_posix()


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

    metric_preview = metrics.head(30).to_html(index=False, classes="metrics", border=0) if not metrics.empty else ""
    image_html = "\n".join(
        f'<section><h2>{escape(path.stem.replace("_", " ").title())}</h2>'
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
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 18px; }}
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
      {metric_preview}
    </section>
    <div class="grid">
      {image_html}
    </div>
  </main>
</body>
</html>
"""
    ensure_parent(report_path)
    report_path.write_text(html, encoding="utf-8")
    return report_path
