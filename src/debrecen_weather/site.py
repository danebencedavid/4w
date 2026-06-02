"""GitHub Pages site builder."""

from __future__ import annotations

import shutil
from pathlib import Path

from .config import ensure_dir, resolve_path


def build_static_site(cfg: dict) -> Path:
    """Copy the generated report and figures into a GitHub Pages-ready directory."""
    report_path = resolve_path(cfg, "report_html")
    figures_dir = resolve_path(cfg, "figures_dir")
    site_dir = resolve_path(cfg, "site_dir")
    site_figures = site_dir / "figures"

    if not report_path.exists():
        raise FileNotFoundError(f"Report does not exist: {report_path}")

    if site_dir.exists():
        for child in site_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    ensure_dir(site_figures)

    shutil.copy2(report_path, site_dir / "index.html")
    for figure in figures_dir.glob("*.png"):
        shutil.copy2(figure, site_figures / figure.name)

    metrics_path = resolve_path(cfg, "metrics")
    if metrics_path.exists():
        ensure_dir(site_dir / "data")
        shutil.copy2(metrics_path, site_dir / "data" / "metrics.csv")

    (site_dir / ".nojekyll").write_text("", encoding="utf-8")
    return site_dir
