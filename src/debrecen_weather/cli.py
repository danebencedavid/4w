"""Command line interface for the Debrecen weather calibration project."""

from __future__ import annotations

import argparse
from pathlib import Path

from .align import build_aligned_dataset
from .config import load_config
from .evaluate import evaluate_from_config
from .features import build_feature_dataset
from .ingest_forecasts import fetch_all_open_meteo_latest
from .ingest_forecasts import fetch_all_open_meteo_previous_runs
from .ingest_observations import fetch_live_metar
from .ingest_observations import fetch_iem_asos_observations
from .modeling import train_from_config
from .plots import plot_from_config
from .reporting import write_html_report
from .sample_data import write_sample_data
from .site import build_static_site


def _add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/debrecen.yml", help="Path to project YAML config.")


def make_sample(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    summary = write_sample_data(cfg, days=args.days, seed=args.seed)
    print(f"Wrote {summary.observation_rows:,} observation rows to {summary.observations_path}")
    print(f"Wrote {summary.forecast_rows:,} forecast rows to {summary.forecasts_path}")


def build_dataset(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    aligned = build_aligned_dataset(cfg)
    features = build_feature_dataset(cfg)
    print(f"Aligned rows: {len(aligned):,}")
    print(f"Feature rows: {len(features):,}")


def train(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    _, predictions = train_from_config(cfg, target=args.target)
    metrics = evaluate_from_config(cfg, target=args.target)
    print(f"Prediction rows: {len(predictions):,}")
    print(f"Metric rows: {len(metrics):,}")


def plot(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    figures = plot_from_config(cfg, target=args.target)
    report = write_html_report(cfg, figure_paths=figures)
    print("Figures:")
    for figure in figures:
        print(f"  {figure}")
    print(f"Report: {report}")


def run_all(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    summary = write_sample_data(cfg, days=args.sample_days, seed=args.seed)
    aligned = build_aligned_dataset(cfg)
    features = build_feature_dataset(cfg)
    _, predictions = train_from_config(cfg, target=args.target)
    metrics = evaluate_from_config(cfg, target=args.target)
    figures = plot_from_config(cfg, target=args.target)
    report = write_html_report(cfg, figure_paths=figures)

    print("Pipeline complete")
    print(f"Observations: {summary.observation_rows:,} rows -> {summary.observations_path}")
    print(f"Forecasts: {summary.forecast_rows:,} rows -> {summary.forecasts_path}")
    print(f"Aligned rows: {len(aligned):,}")
    print(f"Feature rows: {len(features):,}")
    print(f"Prediction rows: {len(predictions):,}")
    print(f"Metric rows: {len(metrics):,}")
    print(f"Report: {report}")


def fetch_real(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    observations = fetch_iem_asos_observations(cfg, start_date=args.start_date, end_date=args.end_date)
    forecasts = fetch_all_open_meteo_previous_runs(cfg, start_date=args.start_date, end_date=args.end_date)
    print(f"Fetched real observations: {len(observations):,} rows")
    print(f"Fetched real forecasts: {len(forecasts):,} rows")


def run_real(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    observations = fetch_iem_asos_observations(cfg, start_date=args.start_date, end_date=args.end_date)
    forecasts = fetch_all_open_meteo_previous_runs(cfg, start_date=args.start_date, end_date=args.end_date)
    aligned = build_aligned_dataset(cfg)
    features = build_feature_dataset(cfg)
    _, predictions = train_from_config(cfg, target=args.target)
    metrics = evaluate_from_config(cfg, target=args.target)
    figures = plot_from_config(cfg, target=args.target)
    report = write_html_report(cfg, figure_paths=figures)
    site_dir = build_static_site(cfg) if args.publish_site else None

    print("Real-data pipeline complete")
    print(f"Observations: {len(observations):,} rows")
    print(f"Forecasts: {len(forecasts):,} rows")
    print(f"Aligned rows: {len(aligned):,}")
    print(f"Feature rows: {len(features):,}")
    print(f"Prediction rows: {len(predictions):,}")
    print(f"Metric rows: {len(metrics):,}")
    print(f"Report: {report}")
    if site_dir:
        print(f"Site: {site_dir}")


def fetch_live(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    df = fetch_live_metar(cfg)
    print(df.tail().to_string(index=False))


def fetch_latest_forecasts(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    df = fetch_all_open_meteo_latest(cfg)
    print(f"Fetched {len(df):,} forecast rows")


def publish_site(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    site_dir = build_static_site(cfg)
    print(f"Site written to {site_dir}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Debrecen probabilistic weather calibration.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample_parser = subparsers.add_parser("make-sample", help="Generate offline synthetic data.")
    _add_config_arg(sample_parser)
    sample_parser.add_argument("--days", type=int, default=None)
    sample_parser.add_argument("--seed", type=int, default=None)
    sample_parser.set_defaults(func=make_sample)

    build_parser = subparsers.add_parser("build-dataset", help="Align raw data and build features.")
    _add_config_arg(build_parser)
    build_parser.set_defaults(func=build_dataset)

    train_parser = subparsers.add_parser("train", help="Train calibrator and evaluate predictions.")
    _add_config_arg(train_parser)
    train_parser.add_argument("--target", default=None)
    train_parser.set_defaults(func=train)

    plot_parser = subparsers.add_parser("plot", help="Create figures and HTML report.")
    _add_config_arg(plot_parser)
    plot_parser.add_argument("--target", default=None)
    plot_parser.set_defaults(func=plot)

    run_parser = subparsers.add_parser("run-all", help="Run the whole offline pipeline.")
    _add_config_arg(run_parser)
    run_parser.add_argument("--sample-days", type=int, default=None)
    run_parser.add_argument("--seed", type=int, default=None)
    run_parser.add_argument("--target", default=None)
    run_parser.set_defaults(func=run_all)

    fetch_real_parser = subparsers.add_parser("fetch-real", help="Fetch real API observations and previous-run forecasts.")
    _add_config_arg(fetch_real_parser)
    fetch_real_parser.add_argument("--start-date", default=None, help="Inclusive YYYY-MM-DD start date.")
    fetch_real_parser.add_argument("--end-date", default=None, help="Inclusive YYYY-MM-DD end date.")
    fetch_real_parser.set_defaults(func=fetch_real)

    real_parser = subparsers.add_parser("run-real", help="Run the full real API-backed pipeline.")
    _add_config_arg(real_parser)
    real_parser.add_argument("--start-date", default=None, help="Inclusive YYYY-MM-DD start date.")
    real_parser.add_argument("--end-date", default=None, help="Inclusive YYYY-MM-DD end date.")
    real_parser.add_argument("--target", default=None)
    real_parser.add_argument("--publish-site", action="store_true", help="Also write site/index.html for GitHub Pages.")
    real_parser.set_defaults(func=run_real)

    metar_parser = subparsers.add_parser("fetch-live-metar", help="Fetch latest LHDC METAR.")
    _add_config_arg(metar_parser)
    metar_parser.set_defaults(func=fetch_live)

    forecast_parser = subparsers.add_parser("fetch-latest-forecasts", help="Fetch latest Open-Meteo forecasts.")
    _add_config_arg(forecast_parser)
    forecast_parser.set_defaults(func=fetch_latest_forecasts)

    site_parser = subparsers.add_parser("publish-site", help="Copy report assets into the GitHub Pages site directory.")
    _add_config_arg(site_parser)
    site_parser.set_defaults(func=publish_site)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
