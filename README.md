# Adaptive Weather Calibration for Debrecen

This project builds a practical research pipeline for hyperlocal probabilistic weather
forecast calibration in Debrecen, Hungary. It uses Debrecen International Airport
(`LHDC`, WMO `12882`) as the primary observation point and compares raw numerical
weather forecasts with calibrated, uncertainty-aware forecasts.

The main pipeline uses real public APIs:

- IEM ASOS/METAR CSV API for historical `LHDC` observations
- Open-Meteo Previous Runs API for fixed-lead historical forecasts
- NOAA Aviation Weather API for current/live METAR checks

The synthetic generator is still available as an offline development fallback.

## Research Question

Can adaptive, regime-aware ML calibration improve local forecasts for Debrecen while
keeping uncertainty intervals reliable during fog, frost, heat, wind, and frontal regimes?

## Quick Start

```powershell
python -m pip install -e .
debrecen-weather run-real --config configs/debrecen.yml --publish-site
```

If the package is not installed, run from the repo root:

```powershell
$env:PYTHONPATH="src"
python -m debrecen_weather.cli run-real --config configs/debrecen.yml --publish-site
```

The command creates:

- `data/raw/observations/debrecen_lhdc_observations.csv`
- `data/raw/forecasts/debrecen_model_forecasts.csv`
- `data/processed/debrecen_features.csv`
- `data/processed/debrecen_predictions.csv`
- `reports/metrics.csv`
- `reports/debrecen_forecast_calibration_report.html`
- static figures in `reports/figures/`

## Main Commands

```powershell
debrecen-weather make-sample --config configs/debrecen.yml --days 180
debrecen-weather fetch-real --config configs/debrecen.yml --start-date 2025-03-01 --end-date 2025-08-31
debrecen-weather run-real --config configs/debrecen.yml --publish-site
debrecen-weather build-dataset --config configs/debrecen.yml
debrecen-weather train --config configs/debrecen.yml --target temperature_2m
debrecen-weather plot --config configs/debrecen.yml
debrecen-weather publish-site --config configs/debrecen.yml
debrecen-weather fetch-live-metar --config configs/debrecen.yml
```

## GitHub Pages

The workflow in `.github/workflows/pages.yml` builds and deploys the static report.
After pushing to GitHub, set **Settings > Pages > Build and deployment > Source** to
**GitHub Actions**, then run the **Build Debrecen Weather Pages** workflow.

The generated static site lives in `site/` locally and is deployed as a Pages artifact
in CI. See `docs/github_pages.md`.

The GitHub Actions build uses chunked Open-Meteo requests with retry/backoff settings
from `configs/debrecen.yml`, because the Previous Runs API can time out on large
single-range requests from hosted runners.

## Project Structure

```text
configs/                 Debrecen config and modeling settings
data/raw/                Raw observations and forecast model outputs
data/interim/            Aligned forecast-observation tables
data/processed/          Feature matrices, predictions, trained model bundles
docs/                    Research notes and data-source notes
reports/                 Metrics, figures, and generated HTML report
src/debrecen_weather/    Ingestion, features, modeling, metrics, plots, CLI
tests/                   Lightweight offline tests
```

## Current Scope

The default target is `temperature_2m`. The package is structured so dew point, wind,
gusts, precipitation, and fog-risk targets can be added by changing the config and adding
target-specific event metrics.

## Data Sources

- Debrecen observations: IEM ASOS/METAR for `LHDC`
- Live METAR: NOAA Aviation Weather API
- Historical station alternatives: NOAA GHCNh, Meteostat, or HungaroMet ODP
- Forecasts: Open-Meteo Previous Runs API for ICON-EU, ECMWF IFS, GFS, and ARPEGE

See `docs/data_sources.md` and `docs/research_plan.md` for the agent-facing plan.
