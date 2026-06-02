"""Forecast-observation alignment."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import ensure_parent, resolve_path
from .constants import FORECAST_PREFIX, OBSERVED_PREFIX, WEATHER_VARIABLES
from .ingest_forecasts import read_forecasts
from .ingest_observations import read_observations


def align_forecasts_observations(forecasts: pd.DataFrame, observations: pd.DataFrame) -> pd.DataFrame:
    """Join forecasts to observations by valid time.

    The result keeps one row per issued forecast/model/lead and prefixes raw model
    variables with `forecast_` and station observations with `observed_`.
    """
    fc = forecasts.copy()
    obs = observations.copy()
    fc["valid_time_utc"] = pd.to_datetime(fc["valid_time_utc"], utc=True)
    fc["run_time_utc"] = pd.to_datetime(fc["run_time_utc"], utc=True)
    obs["valid_time_utc"] = pd.to_datetime(obs["valid_time_utc"], utc=True)

    fc = fc.rename(columns={variable: f"{FORECAST_PREFIX}{variable}" for variable in WEATHER_VARIABLES})
    obs = obs.rename(
        columns={
            column: f"{OBSERVED_PREFIX}{column}"
            for column in obs.columns
            if column != "valid_time_utc"
        }
    )
    merged = fc.merge(obs, on="valid_time_utc", how="inner", validate="many_to_one")
    return merged.sort_values(["valid_time_utc", "lead_hours", "model"]).reset_index(drop=True)


def add_multimodel_spread(df: pd.DataFrame, variables: list[str] | None = None) -> pd.DataFrame:
    """Add valid-time/lead multi-model spread features for forecast variables."""
    result = df.copy()
    variables = variables or ["temperature_2m", "dew_point_2m", "wind_speed_10m", "pressure_msl"]
    group_cols = ["valid_time_utc", "lead_hours"]
    for variable in variables:
        col = f"forecast_{variable}"
        if col not in result:
            continue
        stats = result.groupby(group_cols)[col].agg(["mean", "std", "min", "max"]).reset_index()
        stats = stats.rename(
            columns={
                "mean": f"model_mean_{variable}",
                "std": f"model_spread_{variable}",
                "min": f"model_min_{variable}",
                "max": f"model_max_{variable}",
            }
        )
        stats[f"model_range_{variable}"] = stats[f"model_max_{variable}"] - stats[f"model_min_{variable}"]
        result = result.merge(stats, on=group_cols, how="left")
    return result


def build_aligned_dataset(cfg: dict) -> pd.DataFrame:
    """Load raw files, align them, and save the interim dataset."""
    forecasts = read_forecasts(resolve_path(cfg, "forecasts_raw"))
    observations = read_observations(resolve_path(cfg, "observations_raw"))
    aligned = add_multimodel_spread(align_forecasts_observations(forecasts, observations))
    output = resolve_path(cfg, "aligned")
    ensure_parent(output)
    aligned.to_csv(output, index=False)
    return aligned
