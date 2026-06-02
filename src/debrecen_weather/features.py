"""Feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ensure_parent, resolve_path
from .regimes import add_lead_buckets, add_regime_labels


def _cyclical(series: pd.Series, period: float) -> tuple[pd.Series, pd.Series]:
    angle = 2 * np.pi * series.astype(float) / period
    return np.sin(angle), np.cos(angle)


def _wind_components(direction_degrees: pd.Series) -> tuple[pd.Series, pd.Series]:
    radians = np.deg2rad(direction_degrees.astype(float))
    return np.sin(radians), np.cos(radians)


def add_time_features(df: pd.DataFrame, timezone: str) -> pd.DataFrame:
    result = df.copy()
    result["valid_time_utc"] = pd.to_datetime(result["valid_time_utc"], utc=True)
    result["run_time_utc"] = pd.to_datetime(result["run_time_utc"], utc=True)
    result["valid_time_local"] = result["valid_time_utc"].dt.tz_convert(timezone)
    result["hour"] = result["valid_time_local"].dt.hour
    result["month"] = result["valid_time_local"].dt.month
    result["dayofyear"] = result["valid_time_local"].dt.dayofyear
    result["hour_sin"], result["hour_cos"] = _cyclical(result["hour"], 24)
    result["doy_sin"], result["doy_cos"] = _cyclical(result["dayofyear"], 365.25)
    result["season"] = pd.cut(
        result["month"],
        bins=[0, 2, 5, 8, 11, 12],
        labels=["winter", "spring", "summer", "autumn", "winter2"],
        include_lowest=True,
    ).astype(str).replace({"winter2": "winter"})
    return result


def add_observation_lags(df: pd.DataFrame, variables: list[str] | None = None) -> pd.DataFrame:
    """Add recent observed values at the target valid time minus each lag."""
    result = df.copy()
    variables = variables or ["temperature_2m", "dew_point_2m", "pressure_msl", "wind_speed_10m"]
    unique_obs = (
        result[["valid_time_utc"] + [f"observed_{variable}" for variable in variables if f"observed_{variable}" in result]]
        .drop_duplicates("valid_time_utc")
        .sort_values("valid_time_utc")
    )
    for lag in [1, 3, 6, 12, 24]:
        lagged = unique_obs.copy()
        lagged["valid_time_utc"] = lagged["valid_time_utc"] + pd.to_timedelta(lag, unit="h")
        rename = {
            f"observed_{variable}": f"obs_lag_{lag}h_{variable}"
            for variable in variables
            if f"observed_{variable}" in unique_obs
        }
        lagged = lagged.rename(columns=rename)
        result = result.merge(lagged[["valid_time_utc"] + list(rename.values())], on="valid_time_utc", how="left")

    if "obs_lag_1h_pressure_msl" in result and "obs_lag_3h_pressure_msl" in result:
        result["pressure_tendency_3h"] = result["obs_lag_1h_pressure_msl"] - result["obs_lag_3h_pressure_msl"]
    if "obs_lag_1h_temperature_2m" in result and "obs_lag_3h_temperature_2m" in result:
        result["temperature_tendency_3h"] = result["obs_lag_1h_temperature_2m"] - result["obs_lag_3h_temperature_2m"]
    return result


def add_physical_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "forecast_temperature_2m" in result and "forecast_dew_point_2m" in result:
        result["forecast_temp_dewpoint_spread"] = result["forecast_temperature_2m"] - result["forecast_dew_point_2m"]
    if "observed_temperature_2m" in result and "observed_dew_point_2m" in result:
        result["observed_temp_dewpoint_spread"] = result["observed_temperature_2m"] - result["observed_dew_point_2m"]
    if "forecast_wind_direction_10m" in result:
        result["forecast_wind_dir_sin"], result["forecast_wind_dir_cos"] = _wind_components(
            result["forecast_wind_direction_10m"]
        )
    return result


def build_features(df: pd.DataFrame, timezone: str) -> pd.DataFrame:
    """Create the full feature table from aligned data."""
    result = add_time_features(df, timezone=timezone)
    result = add_observation_lags(result)
    result = add_physical_features(result)
    result = add_lead_buckets(result)
    result = add_regime_labels(result)
    return result


def build_feature_dataset(cfg: dict) -> pd.DataFrame:
    """Load aligned data, create features, and save the processed table."""
    aligned_path = resolve_path(cfg, "aligned")
    aligned = pd.read_csv(aligned_path)
    features = build_features(aligned, timezone=cfg["project"]["timezone"])
    output = resolve_path(cfg, "features")
    ensure_parent(output)
    features.to_csv(output, index=False)
    return features
