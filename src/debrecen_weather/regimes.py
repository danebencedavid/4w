"""Weather regime labels for local forecast calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_regime_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple Debrecen-oriented weather regime labels."""
    result = df.copy()
    hour = result["valid_time_local"].dt.hour if "valid_time_local" in result else result["valid_time_utc"].dt.hour
    night = (hour <= 7) | (hour >= 20)

    temp = result.get("forecast_temperature_2m", pd.Series(np.nan, index=result.index))
    dew = result.get("forecast_dew_point_2m", pd.Series(np.nan, index=result.index))
    rh = result.get("forecast_relative_humidity_2m", pd.Series(np.nan, index=result.index))
    wind = result.get("forecast_wind_speed_10m", pd.Series(np.nan, index=result.index))
    cloud = result.get("forecast_cloud_cover", pd.Series(np.nan, index=result.index))
    precip = result.get("forecast_precipitation", pd.Series(np.nan, index=result.index))
    cape = result.get("forecast_cape", pd.Series(np.nan, index=result.index))
    pressure_tendency = result.get("pressure_tendency_3h", pd.Series(0.0, index=result.index)).abs()

    spread = temp - dew
    conditions = [
        (rh >= 92) & (spread <= 2.2) & (wind <= 3.5) & night,
        (temp <= 2.0) & (wind <= 4.5) & (cloud <= 55) & night,
        temp >= 30.0,
        (cape >= 650) | ((precip >= 1.0) & (temp >= 16)),
        (pressure_tendency >= 2.0) | (wind >= 8.0),
        night & (wind <= 3.0) & (cloud <= 35),
    ]
    labels = [
        "fog_prone",
        "frost_prone",
        "heat",
        "convective",
        "frontal_or_windy",
        "radiative_night",
    ]
    result["regime"] = np.select(conditions, labels, default="ordinary")
    return result


def add_lead_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """Add operationally meaningful lead-time buckets."""
    result = df.copy()
    result["lead_bucket"] = pd.cut(
        result["lead_hours"],
        bins=[-0.1, 6, 12, 24, 48, 96, 240],
        labels=["001-006h", "007-012h", "013-024h", "025-048h", "049-096h", "097h+"],
    ).astype(str)
    return result
