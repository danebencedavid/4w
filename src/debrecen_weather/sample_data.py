"""Synthetic Debrecen-like data for offline development and tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ensure_parent, resolve_path
from .constants import FORECAST_COLUMNS, OBSERVATION_COLUMNS


@dataclass(frozen=True)
class SampleDataSummary:
    observations_path: Path
    forecasts_path: Path
    observation_rows: int
    forecast_rows: int


def _relative_humidity_from_dewpoint(temp_c: np.ndarray, dewpoint_c: np.ndarray) -> np.ndarray:
    saturation_actual = np.exp((17.625 * dewpoint_c) / (243.04 + dewpoint_c))
    saturation_temp = np.exp((17.625 * temp_c) / (243.04 + temp_c))
    return np.clip(100 * saturation_actual / saturation_temp, 8, 100)


def generate_observations(
    start: str = "2024-01-01T00:00:00Z",
    days: int = 180,
    seed: int = 42,
) -> pd.DataFrame:
    """Create hourly Debrecen-like observations.

    The synthetic series intentionally includes local phenomena relevant to Debrecen:
    hot summer days, frost-prone clear nights, fog-prone high-humidity mornings, and
    windy frontal periods.
    """
    rng = np.random.default_rng(seed)
    times = pd.date_range(start=start, periods=days * 24, freq="h", tz="UTC")
    hour = times.hour.to_numpy()
    doy = times.dayofyear.to_numpy()

    seasonal = 11.5 + 13.5 * np.sin(2 * np.pi * (doy - 105) / 365)
    diurnal = 5.0 * np.sin(2 * np.pi * (hour - 7) / 24)
    synoptic = 2.0 * np.sin(np.arange(len(times)) * 2 * np.pi / (24 * 7))
    noise = rng.normal(0, 0.9, len(times))
    temp = seasonal + diurnal + synoptic + noise

    # A few structured episodes make evaluation plots more interesting.
    episode_index = np.arange(len(times))
    temp += np.where((episode_index > 24 * 35) & (episode_index < 24 * 43), -5.0, 0.0)
    temp += np.where((episode_index > 24 * 130) & (episode_index < 24 * 142), 4.0, 0.0)

    dewpoint_spread = 3.5 + 2.0 * np.sin(2 * np.pi * (hour - 13) / 24) + rng.gamma(1.4, 0.9, len(times))
    dewpoint = temp - np.clip(dewpoint_spread, 0.6, 12.0)
    relative_humidity = _relative_humidity_from_dewpoint(temp, dewpoint)

    pressure = 1014 + 6 * np.sin(episode_index * 2 * np.pi / (24 * 9)) + rng.normal(0, 1.0, len(times))
    pressure += np.where((episode_index > 24 * 80) & (episode_index < 24 * 83), -12.0, 0.0)

    frontal_pulse = np.where((episode_index > 24 * 80) & (episode_index < 24 * 84), 1.0, 0.0)
    wind_speed = rng.gamma(2.0, 1.5, len(times)) + 5.5 * frontal_pulse
    wind_direction = (210 + 80 * np.sin(episode_index * 2 * np.pi / (24 * 5)) + rng.normal(0, 35, len(times))) % 360
    wind_gusts = wind_speed + rng.gamma(1.2, 1.8, len(times)) + 3.0 * frontal_pulse

    convective_prob = np.clip((temp - 18) / 25, 0, 0.45)
    stratiform_prob = np.where(pressure < 1007, 0.22, 0.04)
    rain_event = rng.random(len(times)) < np.maximum(convective_prob, stratiform_prob)
    precipitation = np.where(rain_event, rng.gamma(1.1, 1.8, len(times)), 0.0)
    precipitation = np.where((hour >= 12) & (hour <= 19), precipitation * 1.4, precipitation)

    cloud_cover = np.clip(25 + 0.65 * relative_humidity + 8 * precipitation + rng.normal(0, 15, len(times)), 0, 100)
    cape = np.clip((temp - 18) * 120 + rng.normal(0, 160, len(times)), 0, 2200)

    fog_prone = (relative_humidity > 93) & (wind_speed < 3.0) & ((hour <= 8) | (hour >= 22))
    visibility = np.where(fog_prone, rng.uniform(300, 1800, len(times)), rng.uniform(8000, 30000, len(times)))
    visibility = np.where(precipitation > 1.0, np.minimum(visibility, rng.uniform(2500, 9000, len(times))), visibility)

    obs = pd.DataFrame(
        {
            "valid_time_utc": times,
            "temperature_2m": temp,
            "dew_point_2m": dewpoint,
            "relative_humidity_2m": relative_humidity,
            "pressure_msl": pressure,
            "wind_speed_10m": wind_speed,
            "wind_direction_10m": wind_direction,
            "wind_gusts_10m": wind_gusts,
            "precipitation": precipitation,
            "cloud_cover": cloud_cover,
            "cape": cape,
            "visibility": visibility,
        }
    )
    return obs[OBSERVATION_COLUMNS]


def generate_forecasts(
    observations: pd.DataFrame,
    models: list[str],
    lead_hours: list[int],
    seed: int = 42,
) -> pd.DataFrame:
    """Create synthetic issued forecasts from observations with realistic biases."""
    rng = np.random.default_rng(seed + 7)
    obs = observations.copy()
    obs["valid_time_utc"] = pd.to_datetime(obs["valid_time_utc"], utc=True)
    obs["hour"] = obs["valid_time_utc"].dt.hour
    obs["doy"] = obs["valid_time_utc"].dt.dayofyear

    model_temp_bias = {
        "icon_eu": -0.05,
        "ecmwf_ifs025": -0.20,
        "ecmwf_aifs025_single": 0.05,
        "gfs_global": 0.45,
    }
    model_wind_bias = {
        "icon_eu": 0.10,
        "ecmwf_ifs025": -0.15,
        "ecmwf_aifs025_single": -0.05,
        "gfs_global": 0.35,
    }

    frames = []
    base = np.arange(len(obs))
    for model in models:
        for lead in lead_hours:
            lead_scale = np.sqrt(max(lead, 1))
            night = (obs["hour"].to_numpy() <= 6) | (obs["hour"].to_numpy() >= 21)
            summer = (obs["doy"].to_numpy() >= 150) & (obs["doy"].to_numpy() <= 250)
            frost_prone = (obs["temperature_2m"].to_numpy() < 2.5) & night
            heat_prone = (obs["temperature_2m"].to_numpy() > 29) & summer

            temp_bias = model_temp_bias.get(model, 0.0) + 0.015 * lead
            temp_bias += np.where(frost_prone, 0.7, 0.0)
            temp_bias += np.where(heat_prone, -0.45, 0.0)
            temp_noise = rng.normal(0, 0.35 + 0.18 * lead_scale, len(obs))

            dew_noise = rng.normal(0, 0.45 + 0.12 * lead_scale, len(obs))
            pressure_noise = rng.normal(0, 0.8 + 0.25 * lead_scale, len(obs))
            wind_noise = rng.normal(0, 0.45 + 0.18 * lead_scale, len(obs))
            precip_noise = rng.gamma(0.7, 0.25 + 0.02 * lead, len(obs))

            forecast_temp = obs["temperature_2m"].to_numpy() + temp_bias + temp_noise
            forecast_dew = obs["dew_point_2m"].to_numpy() + 0.6 * temp_bias + dew_noise
            forecast_rh = _relative_humidity_from_dewpoint(forecast_temp, forecast_dew)
            forecast_wind = np.clip(
                obs["wind_speed_10m"].to_numpy()
                + model_wind_bias.get(model, 0.0)
                + 0.02 * lead
                + wind_noise,
                0,
                None,
            )
            forecast_gusts = np.clip(
                obs["wind_gusts_10m"].to_numpy() + 1.1 * model_wind_bias.get(model, 0.0) + wind_noise,
                0,
                None,
            )
            forecast_precip = np.clip(obs["precipitation"].to_numpy() * rng.normal(1.0, 0.45, len(obs)) + precip_noise - 0.2, 0, None)
            forecast_cloud = np.clip(obs["cloud_cover"].to_numpy() + rng.normal(0, 12 + lead, len(obs)), 0, 100)
            forecast_cape = np.clip(obs["cape"].to_numpy() + rng.normal(0, 80 + 8 * lead, len(obs)), 0, None)

            frame = pd.DataFrame(
                {
                    "run_time_utc": obs["valid_time_utc"] - pd.to_timedelta(lead, unit="h"),
                    "valid_time_utc": obs["valid_time_utc"],
                    "lead_hours": lead,
                    "model": model,
                    "temperature_2m": forecast_temp,
                    "dew_point_2m": forecast_dew,
                    "relative_humidity_2m": forecast_rh,
                    "pressure_msl": obs["pressure_msl"].to_numpy() + pressure_noise,
                    "wind_speed_10m": forecast_wind,
                    "wind_direction_10m": (obs["wind_direction_10m"].to_numpy() + rng.normal(0, 8 + lead, len(obs))) % 360,
                    "wind_gusts_10m": forecast_gusts,
                    "precipitation": forecast_precip,
                    "cloud_cover": forecast_cloud,
                    "cape": forecast_cape,
                }
            )
            # Avoid a perfectly complete grid: real public data has occasional gaps.
            keep = rng.random(len(frame)) > (0.002 + 0.0002 * lead)
            frames.append(frame.loc[keep])

    forecasts = pd.concat(frames, ignore_index=True)
    return forecasts[FORECAST_COLUMNS].sort_values(["valid_time_utc", "lead_hours", "model"])


def write_sample_data(cfg: dict, days: int | None = None, seed: int | None = None) -> SampleDataSummary:
    """Generate and save synthetic observation and forecast CSVs."""
    sample_cfg = cfg.get("sample_data", {})
    days = days or int(sample_cfg.get("days", 180))
    seed = int(seed if seed is not None else cfg.get("training", {}).get("random_seed", 42))
    start = sample_cfg.get("start", "2024-01-01T00:00:00Z")

    observations = generate_observations(start=start, days=days, seed=seed)
    forecasts = generate_forecasts(
        observations,
        models=list(cfg["forecast"]["models"]),
        lead_hours=list(cfg["forecast"]["lead_hours"]),
        seed=seed,
    )

    obs_path = resolve_path(cfg, "observations_raw")
    forecast_path = resolve_path(cfg, "forecasts_raw")
    ensure_parent(obs_path)
    ensure_parent(forecast_path)
    observations.to_csv(obs_path, index=False)
    forecasts.to_csv(forecast_path, index=False)

    return SampleDataSummary(
        observations_path=obs_path,
        forecasts_path=forecast_path,
        observation_rows=len(observations),
        forecast_rows=len(forecasts),
    )
