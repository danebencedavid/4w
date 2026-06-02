"""Forecast ingestion utilities."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from time import sleep

import numpy as np
import pandas as pd
import requests

from .config import ensure_parent, resolve_path
from .constants import FORECAST_COLUMNS, WEATHER_VARIABLES
from .ingest_observations import real_date_window


OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"


def fetch_open_meteo_latest(cfg: dict, model: str, output_path: str | Path | None = None) -> pd.DataFrame:
    """Fetch the latest Open-Meteo point forecast for one model.

    This is intended for the live/dashboard path. Historical training should use
    Previous Runs or Single Runs so issued lead times are preserved.
    """
    station = cfg["station"]
    variables = list(cfg["forecast"]["hourly_variables"])
    retrieved = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    response = requests.get(
        OPEN_METEO_FORECAST_URL,
        params={
            "latitude": station["latitude"],
            "longitude": station["longitude"],
            "elevation": station["elevation_m"],
            "hourly": ",".join(variables),
            "models": model,
            "timezone": "UTC",
            "forecast_days": 7,
            "wind_speed_unit": "ms",
        },
        headers={"User-Agent": "debrecen-weather-calibration/0.1"},
        timeout=45,
    )
    response.raise_for_status()
    payload = response.json()
    hourly = payload.get("hourly", {})
    times = pd.to_datetime(hourly.get("time", []), utc=True)
    if len(times) == 0:
        return pd.DataFrame(columns=FORECAST_COLUMNS)

    frame = pd.DataFrame({"valid_time_utc": times})
    for variable in WEATHER_VARIABLES:
        frame[variable] = hourly.get(variable)
    frame["run_time_utc"] = retrieved
    frame["lead_hours"] = (frame["valid_time_utc"] - pd.Timestamp(retrieved)).dt.total_seconds() / 3600
    frame["lead_hours"] = frame["lead_hours"].round().astype(int)
    frame["model"] = model
    frame = frame.loc[frame["lead_hours"] >= 0, FORECAST_COLUMNS]

    output = Path(output_path) if output_path else resolve_path(cfg, "forecasts_raw")
    ensure_parent(output)
    if output.exists() and not frame.empty:
        previous = pd.read_csv(output)
        frame = pd.concat([previous, frame], ignore_index=True).drop_duplicates(
            ["run_time_utc", "valid_time_utc", "model"]
        )
    frame.to_csv(output, index=False)
    return frame


def fetch_all_open_meteo_latest(cfg: dict, output_path: str | Path | None = None) -> pd.DataFrame:
    """Fetch latest forecasts for all configured Open-Meteo model names."""
    frames = [fetch_open_meteo_latest(cfg, model, output_path=None) for model in cfg["forecast"]["models"]]
    frames = [frame.dropna(axis=1, how="all") for frame in frames if not frame.empty]
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=FORECAST_COLUMNS)
    result = result.reindex(columns=FORECAST_COLUMNS)
    output = Path(output_path) if output_path else resolve_path(cfg, "forecasts_raw")
    ensure_parent(output)
    result.to_csv(output, index=False)
    return result


def _previous_run_hourly_names(variables: list[str], previous_run_days: list[int]) -> list[str]:
    return [
        f"{variable}_previous_day{day}"
        for variable in variables
        for day in previous_run_days
    ]


def _previous_runs_response_to_frame(
    payload: dict,
    model: str,
    variables: list[str],
    previous_run_days: list[int],
) -> pd.DataFrame:
    hourly = payload.get("hourly", {})
    valid_times = pd.to_datetime(hourly.get("time", []), utc=True)
    if len(valid_times) == 0:
        return pd.DataFrame(columns=FORECAST_COLUMNS)

    frames = []
    for day in previous_run_days:
        lead = int(day) * 24
        frame = pd.DataFrame(
            {
                "valid_time_utc": valid_times,
                "run_time_utc": valid_times - pd.to_timedelta(lead, unit="h"),
                "lead_hours": lead,
                "model": model,
            }
        )
        for variable in WEATHER_VARIABLES:
            key = f"{variable}_previous_day{day}"
            if variable in variables and key in hourly:
                frame[variable] = hourly.get(key)
            else:
                frame[variable] = np.nan
        frames.append(frame)
    return pd.concat(frames, ignore_index=True).reindex(columns=FORECAST_COLUMNS)


def fetch_open_meteo_previous_runs(
    cfg: dict,
    model: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Fetch fixed-lead historical forecast values from Open-Meteo Previous Runs."""
    start, end = real_date_window(cfg, start_date=start_date, end_date=end_date)
    real_cfg = cfg.get("real_data", {})
    station = cfg["station"]
    variables = list(real_cfg.get("forecast_variables", cfg["forecast"]["hourly_variables"]))
    previous_run_days = [int(day) for day in real_cfg.get("previous_run_days", [1, 2])]
    hourly_names = _previous_run_hourly_names(variables, previous_run_days)

    response = requests.get(
        OPEN_METEO_PREVIOUS_RUNS_URL,
        params={
            "latitude": station["latitude"],
            "longitude": station["longitude"],
            "elevation": station["elevation_m"],
            "start_date": start,
            "end_date": end,
            "hourly": ",".join(hourly_names),
            "models": model,
            "timezone": "UTC",
            "wind_speed_unit": "ms",
            "precipitation_unit": "mm",
        },
        headers={"User-Agent": "debrecen-weather-calibration/0.1"},
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("error"):
        raise ValueError(payload.get("reason", f"Open-Meteo returned an error for {model}"))
    return _previous_runs_response_to_frame(payload, model, variables, previous_run_days)


def fetch_all_open_meteo_previous_runs(
    cfg: dict,
    start_date: str | None = None,
    end_date: str | None = None,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Fetch previous-run forecasts for all configured models."""
    frames = []
    pause = float(cfg.get("real_data", {}).get("request_pause_seconds", 1.1))
    for idx, model in enumerate(cfg["forecast"]["models"]):
        if idx:
            sleep(pause)
        try:
            frames.append(
                fetch_open_meteo_previous_runs(
                    cfg,
                    model=model,
                    start_date=start_date,
                    end_date=end_date,
                )
            )
        except requests.HTTPError as exc:
            print(f"Skipping model {model}: {exc}")
        except ValueError as exc:
            print(f"Skipping model {model}: {exc}")
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=FORECAST_COLUMNS)
    result = result.dropna(subset=["temperature_2m"], how="all")
    output = Path(output_path) if output_path else resolve_path(cfg, "forecasts_raw")
    ensure_parent(output)
    result.to_csv(output, index=False)
    return result


def read_forecasts(path: str | Path) -> pd.DataFrame:
    """Read forecast CSV and normalize timestamps."""
    df = pd.read_csv(path)
    df["run_time_utc"] = pd.to_datetime(df["run_time_utc"], utc=True)
    df["valid_time_utc"] = pd.to_datetime(df["valid_time_utc"], utc=True)
    return df.reindex(columns=FORECAST_COLUMNS)
