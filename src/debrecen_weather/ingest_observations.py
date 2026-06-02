"""Observation ingestion utilities."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from .config import ensure_parent, resolve_path
from .constants import OBSERVATION_COLUMNS


AWC_METAR_URL = "https://aviationweather.gov/api/data/metar"
IEM_ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def _as_float(record: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = record.get(key)
        if value in (None, "", "M"):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def parse_awc_metar_json(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Parse NOAA Aviation Weather Center METAR JSON into the project schema."""
    rows = []
    for record in records:
        valid = record.get("obsTime") or record.get("reportTime") or record.get("receiptTime")
        if not valid:
            continue
        rows.append(
            {
                "valid_time_utc": pd.to_datetime(valid, utc=True),
                "temperature_2m": _as_float(record, "temp", "temp_c"),
                "dew_point_2m": _as_float(record, "dewp", "dewpoint", "dewpoint_c"),
                "relative_humidity_2m": _as_float(record, "rh", "relativeHumidity"),
                "pressure_msl": _as_float(record, "slp", "mslp"),
                "wind_speed_10m": _as_float(record, "wspd", "wind_speed_kt"),
                "wind_direction_10m": _as_float(record, "wdir", "wind_dir_degrees"),
                "wind_gusts_10m": _as_float(record, "wgst", "wind_gust_kt"),
                "precipitation": _as_float(record, "precip"),
                "cloud_cover": None,
                "cape": None,
                "visibility": _as_float(record, "visib", "visibility"),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=OBSERVATION_COLUMNS)
    # NOAA aviation JSON typically reports wind in knots and visibility in statute miles.
    for col in ["wind_speed_10m", "wind_gusts_10m"]:
        if col in df:
            df[col] = df[col].astype(float) * 0.514444
    if "visibility" in df:
        df["visibility"] = df["visibility"].astype(float) * 1609.344
    return df.reindex(columns=OBSERVATION_COLUMNS)


def fetch_live_metar(cfg: dict, output_path: str | Path | None = None) -> pd.DataFrame:
    """Fetch the latest METAR for Debrecen LHDC from NOAA Aviation Weather Center."""
    station = cfg["station"]["icao"]
    response = requests.get(
        AWC_METAR_URL,
        params={"ids": station, "format": "json"},
        headers={"User-Agent": "debrecen-weather-calibration/0.1"},
        timeout=30,
    )
    response.raise_for_status()
    records = response.json()
    if isinstance(records, dict):
        records = records.get("data", [records])
    df = parse_awc_metar_json(records)
    if output_path is None:
        output = resolve_path(cfg, "observations_raw")
    else:
        output = Path(output_path)
    ensure_parent(output)
    if output.exists() and not df.empty:
        previous = pd.read_csv(output)
        df = pd.concat([previous, df], ignore_index=True).drop_duplicates("valid_time_utc")
    df.to_csv(output, index=False)
    return df


def real_date_window(cfg: dict, start_date: str | None = None, end_date: str | None = None) -> tuple[str, str]:
    """Resolve the real-data date window.

    If dates are omitted, use a rolling window that ends a few days before today so
    station archives and previous-run forecast archives have time to settle.
    """
    real_cfg = cfg.get("real_data", {})
    if start_date is None:
        start_date = real_cfg.get("start_date")
    if end_date is None:
        end_date = real_cfg.get("end_date")
    if end_date is None:
        lag = int(real_cfg.get("data_lag_days", 7))
        end = datetime.now(timezone.utc).date() - timedelta(days=lag)
    else:
        end = date.fromisoformat(str(end_date))
    if start_date is None:
        days = int(real_cfg.get("history_days", 180))
        start = end - timedelta(days=days - 1)
    else:
        start = date.fromisoformat(str(start_date))
    if start > end:
        raise ValueError(f"start_date {start} is after end_date {end}")
    return start.isoformat(), end.isoformat()


def _fahrenheit_to_celsius(value: pd.Series) -> pd.Series:
    return (value - 32.0) * 5.0 / 9.0


def _parse_iem_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.replace({"M": np.nan, "": np.nan, " ": np.nan}), errors="coerce")


def parse_iem_asos_csv(text: str) -> pd.DataFrame:
    """Parse IEM ASOS CSV data into the project observation schema."""
    df = pd.read_csv(StringIO(text), comment="#")
    if df.empty:
        return pd.DataFrame(columns=OBSERVATION_COLUMNS)
    if "valid" not in df.columns:
        raise ValueError("IEM ASOS response did not contain a 'valid' column")

    valid = pd.to_datetime(df["valid"], utc=True, errors="coerce")
    parsed = pd.DataFrame({"valid_time_utc": valid})
    parsed["temperature_2m"] = _fahrenheit_to_celsius(_parse_iem_numeric(df.get("tmpf", pd.Series(dtype=float))))
    parsed["dew_point_2m"] = _fahrenheit_to_celsius(_parse_iem_numeric(df.get("dwpf", pd.Series(dtype=float))))
    parsed["relative_humidity_2m"] = _parse_iem_numeric(df.get("relh", pd.Series(dtype=float)))
    parsed["pressure_msl"] = _parse_iem_numeric(df.get("mslp", pd.Series(dtype=float)))
    parsed["wind_speed_10m"] = _parse_iem_numeric(df.get("sknt", pd.Series(dtype=float))) * 0.514444
    parsed["wind_direction_10m"] = _parse_iem_numeric(df.get("drct", pd.Series(dtype=float)))
    parsed["wind_gusts_10m"] = _parse_iem_numeric(df.get("gust", pd.Series(dtype=float))) * 0.514444
    parsed["precipitation"] = _parse_iem_numeric(df.get("p01i", pd.Series(dtype=float))) * 25.4
    parsed["cloud_cover"] = np.nan
    parsed["cape"] = np.nan
    parsed["visibility"] = _parse_iem_numeric(df.get("vsby", pd.Series(dtype=float))) * 1609.344
    parsed = parsed.dropna(subset=["valid_time_utc"]).sort_values("valid_time_utc")

    # METAR reports are often sub-hourly. Aggregate to one verification row per hour.
    parsed["valid_time_utc"] = parsed["valid_time_utc"].dt.floor("h")
    aggregations = {
        "temperature_2m": "mean",
        "dew_point_2m": "mean",
        "relative_humidity_2m": "mean",
        "pressure_msl": "mean",
        "wind_speed_10m": "mean",
        "wind_direction_10m": "mean",
        "wind_gusts_10m": "max",
        "precipitation": "sum",
        "cloud_cover": "mean",
        "cape": "mean",
        "visibility": "mean",
    }
    hourly = parsed.groupby("valid_time_utc", as_index=False).agg(aggregations)
    return hourly.reindex(columns=OBSERVATION_COLUMNS)


def fetch_iem_asos_observations(
    cfg: dict,
    start_date: str | None = None,
    end_date: str | None = None,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Fetch historical METAR/ASOS observations for LHDC from IEM."""
    start, end = real_date_window(cfg, start_date=start_date, end_date=end_date)
    params: list[tuple[str, str]] = [
        ("station", cfg["station"]["icao"]),
        ("sts", f"{start}T00:00:00Z"),
        ("ets", f"{end}T23:59:59Z"),
        ("tz", "Etc/UTC"),
        ("format", "onlycomma"),
        ("latlon", "no"),
        ("elev", "no"),
        ("missing", "empty"),
        ("trace", "0.0001"),
        ("report_type", "3"),
        ("report_type", "4"),
    ]
    for column in ["tmpf", "dwpf", "relh", "drct", "sknt", "p01i", "mslp", "vsby", "gust"]:
        params.append(("data", column))

    response = requests.get(
        IEM_ASOS_URL,
        params=params,
        headers={"User-Agent": "debrecen-weather-calibration/0.1"},
        timeout=90,
    )
    response.raise_for_status()
    df = parse_iem_asos_csv(response.text)
    output = Path(output_path) if output_path else resolve_path(cfg, "observations_raw")
    ensure_parent(output)
    df.to_csv(output, index=False)
    return df


def fetch_open_meteo_archive_observations(
    cfg: dict,
    start_date: str | None = None,
    end_date: str | None = None,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Fetch reanalysis-based observations from Open-Meteo as a fallback."""
    start, end = real_date_window(cfg, start_date=start_date, end_date=end_date)
    station = cfg["station"]
    variables = [
        "temperature_2m",
        "dew_point_2m",
        "relative_humidity_2m",
        "pressure_msl",
        "wind_speed_10m",
        "wind_direction_10m",
        "wind_gusts_10m",
        "precipitation",
        "cloud_cover",
    ]
    response = requests.get(
        OPEN_METEO_ARCHIVE_URL,
        params={
            "latitude": station["latitude"],
            "longitude": station["longitude"],
            "elevation": station["elevation_m"],
            "start_date": start,
            "end_date": end,
            "hourly": ",".join(variables),
            "timezone": "UTC",
            "wind_speed_unit": "ms",
            "precipitation_unit": "mm",
        },
        headers={"User-Agent": "debrecen-weather-calibration/0.1"},
        timeout=90,
    )
    response.raise_for_status()
    hourly = response.json().get("hourly", {})
    df = pd.DataFrame({"valid_time_utc": pd.to_datetime(hourly.get("time", []), utc=True)})
    for variable in variables:
        df[variable] = hourly.get(variable)
    df["cape"] = np.nan
    df["visibility"] = np.nan
    df = df.reindex(columns=OBSERVATION_COLUMNS)
    output = Path(output_path) if output_path else resolve_path(cfg, "observations_raw")
    ensure_parent(output)
    df.to_csv(output, index=False)
    return df


def read_observations(path: str | Path) -> pd.DataFrame:
    """Read observation CSV and normalize timestamps."""
    df = pd.read_csv(path)
    df["valid_time_utc"] = pd.to_datetime(df["valid_time_utc"], utc=True)
    return df.reindex(columns=OBSERVATION_COLUMNS)
