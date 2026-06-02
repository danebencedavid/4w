# Data Sources

## Observation Point

Primary station:

- Name: Debrecen International Airport
- ICAO: `LHDC`
- WMO: `12882`
- Meteostat ID: `12882`
- Approximate station coordinates: `47.4833, 21.6000`
- Elevation: `108 m`
- Time zone: `Europe/Budapest`

The airport is a practical proxy for Debrecen because it has aviation observations and
stable station identifiers. A later extension can compare the airport against a downtown
coordinate to quantify urban/airport differences.

## Observation Sources

Recommended order:

1. IEM ASOS/METAR CSV endpoint for historical `LHDC` airport observations.
2. NOAA Aviation Weather API for recent/live METAR data.
3. NOAA GHCNh for a more official long-term hourly archive.
4. HungaroMet ODP for official Hungarian daily series and sanity checks.
5. Open-Meteo Historical Weather API as a reanalysis fallback if station observations are unavailable.

Current implementation:

```text
https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py
```

It requests `tmpf`, `dwpf`, `relh`, `drct`, `sknt`, `gust`, `p01i`, `mslp`, and `vsby`;
then converts Fahrenheit to Celsius, knots to m/s, inches to mm, and miles to meters.
Sub-hourly METARs are aggregated to hourly verification rows.

## Forecast Sources

For machine-learning training, use issued model forecasts rather than stitched analyses.

Recommended Open-Meteo APIs:

- Previous Runs API: fixed lead-time verification.
- Single Runs API: full forecast horizon for an initialization time.
- Forecast API: operational/latest dashboard values.

Current implementation uses:

```text
https://previous-runs-api.open-meteo.com/v1/forecast
```

For example, `temperature_2m_previous_day1` is treated as the forecast issued 24 hours
before valid time, and `temperature_2m_previous_day2` as the forecast issued 48 hours
before valid time.

Large historical requests are split into chunks by `real_data.forecast_chunk_days`.
This is important for GitHub Actions, where occasional TLS/read timeouts from public
weather APIs should be retried rather than failing the whole site build.

Candidate models for Debrecen:

- DWD ICON-EU
- ECMWF IFS 0.25 or IFS HRES where available
- ECMWF AIFS
- NOAA GFS
- Météo-France ARPEGE Europe

## Required Forecast Schema

Forecast CSVs should contain:

```text
run_time_utc
valid_time_utc
lead_hours
model
temperature_2m
dew_point_2m
relative_humidity_2m
pressure_msl
wind_speed_10m
wind_direction_10m
wind_gusts_10m
precipitation
cloud_cover
cape
```

## Required Observation Schema

Observation CSVs should contain:

```text
valid_time_utc
temperature_2m
dew_point_2m
relative_humidity_2m
pressure_msl
wind_speed_10m
wind_direction_10m
wind_gusts_10m
precipitation
visibility
```
