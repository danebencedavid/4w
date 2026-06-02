"""Shared schema constants."""

OBSERVATION_TIME = "valid_time_utc"
FORECAST_TIME = "run_time_utc"
MODEL = "model"
LEAD = "lead_hours"

WEATHER_VARIABLES = [
    "temperature_2m",
    "dew_point_2m",
    "relative_humidity_2m",
    "pressure_msl",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "precipitation",
    "cloud_cover",
    "cape",
]

OBSERVATION_VARIABLES = WEATHER_VARIABLES + ["visibility"]

FORECAST_COLUMNS = [FORECAST_TIME, OBSERVATION_TIME, LEAD, MODEL] + WEATHER_VARIABLES
OBSERVATION_COLUMNS = [OBSERVATION_TIME] + OBSERVATION_VARIABLES

FORECAST_PREFIX = "forecast_"
OBSERVED_PREFIX = "observed_"
