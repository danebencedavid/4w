"""Model training and prediction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .conformal import GroupConformalInterval
from .config import ensure_parent, resolve_path


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _quantile_estimator(quantile: float, seed: int, max_iter: int = 160):
    try:
        return HistGradientBoostingRegressor(
            loss="quantile",
            quantile=quantile,
            max_iter=max_iter,
            learning_rate=0.055,
            l2_regularization=0.02,
            min_samples_leaf=35,
            random_state=seed,
        )
    except TypeError:
        return GradientBoostingRegressor(
            loss="quantile",
            alpha=quantile,
            n_estimators=max_iter,
            learning_rate=0.045,
            max_depth=3,
            min_samples_leaf=20,
            random_state=seed,
        )


def default_feature_columns(df: pd.DataFrame, target: str) -> tuple[list[str], list[str]]:
    """Select robust default numeric/categorical features present in a dataframe."""
    numeric_candidates = [
        "lead_hours",
        "forecast_temperature_2m",
        "forecast_dew_point_2m",
        "forecast_relative_humidity_2m",
        "forecast_pressure_msl",
        "forecast_wind_speed_10m",
        "forecast_wind_gusts_10m",
        "forecast_precipitation",
        "forecast_cloud_cover",
        "forecast_cape",
        "forecast_temp_dewpoint_spread",
        "forecast_wind_dir_sin",
        "forecast_wind_dir_cos",
        f"model_mean_{target}",
        f"model_spread_{target}",
        f"model_range_{target}",
        "model_mean_temperature_2m",
        "model_spread_temperature_2m",
        "model_range_temperature_2m",
        "model_mean_dew_point_2m",
        "model_spread_dew_point_2m",
        "model_range_dew_point_2m",
        "hour_sin",
        "hour_cos",
        "doy_sin",
        "doy_cos",
        "pressure_tendency_3h",
        "temperature_tendency_3h",
        "obs_lag_1h_temperature_2m",
        "obs_lag_3h_temperature_2m",
        "obs_lag_6h_temperature_2m",
        "obs_lag_12h_temperature_2m",
        "obs_lag_24h_temperature_2m",
        "obs_lag_1h_dew_point_2m",
        "obs_lag_3h_dew_point_2m",
        "obs_lag_1h_pressure_msl",
        "obs_lag_3h_pressure_msl",
        "obs_lag_1h_wind_speed_10m",
        "obs_lag_3h_wind_speed_10m",
    ]
    categorical_candidates = ["model", "season", "lead_bucket", "regime"]
    numeric = list(
        dict.fromkeys(
            column
            for column in numeric_candidates
            if column in df.columns and df[column].notna().any()
        )
    )
    categorical = list(dict.fromkeys(column for column in categorical_candidates if column in df.columns))
    return numeric, categorical


def make_preprocessor(numeric_columns: list[str], categorical_columns: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_columns),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", _one_hot_encoder()),
                    ]
                ),
                categorical_columns,
            ),
        ],
        remainder="drop",
    )


@dataclass
class ForecastCalibrator:
    target: str
    quantiles: list[float]
    numeric_columns: list[str]
    categorical_columns: list[str]
    models: dict[float, Pipeline]
    conformal: GroupConformalInterval | None = None
    interval_low: float = 0.1
    interval_high: float = 0.9

    @property
    def feature_columns(self) -> list[str]:
        return self.numeric_columns + self.categorical_columns

    def predict(self, df: pd.DataFrame, apply_conformal: bool = True) -> pd.DataFrame:
        result = df.copy()
        raw_col = f"forecast_{self.target}"
        if raw_col not in result:
            raise ValueError(f"Raw forecast column {raw_col!r} is required for residual calibration")
        for quantile in self.quantiles:
            name = quantile_name(quantile)
            residual = self.models[quantile].predict(result[self.feature_columns])
            result[name] = result[raw_col].astype(float) + residual

        q_cols = [quantile_name(q) for q in self.quantiles]
        sorted_values = np.sort(result[q_cols].to_numpy(dtype=float), axis=1)
        for idx, col in enumerate(q_cols):
            result[col] = sorted_values[:, idx]

        low_col = quantile_name(self.interval_low)
        high_col = quantile_name(self.interval_high)
        if apply_conformal and self.conformal is not None and low_col in result and high_col in result:
            result = self.conformal.transform(result, lower_col=low_col, upper_col=high_col)
        return result


def quantile_name(quantile: float) -> str:
    return f"q{int(round(quantile * 100)):02d}"


def time_based_split(
    df: pd.DataFrame,
    validation_fraction: float,
    test_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by sorted unique valid times."""
    work = df.copy()
    work["valid_time_utc"] = pd.to_datetime(work["valid_time_utc"], utc=True)
    unique_times = pd.Index(sorted(work["valid_time_utc"].unique()))
    n = len(unique_times)
    test_start = int(np.floor(n * (1 - test_fraction)))
    validation_start = int(np.floor(n * (1 - test_fraction - validation_fraction)))
    train_times = set(unique_times[:validation_start])
    validation_times = set(unique_times[validation_start:test_start])
    test_times = set(unique_times[test_start:])
    train = work[work["valid_time_utc"].isin(train_times)].copy()
    validation = work[work["valid_time_utc"].isin(validation_times)].copy()
    test = work[work["valid_time_utc"].isin(test_times)].copy()
    return train, validation, test


def fit_calibrator(
    df: pd.DataFrame,
    target: str,
    quantiles: list[float],
    interval_low: float,
    interval_high: float,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
    max_train_rows: int,
    conformal_group_columns: list[str],
    min_group_size_for_conformal: int,
) -> tuple[ForecastCalibrator, pd.DataFrame, pd.DataFrame]:
    """Train quantile models and conformalize them using validation data."""
    y_col = f"observed_{target}"
    raw_col = f"forecast_{target}"
    if y_col not in df.columns:
        raise ValueError(f"Target column {y_col!r} is missing")
    if raw_col not in df.columns:
        raise ValueError(f"Raw forecast column {raw_col!r} is missing")
    work = df.dropna(subset=[y_col, raw_col]).copy()
    train, validation, test = time_based_split(work, validation_fraction, test_fraction)

    if max_train_rows and len(train) > max_train_rows:
        train_fit = train.sample(n=max_train_rows, random_state=seed).sort_values("valid_time_utc")
    else:
        train_fit = train

    numeric_columns, categorical_columns = default_feature_columns(work, target)
    preprocessor = make_preprocessor(numeric_columns, categorical_columns)
    models: dict[float, Pipeline] = {}
    for quantile in quantiles:
        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", _quantile_estimator(quantile, seed=seed + int(quantile * 1000))),
            ]
        )
        residual = train_fit[y_col].astype(float) - train_fit[raw_col].astype(float)
        pipeline.fit(train_fit[numeric_columns + categorical_columns], residual)
        models[quantile] = pipeline

    calibrator = ForecastCalibrator(
        target=target,
        quantiles=quantiles,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        models=models,
        interval_low=interval_low,
        interval_high=interval_high,
    )

    validation_predictions = calibrator.predict(validation, apply_conformal=False)
    alpha = max(0.001, 1.0 - (interval_high - interval_low))
    conformal = GroupConformalInterval(
        alpha=alpha,
        group_columns=[column for column in conformal_group_columns if column in validation_predictions],
        min_group_size=min_group_size_for_conformal,
    ).fit(
        validation_predictions,
        y_col=y_col,
        lower_col=quantile_name(interval_low),
        upper_col=quantile_name(interval_high),
    )
    calibrator.conformal = conformal
    validation_predictions = calibrator.predict(validation, apply_conformal=True)
    test_predictions = calibrator.predict(test, apply_conformal=True)
    validation_predictions["split"] = "validation"
    test_predictions["split"] = "test"
    return calibrator, validation_predictions, test_predictions


def train_from_config(cfg: dict, target: str | None = None) -> tuple[ForecastCalibrator, pd.DataFrame]:
    """Train from the configured feature table and save model/predictions."""
    target = target or cfg["training"]["target"]
    features = pd.read_csv(resolve_path(cfg, "features"))
    calibrator, validation_predictions, test_predictions = fit_calibrator(
        features,
        target=target,
        quantiles=[float(q) for q in cfg["training"]["quantiles"]],
        interval_low=float(cfg["training"]["interval_low"]),
        interval_high=float(cfg["training"]["interval_high"]),
        validation_fraction=float(cfg["training"]["validation_fraction"]),
        test_fraction=float(cfg["training"]["test_fraction"]),
        seed=int(cfg["training"]["random_seed"]),
        max_train_rows=int(cfg["training"].get("max_train_rows", 0)),
        conformal_group_columns=list(cfg["training"].get("conformal_group_columns", [])),
        min_group_size_for_conformal=int(cfg["training"].get("min_group_size_for_conformal", 80)),
    )
    predictions = pd.concat([validation_predictions, test_predictions], ignore_index=True)
    model_path = resolve_path(cfg, "model_bundle")
    predictions_path = resolve_path(cfg, "predictions")
    ensure_parent(model_path)
    ensure_parent(predictions_path)
    joblib.dump(calibrator, model_path)
    predictions.to_csv(predictions_path, index=False)
    return calibrator, predictions


def load_model(path: str | Path) -> ForecastCalibrator:
    return joblib.load(path)
