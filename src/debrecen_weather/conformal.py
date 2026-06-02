"""Conformal calibration for quantile prediction intervals."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    clean = np.asarray(scores, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return 0.0
    rank = min(1.0, np.ceil((clean.size + 1) * (1 - alpha)) / clean.size)
    try:
        return float(np.quantile(clean, rank, method="higher"))
    except TypeError:
        return float(np.quantile(clean, rank, interpolation="higher"))


@dataclass
class GroupConformalInterval:
    """Conformalize a lower/upper quantile interval, optionally by groups."""

    alpha: float
    group_columns: list[str] = field(default_factory=list)
    min_group_size: int = 80
    global_adjustment_: float = 0.0
    group_adjustments_: dict[tuple[Any, ...], float] = field(default_factory=dict)

    def fit(
        self,
        df: pd.DataFrame,
        y_col: str,
        lower_col: str,
        upper_col: str,
    ) -> "GroupConformalInterval":
        y = df[y_col].to_numpy(dtype=float)
        lower = df[lower_col].to_numpy(dtype=float)
        upper = df[upper_col].to_numpy(dtype=float)
        scores = np.maximum.reduce([lower - y, y - upper, np.zeros_like(y)])
        self.global_adjustment_ = _conformal_quantile(scores, self.alpha)

        self.group_adjustments_ = {}
        if self.group_columns:
            work = df[self.group_columns].copy()
            work["_score"] = scores
            for key, group in work.groupby(self.group_columns, dropna=False):
                if not isinstance(key, tuple):
                    key = (key,)
                if len(group) >= self.min_group_size:
                    self.group_adjustments_[key] = _conformal_quantile(group["_score"].to_numpy(), self.alpha)
        return self

    def adjustments_for(self, df: pd.DataFrame) -> np.ndarray:
        if not self.group_columns:
            return np.full(len(df), self.global_adjustment_, dtype=float)
        adjustments = []
        for _, row in df[self.group_columns].iterrows():
            key = tuple(row[column] for column in self.group_columns)
            adjustments.append(self.group_adjustments_.get(key, self.global_adjustment_))
        return np.asarray(adjustments, dtype=float)

    def transform(self, df: pd.DataFrame, lower_col: str, upper_col: str) -> pd.DataFrame:
        result = df.copy()
        adjustment = self.adjustments_for(result)
        result[f"{lower_col}_conformal"] = result[lower_col].astype(float) - adjustment
        result[f"{upper_col}_conformal"] = result[upper_col].astype(float) + adjustment
        result["conformal_adjustment"] = adjustment
        return result
