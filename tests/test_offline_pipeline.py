import tempfile
import unittest
from pathlib import Path

import pandas as pd

from debrecen_weather.align import add_multimodel_spread, align_forecasts_observations
from debrecen_weather.evaluate import evaluate_events
from debrecen_weather.features import build_features
from debrecen_weather.modeling import fit_calibrator
from debrecen_weather.sample_data import generate_forecasts, generate_observations


class OfflinePipelineTest(unittest.TestCase):
    def test_synthetic_alignment_features_and_training(self):
        observations = generate_observations(days=25, seed=7)
        forecasts = generate_forecasts(
            observations,
            models=["icon_eu", "gfs_global"],
            lead_hours=[1, 6, 24],
            seed=7,
        )
        aligned = add_multimodel_spread(align_forecasts_observations(forecasts, observations))
        features = build_features(aligned, timezone="Europe/Budapest")

        self.assertGreater(len(features), 1000)
        self.assertIn("regime", features)
        self.assertIn("lead_bucket", features)
        self.assertIn("model_spread_temperature_2m", features)

        calibrator, validation, test = fit_calibrator(
            features,
            target="temperature_2m",
            quantiles=[0.1, 0.5, 0.9],
            interval_low=0.1,
            interval_high=0.9,
            validation_fraction=0.2,
            test_fraction=0.2,
            seed=7,
            max_train_rows=3000,
            conformal_group_columns=["lead_bucket", "regime"],
            min_group_size_for_conformal=10,
        )

        self.assertIn("q50", test)
        self.assertIn("q10_conformal", test)
        self.assertIn("q90_conformal", test)
        self.assertGreater(len(calibrator.feature_columns), 5)
        self.assertFalse(test["q50"].isna().any())

    def test_temperature_event_directions(self):
        predictions = pd.DataFrame(
            {
                "split": ["test", "test", "test"],
                "observed_temperature_2m": [-2.0, 1.0, 31.0],
                "q10": [-3.0, 0.0, 29.0],
                "q50": [-2.0, 1.0, 31.0],
                "q90": [-1.0, 2.0, 33.0],
            }
        )
        metrics = evaluate_events(
            predictions,
            target="temperature_2m",
            quantiles=[0.1, 0.5, 0.9],
            event_thresholds={
                "frost": {"threshold": 0.0, "direction": "below"},
                "warm_day": {"threshold": 30.0, "direction": "above"},
            },
        )
        frost = metrics.loc[metrics["event"] == "frost"].iloc[0]
        warm_day = metrics.loc[metrics["event"] == "warm_day"].iloc[0]
        self.assertAlmostEqual(frost["base_rate"], 1 / 3)
        self.assertAlmostEqual(warm_day["base_rate"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
