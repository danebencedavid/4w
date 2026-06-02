# Research Plan

## Project

Adaptive probabilistic weather forecast calibration for Debrecen, Hungary.

The goal is not to replace numerical weather prediction. The goal is to post-process
existing model forecasts so they become more useful at one local point: Debrecen
International Airport / LHDC, which also acts as a proxy for the city.

## Main Question

Can adaptive, regime-aware calibration improve local forecast accuracy and uncertainty
reliability for Debrecen?

## Hypotheses

1. Raw NWP forecasts have local biases that depend on lead time, season, hour of day,
   and weather regime.
2. Multi-model disagreement is predictive of forecast uncertainty.
3. Quantile regression improves interval sharpness, but conformal calibration improves
   empirical coverage.
4. Regime-aware conformal calibration helps most during frost nights, heat events,
   fog-prone mornings, windy frontal passages, and convective weather.

## Experimental Design

Use time-based splits only:

- Train: earliest 60% of available rows.
- Validation/calibration: next 20%.
- Test: final 20%.

For a production study, split by absolute dates:

- Train through 2024-12-31.
- Validate on 2025.
- Test on 2026 onward.

## Targets

Start with `temperature_2m`; then add:

- `dew_point_2m`
- `wind_speed_10m`
- `wind_gusts_10m`
- `precipitation`
- `visibility` or fog-risk proxy

## Models

Baselines:

- Raw model forecast.
- Rolling climatological bias correction by lead bucket, hour, and month.
- Multi-model mean.

ML:

- Gradient boosting median correction.
- Quantile gradient boosting for 5%, 10%, 50%, 90%, 95%.
- Conformalized quantile regression.
- Regime-aware conformal calibration.

## Metrics

Point metrics:

- MAE
- RMSE
- Bias
- Skill score against raw forecast

Probabilistic metrics:

- Pinball loss
- Prediction interval coverage
- Prediction interval width
- Calibration error

Event metrics:

- Brier score
- Reliability diagram
- ROC-AUC and average precision when sample size permits

## Visualizations

Data:

- Station and model-location map.
- Missingness calendar.
- Seasonal climatology.
- Forecast bias heatmap by hour and lead.
- Error distribution by season/regime.
- Wind rose.
- Model disagreement over time.

Results:

- Meteogram with raw forecast, calibrated median, observations, and intervals.
- MAE/RMSE by lead.
- Skill improvement by model and lead.
- Empirical vs nominal coverage.
- Reliability curves for frost/heat/wind.
- PIT-style rank histogram where full distributions are available.
- Case studies for frost, heat, fog, frontal wind, and convective rain.

## Practical Deliverables

- Reproducible Python pipeline.
- Data schema that supports real APIs and offline sample data.
- Trained model bundle.
- Metrics table.
- Static HTML dashboard/report.
- Clear README explaining how to rerun and extend.
