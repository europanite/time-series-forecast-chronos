# Local Time-Series Forecasting with Docker Compose

[![CI](https://github.com/europanite/time-series-forecast-chronos/actions/workflows/ci.yml/badge.svg)](https://github.com/europanite/time-series-forecast-chronos/actions/workflows/ci.yml)
[![CodeQL Advanced](https://github.com/europanite/time-series-forecast-chronos/actions/workflows/codeql.yml/badge.svg)](https://github.com/europanite/time-series-forecast-chronos/actions/workflows/codeql.yml)
[![pages-build-deployment](https://github.com/europanite/time-series-forecast-chronos/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/europanite/time-series-forecast-chronos/actions/workflows/pages/pages-build-deployment)
[![Pytest](https://github.com/europanite/time-series-forecast-chronos/actions/workflows/pytest.yml/badge.svg)](https://github.com/europanite/time-series-forecast-chronos/actions/workflows/pytest.yml)
[![Python Lint](https://github.com/europanite/time-series-forecast-chronos/actions/workflows/lint.yml/badge.svg)](https://github.com/europanite/time-series-forecast-chronos/actions/workflows/lint.yml)

A minimal Docker Compose repository for running local time-series foundation models.

The repository supports three interchangeable forecasting backends:

- `chronos2`: Chronos-2 style forecasting through `chronos-forecasting`
- `timesfm`: TimesFM 2.5 style forecasting through google-research/timesfm
- `seasonal_naive`: offline smoke-test backend that needs no model download

The main use case is to try zero-shot forecasting for data such as electricity demand, renewable energy generation, or market prices.

## What this repository does

- Runs both a CLI and an API with Docker Compose
- Lets you select `chronos2`, `timesfm`, or `seasonal_naive` at runtime
- Downloads model files from Hugging Face on the first model run
- Reuses the Docker volume cache after the first download
- Reads CSV input and writes forecast results as CSV and PNG
- Outputs a shared schema: `id`, `timestamp`, `predictions`, `0.1`, `0.5`, `0.9`
- Keeps a model-free backend for validating Docker, CSV parsing, and plotting

> Note: An internet connection is required for the first model download. After that, model files are kept in the `hf-cache` Docker volume and reused locally.

## Backend guide

| Backend | Model family | Best use in this repo | Notes |
|---|---|---|---|
| `chronos2` | Amazon Chronos-2 | Multivariate or covariate-informed experiments | Recommended when you want to include known future covariates |
| `timesfm` | Google TimesFM 2.5 | Simple univariate zero-shot comparison | The minimal wrapper intentionally ignores future covariate columns |
| `seasonal_naive` | No foundation model | Offline smoke test | Useful before downloading any model |

For electricity forecasting, a practical workflow is:

1. Validate the pipeline with `seasonal_naive`.
2. Run `chronos2` with history and known future covariates.
3. Run `timesfm` as a separate univariate foundation-model baseline.
4. Compare both against conventional baselines such as LightGBM, Prophet, or seasonal naive in a rolling backtest.

## 1. Initial setup

Default `.env` values:

```env
DEVICE=cpu
FORECAST_BACKEND=chronos2
CHRONOS_MODEL_ID=autogluon/chronos-2-small
TIMESFM_MODEL_ID=google/timesfm-2.5-200m-pytorch
HF_HOME=/cache/huggingface
PREDICTION_LENGTH=48
```

## 2. Build the container

```bash
docker compose build
```

The image installs both Chronos and TimesFM dependencies. Chronos-2 requires Transformers 4.x, so TimesFM 2.5 is loaded through the official `google-research/timesfm` package instead of the Hugging Face Transformers port, which currently requires Transformers 5.x. This keeps both backends installable in one Docker image.

## 3. Generate sample data

This command generates synthetic electricity-demand-like data with a 30-minute interval.

```bash
docker compose run --rm forecast \
  python -m local_ts_forecast.cli sample-data \
  --output-dir data \
  --prediction-length 48
```

Generated files:

- `data/sample_history.csv`
- `data/sample_future.csv`

`sample_history.csv` represents historical observations.  
`sample_future.csv` represents known future covariates such as weather forecasts.

## 4. Smoke test without downloading a model

```bash
docker compose run --rm forecast \
  python -m local_ts_forecast.cli validate \
  --input data/sample_history.csv \
  --future-input data/sample_future.csv \
  --prediction-length 48
```

This uses the `seasonal_naive` backend to validate the CSV format and the end-to-end pipeline. It does not download or run a foundation model.

You can also run the smoke-test backend through the main forecast command:

```bash
docker compose run --rm forecast \
  python -m local_ts_forecast.cli forecast \
  --backend seasonal_naive \
  --input data/sample_history.csv \
  --future-input data/sample_future.csv \
  --output outputs/forecast_seasonal_naive.csv \
  --plot outputs/forecast_seasonal_naive.png \
  --prediction-length 48
```

## 5. Run a forecast with Chronos-2

```bash
docker compose run --rm forecast \
  python -m local_ts_forecast.cli forecast \
  --backend chronos2 \
  --input data/sample_history.csv \
  --future-input data/sample_future.csv \
  --output outputs/forecast_chronos2.csv \
  --plot outputs/forecast_chronos2.png \
  --prediction-length 48
```

Chronos-2 is the better default in this repository when you want to test known future covariates or multiple related series.

## 6. Run a forecast with TimesFM

```bash
docker compose run --rm forecast \
  python -m local_ts_forecast.cli forecast \
  --backend timesfm \
  --input data/sample_history.csv \
  --future-input data/sample_future.csv \
  --output outputs/forecast_timesfm.csv \
  --plot outputs/forecast_timesfm.png \
  --prediction-length 48 \
  --context-length 1024
```

The TimesFM backend uses `TIMESFM_MODEL_ID` by default. You can override it directly:

```bash
docker compose run --rm forecast \
  python -m local_ts_forecast.cli forecast \
  --backend timesfm \
  --model-id google/timesfm-2.5-200m-pytorch \
  --input data/sample_history.csv \
  --future-input data/sample_future.csv \
  --output outputs/forecast_timesfm.csv \
  --plot outputs/forecast_timesfm.png \
  --prediction-length 48
```

Important limitation: the minimal TimesFM wrapper in this repository intentionally treats TimesFM as a univariate zero-shot model. It uses the target history and future timestamps, but it does not use additional future covariate columns. Use the `chronos2` backend for covariate-informed experiments.

## 7. Run as an API

```bash
docker compose up api
```

Health check:

```bash
curl http://localhost:8000/health
```

Run the API example:

```bash
docker compose run --rm forecast python scripts/api_example.py
```

To request TimesFM from the API, set `backend` to `timesfm` in the JSON payload.

## 8. Run with an NVIDIA GPU

This requires NVIDIA Container Toolkit on the host machine.

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm forecast \
  python -m local_ts_forecast.cli forecast \
  --backend chronos2 \
  --input data/sample_history.csv \
  --future-input data/sample_future.csv \
  --output outputs/forecast_chronos2.csv \
  --plot outputs/forecast_chronos2.png \
  --prediction-length 48 \
  --device cuda
```

For TimesFM, GPU behavior depends on the installed TimesFM/PyTorch backend and the host environment.

## CSV format

Minimum historical input:

```csv
id,timestamp,target
area_001,2026-01-01 00:00:00,4120.5
area_001,2026-01-01 00:30:00,4142.1
```

The `id` column is optional. If it is omitted, the series is treated as `series_1`.

Future covariates can be provided as follows:

```csv
id,timestamp,temperature_forecast,solar_forecast,holiday
area_001,2026-01-15 00:00:00,8.2,0.0,0
area_001,2026-01-15 00:30:00,8.0,0.0,0
```

Typical covariates for electricity forecasting include:

- Temperature forecast
- Solar generation forecast
- Wind forecast
- Calendar features
- Holiday flags
- Customer or area identifiers
- Market-related indicators

Backend behavior:

- `chronos2` passes the history and future dataframe to Chronos-2.
- `timesfm` uses the historical target values and future timestamps only.
- `seasonal_naive` repeats the latest seasonal pattern.
