# Local Time-Series Forecasting with Docker Compose

A minimal Docker Compose repository for running a local time-series foundation model.

The main use case is to try zero-shot probabilistic forecasting with Chronos-2 style models for time-series data such as electricity demand, renewable energy generation, or market prices.

## What this repository does

- Runs both a CLI and an API with Docker Compose
- Downloads a Chronos-2 style model from Hugging Face on the first run
- Reuses the Docker volume cache after the first download
- Reads CSV input and writes forecast results as CSV and PNG
- Outputs probabilistic forecasts with `0.1`, `0.5`, and `0.9` quantiles
- Includes a `seasonal_naive` backend for smoke testing without downloading a model

> Note: An internet connection is required only for the first model download. After that, the model files are kept in the `hf-cache` Docker volume and reused locally.

## 1. Initial setup

For a first CPU run, use:

```env
DEVICE=cpu
MODEL_ID=autogluon/chronos-2-small
PREDICTION_LENGTH=48
```

For a lighter test, use:

```env
MODEL_ID=amazon/chronos-bolt-tiny
```

## 2. Build the container

```bash
docker compose build
```

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

This uses the `seasonal_naive` backend to validate the CSV format and the end-to-end pipeline. It does not download or run a Chronos model.

## 5. Run a forecast with Chronos-2

```bash
docker compose run --rm forecast \
  python -m local_ts_forecast.cli forecast \
  --input data/sample_history.csv \
  --future-input data/sample_future.csv \
  --output outputs/forecast.csv \
  --plot outputs/forecast.png \
  --prediction-length 48
```

Outputs:

- `outputs/forecast.csv`
- `outputs/forecast.png`

The CSV includes probabilistic forecast columns such as `q10`, `q50`, and `q90`.

## 6. Run as an API

```bash
docker compose up api
```

In another terminal, run:

```bash
docker compose run --rm forecast python scripts/api_example.py
```

Health check:

```bash
curl http://localhost:8000/health
```

## 7. Run with an NVIDIA GPU

This requires NVIDIA Container Toolkit on the host machine.

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm forecast \
  python -m local_ts_forecast.cli forecast \
  --input data/sample_history.csv \
  --future-input data/sample_future.csv \
  --output outputs/forecast.csv \
  --plot outputs/forecast.png \
  --prediction-length 48 \
  --device cuda
```

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

## Interview explanation note

This repository is intended to make the implementation image concrete, not only to mention time-series foundation models by name.

Example explanation:

> I built a small Docker Compose repository that runs Chronos-2 locally. It reads demand history and known future covariates such as weather forecasts from CSV, then outputs a 48-step forecast for the next day. It does not only output a point forecast; it also outputs the 0.1, 0.5, and 0.9 quantiles, so the result can be connected to risk evaluation using forecast intervals. In a real electricity business setting, I would evaluate this not only with MAPE, but also with imbalance cost, trading P&L, and operational risk.

## Practical limitations

- This is a minimal local validation repository, not a production forecasting platform.
- The first model download requires network access.
- Model licenses, internal-use permission, and data handling rules must be checked separately.
- Real electricity forecasting requires more engineering around holidays, weather forecasts, customer attributes, installed capacity, market rules, missing-value handling, outlier correction, and rolling backtests.

## Suggested next steps

- Add a rolling backtest command.
- Compare Chronos-2 against LightGBM, Prophet, and seasonal naive baselines.
- Add electricity-domain metrics such as MAPE, pinball loss, imbalance cost, and trading P&L.
- Add a Streamlit or FastAPI dashboard for reviewing forecast results.
