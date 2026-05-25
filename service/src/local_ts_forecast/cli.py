from __future__ import annotations

import argparse
import os
from pathlib import Path

from .forecaster import ForecastConfig, build_forecaster
from .io import ensure_parent, read_csv
from .plotting import plot_forecast
from .sample_data import write_sample_data
from .settings import get_settings


def parse_quantiles(value: str) -> tuple[float, ...]:
    values = tuple(float(v.strip()) for v in value.split(",") if v.strip())
    if not values:
        raise argparse.ArgumentTypeError("quantiles must not be empty")
    for v in values:
        if not 0 < v < 1:
            raise argparse.ArgumentTypeError("quantiles must be between 0 and 1")
    return values


def forecast_command(args: argparse.Namespace) -> None:
    settings = get_settings()
    history_df = read_csv(args.input)
    future_df = read_csv(args.future_input) if args.future_input else None

    config = ForecastConfig(
        model_id=args.model_id or settings.model_id_for_backend(args.backend),
        device=args.device or settings.device,
        prediction_length=args.prediction_length or settings.prediction_length,
        quantile_levels=args.quantiles,
        target=args.target,
        batch_size=args.batch_size,
        context_length=args.context_length,
        backend=args.backend,
    )
    forecaster = build_forecaster(config)
    pred_df = forecaster.predict(history_df, future_df)

    output_path = ensure_parent(args.output)
    pred_df.to_csv(output_path, index=False)
    print(f"forecast_csv={output_path}")

    if args.plot:
        plot_path = plot_forecast(history_df, pred_df, args.plot, target_column=args.target)
        print(f"forecast_plot={plot_path}")


def sample_data_command(args: argparse.Namespace) -> None:
    history_path, future_path = write_sample_data(
        output_dir=args.output_dir,
        periods=args.periods,
        prediction_length=args.prediction_length,
        freq=args.freq,
    )
    print(f"history_csv={history_path}")
    print(f"future_csv={future_path}")


def validate_command(args: argparse.Namespace) -> None:
    history_df = read_csv(args.input)
    future_df = read_csv(args.future_input) if args.future_input else None
    config = ForecastConfig(
        model_id="offline-smoke-test",
        device="cpu",
        prediction_length=args.prediction_length,
        backend="seasonal_naive",
    )
    pred_df = build_forecaster(config).predict(history_df, future_df)
    print("validation=ok")
    print(f"rows={len(pred_df)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local time-series forecasting with Chronos-2, TimesFM, or seasonal naive")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample = subparsers.add_parser("sample-data", help="Generate synthetic electricity-like sample data")
    sample.add_argument("--output-dir", default="data", help="Directory to write sample CSVs")
    sample.add_argument("--periods", type=int, default=14 * 48, help="History length")
    sample.add_argument("--prediction-length", type=int, default=int(os.getenv("PREDICTION_LENGTH", "48")))
    sample.add_argument("--freq", default="30min")
    sample.set_defaults(func=sample_data_command)

    forecast = subparsers.add_parser("forecast", help="Run forecasting")
    forecast.add_argument("--input", required=True, help="History CSV. Required columns: timestamp,target. Optional: id")
    forecast.add_argument("--future-input", help="Future covariate CSV. Required columns: timestamp. Optional: id/covariates")
    forecast.add_argument("--output", default="outputs/forecast.csv")
    forecast.add_argument("--plot", default="outputs/forecast.png")
    forecast.add_argument("--prediction-length", type=int, default=int(os.getenv("PREDICTION_LENGTH", "48")))
    forecast.add_argument("--model-id", help="Hugging Face model id. Defaults to CHRONOS_MODEL_ID or TIMESFM_MODEL_ID based on backend.")
    forecast.add_argument("--device", default=os.getenv("DEVICE", "cpu"), choices=["cpu", "cuda", "auto"])
    forecast.add_argument("--quantiles", type=parse_quantiles, default=(0.1, 0.5, 0.9))
    forecast.add_argument("--target", default="target")
    forecast.add_argument("--batch-size", type=int)
    forecast.add_argument("--context-length", type=int)
    forecast.add_argument("--backend", choices=["chronos2", "timesfm", "seasonal_naive"], default=os.getenv("FORECAST_BACKEND", "chronos2"))
    forecast.set_defaults(func=forecast_command)

    validate = subparsers.add_parser("validate", help="Validate CSV and pipeline without downloading a model")
    validate.add_argument("--input", required=True)
    validate.add_argument("--future-input")
    validate.add_argument("--prediction-length", type=int, default=int(os.getenv("PREDICTION_LENGTH", "48")))
    validate.set_defaults(func=validate_command)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
