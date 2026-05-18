from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def build_sample_data(
    periods: int = 14 * 48,
    prediction_length: int = 48,
    freq: str = "30min",
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-01-01 00:00:00")
    index = pd.date_range(start=start, periods=periods + prediction_length, freq=freq)

    half_hours = np.arange(len(index))
    daily = np.sin(2 * np.pi * half_hours / 48)
    weekly = np.sin(2 * np.pi * half_hours / (48 * 7))
    temp = 10 + 8 * daily + 2 * weekly + rng.normal(0, 0.8, len(index))
    solar = np.maximum(0, np.sin(2 * np.pi * (half_hours % 48 - 12) / 48))
    holiday = ((index.dayofweek >= 5)).astype(int)

    # Electricity-like demand. It increases when it is cold/hot, drops on holidays,
    # and has half-hour seasonality. This is only sample data.
    target = (
        4200
        + 220 * daily
        + 120 * weekly
        + 14 * np.abs(temp - 18)
        - 180 * holiday
        - 90 * solar
        + rng.normal(0, 45, len(index))
    )

    df = pd.DataFrame(
        {
            "id": "area_001",
            "timestamp": index,
            "target": target.round(2),
            "temperature_forecast": temp.round(2),
            "solar_forecast": solar.round(4),
            "holiday": holiday,
        }
    )
    history = df.iloc[:periods].copy()
    future = df.iloc[periods:].drop(columns=["target"]).copy()
    return history, future


def write_sample_data(
    output_dir: str | Path = "data",
    periods: int = 14 * 48,
    prediction_length: int = 48,
    freq: str = "30min",
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history, future = build_sample_data(periods=periods, prediction_length=prediction_length, freq=freq)
    history_path = output_dir / "sample_history.csv"
    future_path = output_dir / "sample_future.csv"
    history.to_csv(history_path, index=False)
    future.to_csv(future_path, index=False)
    return history_path, future_path
