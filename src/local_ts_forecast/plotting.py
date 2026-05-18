from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .io import ensure_parent, normalize_history


def plot_forecast(
    history_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    output_path: str | Path,
    id_value: str | None = None,
    id_column: str = "id",
    timestamp_column: str = "timestamp",
    target_column: str = "target",
) -> Path:
    history_df = normalize_history(history_df)
    pred_df = pred_df.copy()
    pred_df[timestamp_column] = pd.to_datetime(pred_df[timestamp_column])

    if id_value is None:
        id_value = str(history_df[id_column].iloc[0])

    hist = history_df[history_df[id_column].astype(str) == id_value].tail(240)
    pred = pred_df[pred_df[id_column].astype(str) == id_value]

    if hist.empty:
        raise ValueError(f"No history found for id={id_value}")
    if pred.empty:
        raise ValueError(f"No prediction found for id={id_value}")

    path = ensure_parent(output_path)
    fig = plt.figure(figsize=(12, 4))
    plt.plot(hist[timestamp_column], hist[target_column], label="history")
    y_column = "predictions" if "predictions" in pred.columns else "0.5"
    plt.plot(pred[timestamp_column], pred[y_column], label="forecast")

    if "0.1" in pred.columns and "0.9" in pred.columns:
        plt.fill_between(pred[timestamp_column], pred["0.1"], pred["0.9"], alpha=0.2, label="p10-p90")

    plt.title(f"Forecast: {id_value}")
    plt.xlabel("timestamp")
    plt.ylabel(target_column)
    plt.legend()
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)
    return path
