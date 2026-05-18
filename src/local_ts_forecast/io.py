from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_HISTORY_COLUMNS = {"id", "timestamp", "target"}
REQUIRED_FUTURE_COLUMNS = {"id", "timestamp"}


class DataValidationError(ValueError):
    pass


def read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise DataValidationError(f"CSV is empty: {path}")
    return df


def normalize_history(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "id" not in df.columns:
        df.insert(0, "id", "series_1")

    missing = REQUIRED_HISTORY_COLUMNS - set(df.columns)
    if missing:
        raise DataValidationError(
            "history CSV must contain columns: id, timestamp, target. "
            f"Missing: {sorted(missing)}"
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
    df["target"] = pd.to_numeric(df["target"], errors="raise")
    df = df.sort_values(["id", "timestamp"]).reset_index(drop=True)
    return df


def normalize_future(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None:
        return None
    df = df.copy()
    if "id" not in df.columns:
        df.insert(0, "id", "series_1")

    missing = REQUIRED_FUTURE_COLUMNS - set(df.columns)
    if missing:
        raise DataValidationError(
            "future CSV must contain columns: id, timestamp. "
            f"Missing: {sorted(missing)}"
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
    df = df.sort_values(["id", "timestamp"]).reset_index(drop=True)
    return df


def validate_prediction_length(prediction_length: int) -> int:
    if prediction_length <= 0:
        raise DataValidationError("prediction_length must be positive")
    return prediction_length


def infer_frequency(df: pd.DataFrame, id_column: str = "id", timestamp_column: str = "timestamp") -> str | None:
    freqs: list[str] = []
    for _, group in df.groupby(id_column):
        freq = pd.infer_freq(group[timestamp_column])
        if freq:
            freqs.append(freq)
    if not freqs:
        return None
    return freqs[0]


def make_future_frame(
    history_df: pd.DataFrame,
    prediction_length: int,
    id_column: str = "id",
    timestamp_column: str = "timestamp",
) -> pd.DataFrame:
    """Create future id/timestamp rows when no known covariate file is supplied."""
    rows: list[dict[str, object]] = []
    for item_id, group in history_df.groupby(id_column):
        group = group.sort_values(timestamp_column)
        freq = pd.infer_freq(group[timestamp_column])
        if freq is None:
            raise DataValidationError(
                f"Could not infer timestamp frequency for id={item_id!r}. "
                "Provide future-input CSV with explicit timestamps."
            )
        last_ts = group[timestamp_column].iloc[-1]
        future_index = pd.date_range(start=last_ts, periods=prediction_length + 1, freq=freq)[1:]
        for ts in future_index:
            rows.append({id_column: item_id, timestamp_column: ts})
    return pd.DataFrame(rows)


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def select_target_columns(target: str | Iterable[str]) -> str | list[str]:
    if isinstance(target, str):
        return target
    values = list(target)
    if not values:
        raise DataValidationError("target columns must not be empty")
    return values
