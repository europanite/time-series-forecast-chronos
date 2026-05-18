from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from .io import make_future_frame, normalize_future, normalize_history, validate_prediction_length


@dataclass(frozen=True)
class ForecastConfig:
    model_id: str
    device: str = "cpu"
    prediction_length: int = 48
    quantile_levels: tuple[float, ...] = (0.1, 0.5, 0.9)
    id_column: str = "id"
    timestamp_column: str = "timestamp"
    target: str | list[str] = "target"
    batch_size: int | None = None
    context_length: int | None = None
    backend: str = "chronos2"


class Chronos2Forecaster:
    """Thin wrapper around Chronos2Pipeline.

    The model is downloaded from Hugging Face on first use and then stored under HF_HOME.
    With the Docker Compose volume, the second run uses the local cache.
    """

    def __init__(self, config: ForecastConfig) -> None:
        self.config = config
        self._pipeline = None

    @property
    def pipeline(self):  # noqa: ANN201 - external pipeline type differs across versions
        if self._pipeline is None:
            from chronos import Chronos2Pipeline

            self._pipeline = Chronos2Pipeline.from_pretrained(
                self.config.model_id,
                device_map=self.config.device,
            )
        return self._pipeline

    def predict(self, history_df: pd.DataFrame, future_df: pd.DataFrame | None = None) -> pd.DataFrame:
        cfg = self.config
        prediction_length = validate_prediction_length(cfg.prediction_length)
        history_df = normalize_history(history_df)
        future_df = normalize_future(future_df)

        if future_df is None:
            # Chronos-2 can infer future timestamps, but creating them explicitly makes
            # API responses and CSV output easier to inspect.
            future_df = make_future_frame(history_df, prediction_length, cfg.id_column, cfg.timestamp_column)

        kwargs: dict[str, object] = {
            "df": history_df,
            "future_df": future_df,
            "prediction_length": prediction_length,
            "quantile_levels": list(cfg.quantile_levels),
            "id_column": cfg.id_column,
            "timestamp_column": cfg.timestamp_column,
            "target": cfg.target,
        }
        if cfg.batch_size is not None:
            kwargs["batch_size"] = cfg.batch_size
        if cfg.context_length is not None:
            kwargs["context_length"] = cfg.context_length

        pred_df = self.pipeline.predict_df(**kwargs)
        return pred_df


class SeasonalNaiveForecaster:
    """Offline smoke-test backend.

    This is not a foundation model. It exists so the repository can validate data,
    Docker, CLI, and plotting without downloading a model.
    """

    def __init__(self, config: ForecastConfig, season_length: int = 48) -> None:
        self.config = config
        self.season_length = season_length

    def predict(self, history_df: pd.DataFrame, future_df: pd.DataFrame | None = None) -> pd.DataFrame:
        cfg = self.config
        prediction_length = validate_prediction_length(cfg.prediction_length)
        history_df = normalize_history(history_df)
        future_df = normalize_future(future_df)
        if future_df is None:
            future_df = make_future_frame(history_df, prediction_length, cfg.id_column, cfg.timestamp_column)

        rows: list[dict[str, object]] = []
        for item_id, group in history_df.groupby(cfg.id_column):
            group = group.sort_values(cfg.timestamp_column)
            values = group[cfg.target if isinstance(cfg.target, str) else cfg.target[0]].to_numpy()
            if len(values) >= self.season_length:
                template = values[-self.season_length :]
            else:
                template = values
            repeated = [float(template[i % len(template)]) for i in range(prediction_length)]
            target_future = future_df[future_df[cfg.id_column] == item_id].sort_values(cfg.timestamp_column)
            for i, (_, future_row) in enumerate(target_future.head(prediction_length).iterrows()):
                y = repeated[i]
                rows.append(
                    {
                        cfg.id_column: item_id,
                        cfg.timestamp_column: future_row[cfg.timestamp_column],
                        "predictions": y,
                        "0.1": y,
                        "0.5": y,
                        "0.9": y,
                    }
                )
        return pd.DataFrame(rows)


def build_forecaster(config: ForecastConfig):  # noqa: ANN201
    if config.backend == "chronos2":
        return Chronos2Forecaster(config)
    if config.backend == "seasonal_naive":
        return SeasonalNaiveForecaster(config)
    raise ValueError(f"Unknown backend: {config.backend}")
