from __future__ import annotations

from dataclasses import dataclass

import numpy as np
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


def _primary_target(target: str | list[str]) -> str:
    if isinstance(target, str):
        return target
    if not target:
        raise ValueError("target must not be empty")
    return target[0]


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


class TimesFMForecaster:
    """TimesFM backend for univariate zero-shot forecasting.

    TimesFM 2.5 is loaded through the official google-research/timesfm package.
    This avoids the Transformers 5.x dependency required by the Hugging Face
    Transformers port, because Chronos-2 currently requires Transformers 4.x.

    Known future covariate columns are intentionally ignored by this minimal backend.
    Use the Chronos-2 backend when you want native multivariate/covariate-informed
    forecasting in this repository.
    """

    def __init__(self, config: ForecastConfig) -> None:
        self.config = config
        self._model = None

    @property
    def model(self):  # noqa: ANN201 - external model type differs across versions
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _resolve_torch_device(self) -> str:
        import torch

        if self.config.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if self.config.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("DEVICE=cuda was requested, but CUDA is not available in this container.")
            return "cuda"
        return "cpu"

    @staticmethod
    def _patch_timesfm_init_for_hub_kwargs(model_cls: object) -> None:
        """Ignore extra Hub metadata kwargs passed by some huggingface_hub versions."""

        if getattr(model_cls, "_local_ts_forecast_hub_kwargs_patch", False):
            return

        original_init = model_cls.__init__
        ignored_kwargs = {
            "adapter_kwargs",
            "cache_dir",
            "device",
            "force_download",
            "local_files_only",
            "paper_url",
            "proxies",
            "repo_url",
            "resume_download",
            "revision",
            "token",
        }

        def compatible_init(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
            for key in ignored_kwargs:
                kwargs.pop(key, None)
            return original_init(self, *args, **kwargs)

        model_cls.__init__ = compatible_init
        model_cls._local_ts_forecast_hub_kwargs_patch = True

    def _load_model(self):  # noqa: ANN201 - external model type differs across versions
        import torch
        import timesfm

        self._resolve_torch_device()
        model_id = self.config.model_id
        if model_id == "google/timesfm-2.5-200m-transformers":
            # Backward compatibility for older .env files and CLI invocations.
            model_id = "google/timesfm-2.5-200m-pytorch"

        torch.set_float32_matmul_precision("high")
        model_cls = timesfm.TimesFM_2p5_200M_torch
        # The official TimesFM 2.5 PyTorch wrapper does not accept a
        # `device` keyword in `from_pretrained`. Its inner torch module
        # selects cuda:0 when CUDA is visible, otherwise CPU.
        try:
            model = model_cls.from_pretrained(model_id)
        except TypeError as exc:
            if "unexpected keyword argument" not in str(exc):
                raise
            self._patch_timesfm_init_for_hub_kwargs(model_cls)
            model = model_cls.from_pretrained(model_id)

        max_context = cfg_context if (cfg_context := self.config.context_length) and cfg_context > 0 else 1024
        max_horizon = max(128, validate_prediction_length(self.config.prediction_length))
        model.compile(
            timesfm.ForecastConfig(
                max_context=max_context,
                max_horizon=max_horizon,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
        )
        return model

    def predict(self, history_df: pd.DataFrame, future_df: pd.DataFrame | None = None) -> pd.DataFrame:
        cfg = self.config
        prediction_length = validate_prediction_length(cfg.prediction_length)
        target_column = _primary_target(cfg.target)
        history_df = normalize_history(history_df)
        future_df = normalize_future(future_df)

        if future_df is None:
            future_df = make_future_frame(history_df, prediction_length, cfg.id_column, cfg.timestamp_column)

        inputs: list[np.ndarray] = []
        item_ids: list[object] = []
        future_by_id: dict[object, pd.DataFrame] = {}
        for item_id, group in history_df.groupby(cfg.id_column):
            group = group.sort_values(cfg.timestamp_column)
            values = group[target_column].astype(float).to_numpy()
            if cfg.context_length is not None and cfg.context_length > 0:
                values = values[-cfg.context_length :]
            if values.size == 0:
                raise ValueError(f"No values found for id={item_id!r}")
            inputs.append(values)
            item_ids.append(item_id)
            future_by_id[item_id] = future_df[future_df[cfg.id_column] == item_id].sort_values(cfg.timestamp_column)

        point_forecast, quantile_forecast = self.model.forecast(
            horizon=prediction_length,
            inputs=inputs,
        )
        point_forecast = np.asarray(point_forecast)
        quantile_forecast = None if quantile_forecast is None else np.asarray(quantile_forecast)
        if point_forecast.shape[1] < prediction_length:
            raise ValueError(
                "TimesFM returned fewer forecast steps than requested: "
                f"returned={point_forecast.shape[1]}, requested={prediction_length}"
            )
        point_forecast = point_forecast[:, :prediction_length]
        if quantile_forecast is not None:
            quantile_forecast = quantile_forecast[:, :prediction_length, :]

        rows: list[dict[str, object]] = []
        for series_idx, item_id in enumerate(item_ids):
            target_future = future_by_id[item_id].head(prediction_length)
            if len(target_future) < prediction_length:
                raise ValueError(
                    f"future rows for id={item_id!r} are shorter than prediction_length={prediction_length}"
                )
            for step_idx, (_, future_row) in enumerate(target_future.iterrows()):
                prediction = float(point_forecast[series_idx, step_idx])
                row: dict[str, object] = {
                    cfg.id_column: item_id,
                    cfg.timestamp_column: future_row[cfg.timestamp_column],
                    "predictions": prediction,
                }
                row.update(self._extract_quantiles(quantile_forecast, series_idx, step_idx, prediction))
                rows.append(row)
        return pd.DataFrame(rows)

    def _extract_quantiles(
        self,
        quantile_forecast: np.ndarray | None,
        series_idx: int,
        step_idx: int,
        fallback_prediction: float,
    ) -> dict[str, float]:
        result: dict[str, float] = {}
        for q in self.config.quantile_levels:
            column = f"{q:.1f}"
            result[column] = fallback_prediction

        if quantile_forecast is None or quantile_forecast.ndim != 3:
            return result

        # Official TimesFM 2.5 tensors expose the last dimension as:
        # mean, q10, q20, ..., q90.
        for q in self.config.quantile_levels:
            if not 0 < q < 1:
                continue
            decile = int(round(q * 10))
            if 1 <= decile <= 9:
                idx = decile  # 0 is mean; 1..9 are q10..q90.
                if idx < quantile_forecast.shape[2]:
                    result[f"{q:.1f}"] = float(quantile_forecast[series_idx, step_idx, idx])
        return result


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
        target_column = _primary_target(cfg.target)
        for item_id, group in history_df.groupby(cfg.id_column):
            group = group.sort_values(cfg.timestamp_column)
            values = group[target_column].to_numpy()
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
    if config.backend == "timesfm":
        return TimesFMForecaster(config)
    if config.backend == "seasonal_naive":
        return SeasonalNaiveForecaster(config)
    raise ValueError(f"Unknown backend: {config.backend}")
