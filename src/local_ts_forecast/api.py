from __future__ import annotations

from functools import lru_cache
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .forecaster import ForecastConfig, build_forecaster
from .settings import get_settings


class ForecastRequest(BaseModel):
    records: list[dict[str, Any]] = Field(..., description="History rows. Required: timestamp, target. Optional: id/covariates")
    future_records: list[dict[str, Any]] | None = Field(None, description="Known future covariates. Required: timestamp. Optional: id")
    prediction_length: int = 48
    model_id: str | None = None
    device: str | None = None
    backend: str = "chronos2"
    target: str = "target"
    quantile_levels: list[float] = Field(default_factory=lambda: [0.1, 0.5, 0.9])


class ForecastResponse(BaseModel):
    rows: list[dict[str, Any]]


app = FastAPI(title="Local Time-Series Forecast API", version="0.1.0")


@lru_cache(maxsize=4)
def get_forecaster(model_id: str, device: str, backend: str, target: str, prediction_length: int, quantiles: tuple[float, ...]):
    config = ForecastConfig(
        model_id=model_id,
        device=device,
        prediction_length=prediction_length,
        quantile_levels=quantiles,
        backend=backend,
        target=target,
    )
    return build_forecaster(config)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest) -> ForecastResponse:
    settings = get_settings()
    model_id = req.model_id or settings.model_id_for_backend(req.backend)
    device = req.device or settings.device
    try:
        history_df = pd.DataFrame(req.records)
        future_df = pd.DataFrame(req.future_records) if req.future_records is not None else None
        forecaster = get_forecaster(
            model_id,
            device,
            req.backend,
            req.target,
            req.prediction_length,
            tuple(req.quantile_levels),
        )
        pred_df = forecaster.predict(history_df, future_df)
        pred_df = pred_df.copy()
        for column in pred_df.columns:
            if pd.api.types.is_datetime64_any_dtype(pred_df[column]):
                pred_df[column] = pred_df[column].dt.strftime("%Y-%m-%dT%H:%M:%S")
        return ForecastResponse(rows=pred_df.to_dict(orient="records"))
    except Exception as exc:  # noqa: BLE001 - surface validation/model errors through API
        raise HTTPException(status_code=400, detail=str(exc)) from exc
