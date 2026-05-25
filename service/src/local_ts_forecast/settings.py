from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    backend: str = os.getenv("FORECAST_BACKEND", "chronos2")
    chronos_model_id: str = os.getenv("CHRONOS_MODEL_ID", os.getenv("MODEL_ID", "autogluon/chronos-2-small"))
    timesfm_model_id: str = os.getenv("TIMESFM_MODEL_ID", "google/timesfm-2.5-200m-pytorch")
    device: str = os.getenv("DEVICE", "cpu")
    prediction_length: int = int(os.getenv("PREDICTION_LENGTH", "48"))
    hf_home: str = os.getenv("HF_HOME", "/cache/huggingface")

    def model_id_for_backend(self, backend: str | None = None) -> str:
        selected = backend or self.backend
        if selected == "timesfm":
            return self.timesfm_model_id
        return self.chronos_model_id

    @property
    def model_id(self) -> str:
        return self.model_id_for_backend(self.backend)


def get_settings() -> Settings:
    return Settings()
