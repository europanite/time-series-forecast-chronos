from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    model_id: str = os.getenv("MODEL_ID", "autogluon/chronos-2-small")
    device: str = os.getenv("DEVICE", "cpu")
    prediction_length: int = int(os.getenv("PREDICTION_LENGTH", "48"))
    hf_home: str = os.getenv("HF_HOME", "/cache/huggingface")


def get_settings() -> Settings:
    return Settings()
