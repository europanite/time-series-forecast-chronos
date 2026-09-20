FROM python:3.12-slim

ARG PYTORCH_VERSION=2.7.1
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu126

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.txt ./
COPY src ./src

RUN python -m pip install --upgrade pip \
    && python -m pip install "torch==${PYTORCH_VERSION}" --index-url "${PYTORCH_INDEX_URL}" \
    && python -m pip install -r requirements.txt \
    && python -m pip install -e .

COPY . .

CMD ["python", "-m", "local_ts_forecast.cli", "--help"]
