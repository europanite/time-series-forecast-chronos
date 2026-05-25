FROM python:3.11-slim

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
RUN pip install --upgrade pip \
    && pip install "torch==${PYTORCH_VERSION}" --index-url "${PYTORCH_INDEX_URL}" \
    && pip install -r requirements.txt

COPY . .

CMD ["python", "-m", "local_ts_forecast.cli", "--help"]
