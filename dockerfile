# syntax=docker/dockerfile:1.7
ARG PYTHON_VERSION=3.12
ARG AIRFLOW_VERSION=3.3.2
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG CONSTRAINTS_URL=https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt

FROM apache/airflow:${AIRFLOW_VERSION}-python${PYTHON_VERSION} AS airflow

ARG TORCH_INDEX_URL
ARG CONSTRAINTS_URL

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libpq5 \
    && rm -rf /var/lib/apt/lists/*

USER airflow

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --index-url ${TORCH_INDEX_URL} \
        torch==2.14.1 \
        torchvision==0.29.1

COPY --chown=airflow:0 requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt --constraint "${CONSTRAINTS_URL}" \
    && rm -f /tmp/requirements.txt

# Airflow is installed in the image's *user site* (/home/airflow/.local/...),
# and the `airflow` CLI script runs under the system python — which only finds
# it via HOME. HOME is overridden below (to stay writable when AIRFLOW_UID is
# changed), so the user site dir is added explicitly to PYTHONPATH.
ENV PYTHONPATH=/opt/airflow/src:/home/airflow/.local/lib/python3.12/site-packages \
    MPLBACKEND=Agg \
    KERAS_BACKEND=tensorflow \
    TF_CPP_MIN_LOG_LEVEL=2 \
    BRAINSCAN_HOME=/opt/airflow \
    MLFLOW_TRACKING_URI=http://mlflow:5000 \
    HOME=/opt/airflow

COPY --chown=airflow:0 src/ /opt/airflow/src/

RUN mkdir -p /opt/airflow/dags /opt/airflow/logs /opt/airflow/models \
             /opt/airflow/artifacts /opt/airflow/data \
    && chown -R airflow:0 /opt/airflow/dags /opt/airflow/logs /opt/airflow/models \
             /opt/airflow/artifacts /opt/airflow/data \
    && chmod -R g+rwX /opt/airflow/logs /opt/airflow/models \
             /opt/airflow/artifacts /opt/airflow/data

CMD ["airflow", "scheduler"]


FROM python:${PYTHON_VERSION}-slim AS app

ARG TORCH_INDEX_URL

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src \
    MPLBACKEND=Agg \
    KERAS_BACKEND=tensorflow \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    BRAINSCAN_HOME=/app \
    HOME=/app/.home \
    MLFLOW_TRACKING_URI=http://mlflow:5000 \
    KERAS_HOME=/app/.keras \
    HUGGINGFACE_HUB_CACHE=/app/.cache/huggingface \
    TORCH_HOME=/app/.cache/torch

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libpq5 \
        libsm6 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel \
    && pip install --index-url ${TORCH_INDEX_URL} \
        torch==2.14.1 \
        torchvision==0.29.1

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && rm -f /tmp/requirements.txt

COPY src/ /app/src/

RUN mkdir -p /app/models /app/artifacts /app/data /app/logs \
             /app/.keras /app/.cache/torch /app/.cache/huggingface /app/.home \
    && chown -R root:0 /app \
    && chmod -R g+rwX /app/models /app/artifacts /app/data /app/logs \
             /app/.keras /app/.cache /app/.home

RUN useradd --create-home --uid 10001 brainscan \
    && chown -R brainscan:0 /app/.home \
    && chmod -R g+rwX /app/.home
USER brainscan

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "brainscan.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
