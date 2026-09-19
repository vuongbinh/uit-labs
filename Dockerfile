# syntax=docker/dockerfile:1

# ---- builder: resolve deps from the lockfile and bake the model into the HF cache ----
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.12.15 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0
WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Load the model through the app's own loader so the cache holds exactly what runtime reads.
ARG MODEL_ID=bvuong/visoBert-ensemble
ENV HF_HOME=/opt/hf
COPY model.py ./
RUN MODEL_ID="$MODEL_ID" .venv/bin/python -c "from model import Classifier; Classifier()"

# ---- runtime: no build tooling, no network needed ----
FROM python:3.12-slim
ARG MODEL_ID=bvuong/visoBert-ensemble
ENV MODEL_ID=$MODEL_ID \
    HF_HOME=/opt/hf \
    HF_HUB_OFFLINE=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    GRADIO_ANALYTICS_ENABLED=False \
    PYTHONUNBUFFERED=1 \
    PATH=/app/.venv/bin:$PATH

RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=builder --chown=app:app /opt/hf /opt/hf
COPY --chown=app:app app.py ui.py model.py batch.py ./

USER app
EXPOSE 7860
CMD ["python", "app.py"]
