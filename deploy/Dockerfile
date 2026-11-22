# CPU image. Build from your INTERNAL mirrors (set PIP_INDEX_URL / base image registry).
FROM python:3.12-slim
ARG PIP_INDEX_URL
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src \
    CARDAGENT_ROOT=/app HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2
RUN apt-get update && apt-get install -y --no-install-recommends \
      tesseract-ocr tesseract-ocr-ara libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-base.txt requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY config ./config
COPY registry ./registry
COPY tools ./tools
RUN useradd -r -u 10001 agent && mkdir -p /app/data/index /app/data/reference_store /app/models \
    && chown -R agent /app/data
USER agent
# Fail the image build on an invalid registry. The vectors live in Qdrant and are synced at deploy
# time (registry-sync service in docker-compose.yml / a Kubernetes Job), not baked into the image.
RUN python tools/build_index.py --validate-only
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s CMD curl -fs http://localhost:8080/health/ready || exit 1
CMD ["uvicorn", "cardagent.api.app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
