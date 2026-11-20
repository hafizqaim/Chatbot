# Small read-only RAG service. The index is built offline (see README) and baked in.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MODEL_CACHE=/app/models \
    INDEX_DIR=/app/data/index

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY data/index ./data/index

# Pre-download the query-embedding model at build time so the container starts offline and fast.
# Skipped automatically if the index was built without dense embeddings.
RUN python -c "import json, os; from fastembed import TextEmbedding; \
m = json.load(open('data/index/manifest.json')).get('embed_model'); \
m and TextEmbedding(m, cache_dir=os.environ['MODEL_CACHE']); print('embedding model:', m)"

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 10000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import os, urllib.request as u; u.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT', '10000'), timeout=4)"

# Render injects $PORT; one worker keeps memory inside the 512 MB free instance.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1 --proxy-headers --forwarded-allow-ips="*"
