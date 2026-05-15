# syntax=docker/dockerfile:1.6
#
# Two-stage build:
#   1. `builder` installs pip deps and pre-downloads the three large model
#      assets into HF / spacy cache dirs.
#   2. final image copies only the populated caches + site-packages + repo,
#      avoiding the apt build deps that pulled torch wheels.
#
# Models pre-baked:
#   - sentence-transformers/all-MiniLM-L6-v2  (~90 MB)
#   - cross-encoder/nli-deberta-v3-xsmall     (~80 MB)
#   - spacy en_core_web_sm                    (~12 MB)
#
# Artifacts (chroma index, intent model, drift timeline) are COPYed in so
# the first request is instant — no cold-start build of the index.

ARG PY=3.11

# =========================================================================
FROM python:${PY}-slim AS builder
# =========================================================================

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/hf-cache \
    SENTENCE_TRANSFORMERS_HOME=/opt/hf-cache/sentence-transformers \
    HF_HUB_DISABLE_TELEMETRY=1

# build deps for compiled wheels (chromadb -> hnswlib, sentence-transformers
# pulls torch which has its own wheel so usually no gcc needed, but leave
# build-essential available for safety on platforms without prebuilt wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install python deps. CPU-only torch wheel to avoid the multi-GB CUDA pull.
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        "torch==2.5.1+cpu" && \
    pip install -r requirements.txt && \
    pip install gunicorn==23.0.0

# Pre-download models so the runtime image has no network dependency.
RUN python - <<'PY'
from sentence_transformers import SentenceTransformer, CrossEncoder
print("downloading all-MiniLM-L6-v2 ...")
SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("downloading nli-deberta-v3-xsmall ...")
CrossEncoder("cross-encoder/nli-deberta-v3-xsmall")
PY

RUN python -m spacy download en_core_web_sm

# =========================================================================
FROM python:${PY}-slim AS final
# =========================================================================

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf-cache \
    SENTENCE_TRANSFORMERS_HOME=/opt/hf-cache/sentence-transformers \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    PORT=7860

# Runtime-only system libs (no compilers needed at runtime).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Pull installed site-packages, console scripts, and model caches from
# the builder layer. This keeps the final image free of apt build deps.
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin/gunicorn /usr/local/bin/gunicorn
COPY --from=builder /opt/hf-cache /opt/hf-cache

WORKDIR /app

# Code + artifacts. Copying artifacts/ means the chroma index, intent model,
# and drift timeline are already present — first hit is instant.
COPY src/ ./src/
COPY templates/ ./templates/
COPY app.py run_all.py ./
COPY artifacts/ ./artifacts/

# Non-root user for hosts that enforce it (HF Spaces, Cloud Run, etc.)
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app /opt/hf-cache
USER appuser

EXPOSE 7860

# HF Spaces routes traffic to $app_port (7860 from the README front matter).
# Other hosts override via $PORT at runtime; we default to 7860 so the Spaces
# health check finds a listener even if PORT is not injected.
CMD ["sh", "-c", "gunicorn app:app --workers=2 --timeout=120 --bind 0.0.0.0:${PORT:-7860}"]
