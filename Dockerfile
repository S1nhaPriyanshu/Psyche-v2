# =============================================================================
# Psyche v2 — Dockerfile (Senior Review Fixes)
# =============================================================================

FROM python:3.11-slim

# --- Metadata ---
LABEL maintainer="Psyche Bot"
LABEL description="Psyche: High-Reasoning AI Psychologist"

# --- Environment Configuration ---
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    SSL_CERT_DIR=/etc/ssl/certs \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

WORKDIR /app

# --- System Dependencies ---
RUN apt-get update && \
    apt-get install -y --reinstall ca-certificates && \
    update-ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# --- User & Permission Hardening ---
# Hugging Face Spaces run as user 1000. 
# We create the /data volume, chown it to 1000, and switch to that user
# to prevent "Permission Denied" or "Read-Only File System" errors.
RUN mkdir -p /data && \
    useradd -m -u 1000 user && \
    chown -R 1000:1000 /app /data

USER user

# --- Dependencies ---
COPY --chown=user:user requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# --- Application ---
COPY --chown=user:user bot.py .

# HF Spaces mandatory port
EXPOSE 7860

# Launch
CMD ["python", "bot.py"]
