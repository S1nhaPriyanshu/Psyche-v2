# =============================================================================
# Psyche v2 — Dockerfile
# Base: python:3.11-slim for a lightweight, fast build
# Target: Hugging Face Spaces (Docker SDK)
# =============================================================================

FROM python:3.11-slim

# --- System metadata ---
LABEL maintainer="Psyche Bot"
LABEL description="Psyche: Privacy-First AI Behavioral Psychologist"

# --- Set working directory inside the container ---
WORKDIR /app

# --- Create the persistent data directory ---
# Hugging Face mounts its Storage Bucket here.
# chmod 777 ensures the bot process can read/write the SQLite DB
# without permission errors even if the container user differs.
RUN mkdir -p /data && chmod 777 /data

# --- Install Python dependencies ---
# Copy requirements first to leverage Docker layer caching.
# The image won't re-install packages unless requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- Copy application source code ---
COPY bot.py .

# --- Expose the heartbeat port ---
# Hugging Face requires at least one port to be exposed for the Space
# to transition from "Building" to "Running" state.
EXPOSE 7860

# --- Launch the bot ---
CMD ["python", "bot.py"]
