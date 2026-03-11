FROM python:3.11-slim

# ── System deps: FFmpeg + yt-dlp ──────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install yt-dlp (latest stable — pinned via pip for reproducibility)
RUN pip install --no-cache-dir yt-dlp

# ── Python deps ───────────────────────────────────────────────────────────────
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── App ───────────────────────────────────────────────────────────────────────
COPY main.py .

# Temp dir for ephemeral audio files
RUN mkdir -p /tmp/se_audio

EXPOSE 8000

# ── Start ─────────────────────────────────────────────────────────────────────
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
