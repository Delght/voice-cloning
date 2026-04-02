FROM python:3.11-slim

# System dependencies for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
# faster-whisper uses ctranslate2 (C++) — no torch required for STT
RUN pip install --no-cache-dir \
    faster-whisper \
    librosa \
    pydub \
    soundfile \
    numpy \
    fastapi \
    "uvicorn[standard]" \
    pydantic \
    python-multipart

# Copy project code (models/data excluded via .dockerignore)
COPY services/ ./services/
COPY gateway/ ./gateway/

EXPOSE 8001

CMD ["uvicorn", "services.stt.app:app", "--host", "0.0.0.0", "--port", "8001"]
