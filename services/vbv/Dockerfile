FROM python:3.10-slim

WORKDIR /app

# git is required for pip to clone vibevoice from GitHub
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install standard Python deps
COPY services/vbv/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install vibevoice from GitHub source (not on PyPI).
# [streamingtts] extra pins transformers==4.51.3 for streaming inference.
# Same pattern as fish-speech / applio: installed separately from requirements.txt.
RUN pip install --no-cache-dir "vibevoice[streamingtts] @ git+https://github.com/microsoft/VibeVoice.git"

# Create models directory (mounted at runtime via docker-compose volumes)
RUN mkdir -p /app/models

# Copy only the services layer — consistent with other service Dockerfiles
COPY services/ ./services/

EXPOSE 8005

CMD ["uvicorn", "services.vbv.main:app", "--host", "0.0.0.0", "--port", "8005"]
