# RVC Service — Voice Conversion via Applio
# Models mounted as read-only volumes (not baked in).
#
# Build arg TORCH_INDEX:
#   CPU (default): https://download.pytorch.org/whl/cpu
#   CUDA 12.1:     https://download.pytorch.org/whl/cu121
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ARG TORCH_INDEX
RUN pip install --no-cache-dir \
    torch torchaudio --index-url ${TORCH_INDEX}

RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    pydantic \
    python-multipart \
    librosa \
    soundfile \
    numpy

COPY services/ ./services/

EXPOSE 8003

CMD ["uvicorn", "services.rvc.app:app", "--host", "0.0.0.0", "--port", "8003"]
