# TTS Service — VieNeu-TTS (Vietnamese-optimized, default engine)
# For fish-speech, see Dockerfile.fish
#
# Build arg TORCH_INDEX lets you switch torch variant without changing this file:
#   CPU (default): --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu
#   CUDA 12.1:     --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu121
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ARG TORCH_INDEX
RUN pip install --no-cache-dir \
    torch torchaudio --index-url ${TORCH_INDEX}

# vieneu pulls in neucodec and llama-cpp-python automatically
RUN pip install --no-cache-dir \
    vieneu \
    neucodec \
    soundfile \
    numpy \
    fastapi \
    "uvicorn[standard]" \
    pydantic \
    python-multipart

COPY services/ ./services/

EXPOSE 8002

ENV TTS_ENGINES=vieneu

CMD ["uvicorn", "services.tts.app:app", "--host", "0.0.0.0", "--port", "8002"]
