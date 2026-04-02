# Voice Cloning & AI Assistant

Self-hosted voice cloning and conversational AI. Zero external APIs.

## Stack

| Layer | Technology |
| --- | --- |
| STT | [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) |
| TTS | [fishaudio/fish-speech](https://github.com/fishaudio/fish-speech) + [pnnbao97/VieNeu-TTS](https://github.com/pnnbao97/VieNeu-TTS) |
| Voice Conversion | [IAHispano/Applio](https://github.com/IAHispano/Applio) (RVC) |
| LLM | [QwenLM/Qwen3.5](https://github.com/QwenLM/Qwen3.5) via [Mintplex-Labs/anything-llm](https://github.com/Mintplex-Labs/anything-llm) |
| Backend | [FastAPI](https://github.com/fastapi/fastapi) + [Pydantic](https://github.com/pydantic/pydantic) |
| Frontend | [Gradio](https://github.com/gradio-app/gradio) |

## Architecture

```
            User Browser / Mic
                    │ :7860
                    ▼
┌──────────────────────────────────────────────┐
│              Gradio UI (:7860)               │
└────────────────── ┬──────────────────────────┘
                    │ HTTP
                    ▼
┌──────────────────────────────────────────────┐
│           API Gateway (:8000)                │
│   /chat  /transcribe  /tts/*  /convert-voice │
└────┬──────────┬──────────┬───────────┬───────┘
     ▼          ▼          ▼           ▼
    STT        TTS         RVC        LLM
   :8001      :8002       :8003      :8004
```

Conversation pipeline: `Mic → STT → LLM → TTS → (RVC optional) → Speaker`

## Requirements

- [Miniforge](https://github.com/conda-forge/miniforge) + `ffmpeg`
- [Ollama](https://ollama.com) + [Anything-LLM](https://anythingllm.com)

## Usage

```bash
conda activate voice
make help        # list all commands
make run_all     # start STT + TTS + LLM + Gateway
make run_ui      # Gradio at :7860
make health      # check all services
```

fish-speech requires a separate env due to numpy version conflicts:

```bash
make run_tts_fish   # uses voice_fish env automatically
```

### API

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/chat \
    -F "audio=@data/chunks/speech_chunk_0001.wav" -o response.wav

curl -X POST http://localhost:8000/transcribe \
    -F "audio=@data/chunks/speech_chunk_0001.wav"

curl -X POST http://localhost:8000/tts/vieneu \
    -F "text=Xin chào!" \
    -F "ref_audio=@data/raw/reference.wav" \
    -F "ref_text=Exact transcript of the reference audio" \
    -o output.wav
```

> VieNeu: `ref_text` must match the reference audio exactly — omitting it degrades cloning quality significantly.

### Scripts

```bash
python scripts/chunk_audio.py --input data/raw/YOUR_FILE.mp3
python scripts/tts_infer.py --text "Hello" --ref data/chunks/speech_chunk_0001.wav --ref-text "..."
python scripts/vieneu_infer.py --text "Xin chào!"
python scripts/rvc_infer.py --input data/output.wav --model models/rvc/target.pth
```

### Dev

```bash
make check    # ruff format + lint
```

## Docker

```bash
cp .env.example .env   # fill in ANYTHING_LLM_API_KEY
docker compose up --build

# CUDA (cloud)
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d
```

```bash
docker compose ps
docker compose logs -f gateway
docker compose restart tts
docker compose down
```
