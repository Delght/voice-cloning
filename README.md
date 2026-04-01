# Voice Cloning & AI Assistant

A fully self-hosted, privacy-first voice cloning and conversational AI system. No cloud APIs — all inference runs on your own hardware.

## What it does

- **Clone any voice** from a short audio reference (zero-shot)
- **Speak Vietnamese naturally** via a dedicated Vietnamese TTS model
- **Converse with an AI** — your voice in, AI's response out, in your cloned voice
- **Convert voice timbre** using RVC to match a target speaker

## Tech Stack

| Layer | Technology |
| --- | --- |
| Speech-to-Text | [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) |
| Text-to-Speech | [fishaudio/fish-speech](https://github.com/fishaudio/fish-speech) + [pnnbao97/VieNeu-TTS](https://github.com/pnnbao97/VieNeu-TTS) |
| Voice Conversion | [IAHispano/Applio](https://github.com/IAHispano/Applio) (RVC) |
| LLM | [QwenLM/Qwen3.5](https://github.com/QwenLM/Qwen3.5) via [Mintplex-Labs/anything-llm](https://github.com/Mintplex-Labs/anything-llm) |
| Backend | [FastAPI](https://github.com/fastapi/fastapi) + [Pydantic](https://github.com/pydantic/pydantic) |
| Frontend | [Gradio](https://github.com/gradio-app/gradio) |

## Architecture

```
Mic → [STT] → text → [LLM] → response → [TTS] → audio → [RVC] → cloned voice → Speaker
```

## Requirements

- Python 3.11+ with [Miniforge](https://github.com/conda-forge/miniforge) (recommended)
- `ffmpeg` (`brew install ffmpeg` on macOS)
- [Ollama](https://ollama.com) + [Anything-LLM](https://anythingllm.com) for the LLM brain
- Apple Silicon (MPS), Nvidia GPU (CUDA), or CPU — auto-detected

## Quick Start

```bash
conda activate voice
make help
```

### Scripts (CLI)

```bash
# Transcribe audio
python scripts/chunk_audio.py --input data/raw/YOUR_FILE.mp3

# TTS with voice cloning (English)
python scripts/tts_infer.py \
    --text "Hello world" \
    --ref data/chunks/speech_chunk_0001.wav \
    --ref-text "Transcript of the reference audio"

# TTS (Vietnamese)
python scripts/vieneu_infer.py --text "Xin chào!"

# Voice conversion
python scripts/rvc_infer.py --input data/output.wav --model models/rvc/target.pth
```

### Microservices (REST API)

Start each service in its own terminal:

```bash
make run_stt          # :8001
make run_tts          # :8002 (Vietnamese TTS, default)
make run_llm          # :8004 (LLM connector → Anything-LLM)
make run_gateway      # :8000 (proxy + orchestrator)
```

```bash
curl http://localhost:8000/health

# Full conversation pipeline (audio → STT → LLM → TTS → audio)
curl -X POST http://localhost:8000/chat \
    -F "audio=@data/chunks/speech_chunk_0001.wav" -o response.wav

# Individual services
curl -X POST http://localhost:8000/transcribe \
    -F "audio=@data/chunks/speech_chunk_0001.wav"
curl -X POST http://localhost:8000/tts/vieneu \
    -F "text=Xin chào!" -o output.wav
```

### Development

```bash
make check    # format + lint (ruff)
```

## Privacy

All inference runs locally. No data is sent to any external service.
