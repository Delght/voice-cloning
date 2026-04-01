# Voice Cloning & AI Assistant

A fully self-hosted, privacy-first voice cloning and conversational AI system. No cloud APIs — all inference runs on your own hardware.

## What it does

- **Zero-shot voice cloning (best-effort)** from a short reference clip (quality varies by language and recording)
- **Speak Vietnamese naturally** via a dedicated Vietnamese TTS model
- **Converse with an AI** — your voice in, AI's response out (voice similarity depends on the selected TTS/VC setup)
- **High-similarity voice cloning via Voice Conversion (RVC)** using a trained `.pth` model (trained separately)

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

Notes:
- The system can run in TTS-only mode (no RVC).
- RVC requires a trained voice model (`.pth`, optional `.index`). Without it, you can still use TTS, but it won't match a specific target speaker reliably.

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
make run_tts          # :8002 (VieNeu, Vietnamese TTS, default)
make run_llm          # :8004 (LLM connector → Anything-LLM)
make run_gateway      # :8000 (proxy + orchestrator)
```

fish-speech runs in a separate conda environment due to dependency conflicts:

```bash
conda activate voice_fish
make CONDA_ENV=voice_fish run_tts_fish   # :8002 (fish-speech)
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
    -F "text=Xin chào!" \
    -F "ref_audio=@data/raw/reference.wav" \
    -F "ref_text=Exact transcript of the reference audio" \
    -o output.wav
```

> **VieNeu cloning tip:** `ref_text` must exactly match what is spoken in the reference audio. Without it, cloning quality drops significantly. VieNeu uses **standard mode** (not turbo) for proper text-speech alignment.

### Web UI (Gradio)

```bash
make run_ui    # http://localhost:7860
```

4 tabs: Voice Chat, Text-to-Speech, Voice Conversion, Transcribe.

### Development

```bash
make check    # format + lint (ruff)
```

## Privacy

All inference runs locally. No data is sent to any external service.
