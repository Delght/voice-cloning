# Voice Cloning & AI Assistant

A fully self-hosted, privacy-first voice cloning and conversational AI system built on Apple Silicon. No cloud APIs — everything runs locally on your machine.

## What it does

- **Clone any voice** from a short audio reference (zero-shot)
- **Speak Vietnamese naturally** via a dedicated Vietnamese TTS model
- **Converse with an AI** — your voice in, AI's response out, in your cloned voice
- **Convert voice timbre** using RVC to match a target speaker model
- **Flexible Inputs:** Supports both uploading existing audio files and real-time microphone recording

## Tech Stack

| Layer | Technology |
| --- | --- |
| Speech-to-Text | faster-whisper |
| Text-to-Speech | fish-speech + VieNeu-TTS |
| Voice Conversion | Applio (RVC) |
| LLM Brain | Qwen3.5 via Anything-LLM |
| Backend | FastAPI + Pydantic |
| Frontend | Gradio |
| Hardware | Apple M4 Pro · 24GB · MPS backend |

## Architecture

```
User → Microphone
         ↓
   [STT: faster-whisper]  →  transcript
         ↓
   [LLM: Qwen3.5]         →  text response
         ↓
   [TTS: fish-speech]      →  audio
         ↓
   [RVC: Applio] (optional) → cloned voice
         ↓
      Speaker ← User
```

## Status

> Work in progress — building phase by phase.

- [x] Project structure & documentation
- [ ] Phase 1: Audio processing & STT pipeline
- [ ] Phase 2: TTS & voice cloning
- [ ] Phase 3: Microservices & API Gateway
- [ ] Phase 4: LLM integration
- [ ] Phase 5: UI
- [ ] Phase 6: Optimization & Docker

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- [Miniforge](https://github.com/conda-forge/miniforge) (native ARM64 Python)
- `brew install ffmpeg`

## Getting Started

*Installation instructions and setup scripts are coming soon as the phases are completed.*

## Privacy

All inference runs locally. No data is sent to OpenAI, ElevenLabs, or any external service.
