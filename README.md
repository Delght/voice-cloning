# Voice Cloning & AI Assistant

Self-hosted voice cloning and conversational AI. Zero external APIs.

Zero shot voice cloning: Clone any voice from a 10 to 30s audio sample. No training, no fine-tuning required.

## Demo

![Demo UI](docs/images/demo.png)

<video src="https://github.com/user-attachments/assets/8704a7d8-fe15-47e7-85ce-a0235b8c3809" controls width="100%"></video>


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
└───────────────────┬──────────────────────────┘
                    │ HTTP
                    ▼
┌──────────────────────────────────────────────┐
│           API Gateway (:8000)                │
│  /transcribe  /tts/*  /convert-voice  /chat  │
│  /llm/chat                                   │
└──────┬──────────┬──────────┬───────────┬─────┘
       ▼          ▼          ▼           ▼
      STT        TTS         RVC        LLM
     :8001      :8002       :8003      :8004
```

### Pipelines

| Flow | Gateway | Notes |
| --- | --- | --- |
| Transcribe | `/transcribe` | `Audio → STT → text` |
| Voice cloning | `/tts/vieneu`, `/tts/fish-speech` | `Text + ref audio + ref_text → TTS → WAV` |
| Voice conversion | `/convert-voice` | `WAV + RVC .pth → RVC → converted WAV` |
| Conversation (one shot) | `/chat` | `Mic → STT → LLM → TTS → WAV`. E.g. `make chat_sample`, `api_client.chat()` |
| LLM only | `/llm/chat` | JSON `{"message": "..."}` → assistant text (proxied to `:8004`) |
| Voice Chat (Gradio) | `/transcribe` + `/llm/chat` + `/tts/*` | Same stages as `/chat`, split for progress UI; fish-speech needs a ref WAV (upload, or `audio/output/morgan_freeman.wav`, or env) |

## Requirements

- [Miniforge](https://github.com/conda-forge/miniforge) + `ffmpeg`
- [Ollama](https://ollama.com) + [Anything-LLM](https://anythingllm.com) (for the LLM connector)

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

> VieNeu: `ref_text` must match the reference audio exactly. Omitting it degrades cloning quality significantly.

### Scripts

```bash
# Convert mp3/m4a → wav (batch, before using as reference audio)
python scripts/convert_audio.py --input audio/input --output audio/output
# or via make:
make convert_audio

python scripts/chunk_audio.py --input data/raw/YOUR_FILE.mp3
python scripts/tts_infer.py --text "Hello" --ref data/chunks/speech_chunk_0001.wav --ref-text "..."
python scripts/vieneu_infer.py --text "Xin chào!"
python scripts/rvc_infer.py --input data/output.wav --model models/rvc/target.pth
```

> RVC requires a trained `.pth` model. Create `models/rvc/` and add your model before running.

### Dev

```bash
make check    # ruff format + lint
```
## Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ngduythao/voice-clone/blob/main/Voice_Colab.ipynb)

To run this project on Google Colab with GPU support, click the badge above to open the provided notebook. Run the cells sequentially to start all services and obtain the public Gradio link for the UI.

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
