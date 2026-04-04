# Voice Cloning & AI Assistant

Self-hosted voice cloning and conversational AI. Zero external APIs.

Zero shot voice cloning: Clone any voice from a 10 to 30s audio sample. No training, no fine-tuning required.

## Demo

**AI assistant:** LLM is **Gemma**

![Demo UI](docs/images/demo.png)

**Voice cloning:** (reference audio -> cloned speech).

<video src="https://github.com/user-attachments/assets/350ab4ae-8efc-4d35-9ed3-310042c8ef68" controls width="100%"></video>

## Stack

| Layer | Technology |
| --- | --- |
| STT | [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) |
| TTS | [fishaudio/fish-speech](https://github.com/fishaudio/fish-speech) + [pnnbao97/VieNeu-TTS](https://github.com/pnnbao97/VieNeu-TTS) |
| Voice Conversion | [IAHispano/Applio](https://github.com/IAHispano/Applio) (RVC) |
| LLM | [Google Gemma](https://ai.google.dev/gemma) (e.g. Gemma 4 via [Ollama](https://ollama.com)) + [Anything-LLM](https://github.com/Mintplex-Labs/anything-llm) |
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
| Transcribe | `/transcribe` | `Audio -> STT -> text` |
| Voice cloning | `/tts/vieneu`, `/tts/fish-speech` | `Text + ref audio + ref_text -> TTS -> WAV` |
| Voice conversion | `/convert-voice` | `WAV + RVC .pth -> RVC -> converted WAV` |
| Conversation (one shot) | `/chat` | `Mic -> STT -> LLM -> TTS -> WAV`. E.g. `make chat_sample`, `api_client.chat()` |
| LLM only | `/llm/chat` | JSON `{"message": "..."}` -> assistant text (proxied to `:8004`) |
| Voice Chat (Gradio) | `/transcribe` + `/llm/chat` + `/tts/*` | Same stages as `/chat`, split for progress UI; fish ref defaults to tracked `audio/output/morgan_freeman.wav`, or upload / env override |

## Requirements

- [Miniforge](https://github.com/conda-forge/miniforge) + `ffmpeg`
- [Ollama](https://ollama.com) (pull your model, e.g. `ollama pull gemma4`) + [Anything-LLM](https://anythingllm.com) (workspace uses that model)

## Usage

```bash
conda activate voice
make help        # list all commands
make run_all     # start STT + TTS + LLM + Gateway
make run_ui      # Gradio at :7860
make health      # check all services
```

Default TTS is **fish-speech** (`make run_tts` uses conda env `voice_fish`). VieNeu stays available:

```bash
make run_tts_vieneu   # VieNeu only (conda env voice)
```

`POST /chat` uses the same fish ref rule: default file `audio/output/morgan_freeman.wav` (versioned in git), or `CHAT_FISH_REF_AUDIO` / `VOICE_CHAT_FISH_REF_AUDIO`.

### API

```bash
curl http://localhost:8000/health

# Needs fish TTS running + default ref audio/output/morgan_freeman.wav (or CHAT_* / VOICE_* env)
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
# Convert mp3/m4a -> wav (batch, before using as reference audio)
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

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/singswap/voice-cloning/blob/main/Voice_Colab.ipynb)

To run on Google Colab, open the notebook from the badge. It matches **local behavior**: cell 2 starts the same stack as **`make run_all`**, cell 3 the UI like **`make run_ui`** (with `GRADIO_SHARE` for a public Gradio link). Colab uses one pip env instead of conda. Run cells in order after choosing a GPU runtime.

## Docker

Compose uses the **VieNeu** TTS image (`TTS_ENGINES=vieneu` on the `tts` service). Default **fish-speech** applies to local `make run_tts` / Colab / bare `uvicorn` when you do not use that container.

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
