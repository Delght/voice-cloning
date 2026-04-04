# Fiona Anne

Self-hosted voice cloning and conversational AI. Zero external APIs.

Zero shot voice cloning: Clone any voice from a 10 to 30s audio sample. No training, no fine-tuning required.

## Demo

**Assistant:** **Google Gemma 4** via [Ollama](https://ollama.com) + [Anything-LLM](https://github.com/Mintplex-Labs/anything-llm).

![Demo UI](docs/images/demo.png)

**Voice cloning:** short reference clip -> speech in that timbre.

<video src="https://github.com/user-attachments/assets/350ab4ae-8efc-4d35-9ed3-310042c8ef68" controls width="100%"></video>

### Bundled timbre

**Phương Anh** (my special friend; personal recording for this repo, with permission).

Your own clip: paths and transcript in `.env` (see [`.env.example`](.env.example)).

## Stack

| Layer | Technology |
| --- | --- |
| STT | [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) |
| TTS | [fishaudio/fish-speech](https://github.com/fishaudio/fish-speech) + [pnnbao97/VieNeu-TTS](https://github.com/pnnbao97/VieNeu-TTS) |
| Voice Conversion | [IAHispano/Applio](https://github.com/IAHispano/Applio) (RVC) |
| LLM | [Gemma 4](https://ai.google.dev/gemma) via [Ollama](https://ollama.com) + [Anything-LLM](https://github.com/Mintplex-Labs/anything-llm) |
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
| Voice Chat (Gradio) | `/transcribe` + `/llm/chat` + `/tts/*` | Same stages as `/chat`, split for progress UI; fish ref: bundled default, upload, or env (see `.env.example`) |

## Requirements

- [Miniforge](https://github.com/conda-forge/miniforge) + `ffmpeg`
- [Ollama](https://ollama.com) (`ollama pull gemma4` for Gemma 4) + [Anything-LLM](https://anythingllm.com) (workspace uses that model)

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

`POST /chat` resolves fish reference like Voice Chat: bundled default unless `CHAT_FISH_REF_AUDIO` / `VOICE_CHAT_FISH_REF_AUDIO` is set. Set the matching `*_REF_TEXT` in `.env` for best quality (see `.env.example`).

### API

```bash
curl http://localhost:8000/health

# Needs fish TTS + default ref (repo or CHAT_* / VOICE_* in .env) + matching ref_text
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

**Colab:** open the badge notebook, enable a GPU runtime, run cells top to bottom. Same idea as `make run_all` (cell 2) then `make run_ui` (cell 3; set `GRADIO_SHARE` for a public Gradio URL). One pip env, not conda.

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
