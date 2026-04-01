"""STT Microservice — Speech-to-Text via faster-whisper.

Run:
    uvicorn services.stt.app:app --port 8001
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

from .engine import STTEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

engine = STTEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine.load()
    yield
    engine.unload()


app = FastAPI(
    title="Voice — STT Service",
    description="Speech-to-Text transcription via faster-whisper",
    version="0.1.0",
    lifespan=lifespan,
)


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request, exc):
    msg = str(exc).lower()
    if "out of memory" in msg or "oom" in msg:
        return JSONResponse(
            status_code=503,
            content={"error": "Model out of memory", "detail": str(exc)},
        )
    return JSONResponse(
        status_code=500,
        content={"error": "Internal error", "detail": str(exc)},
    )


@app.get("/health")
async def health():
    return {"status": "ok" if engine.ready else "loading", "model": engine.model_size}


@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(..., description="Audio file (wav, mp3, flac, etc.)"),
    model: str = Form("large-v3", description="Whisper model size"),
    beam_size: int = Form(5, description="Beam search width"),
    language: str | None = Form(
        None,
        description=("Optional language hint (e.g. 'vi', 'en'). If omitted, Whisper auto-detects."),
    ),
):
    """Transcribe an audio file to text.

    Accepts any audio format supported by soundfile/librosa.
    Returns JSON with full text, detected language, and per-segment timestamps.
    """
    if model != engine.model_size:
        log.warning(
            "Requested model '%s' but loaded model is '%s'. Using loaded model.",
            model,
            engine.model_size,
        )

    audio_bytes = await audio.read()
    if not audio_bytes:
        return JSONResponse(status_code=400, content={"error": "Empty audio file"})

    log.info("Transcribing %s (%d bytes)", audio.filename, len(audio_bytes))
    result = engine.transcribe(audio_bytes, beam_size=beam_size, language=language)
    log.info("Result: lang=%s, text='%s'", result["language"], result["text"][:80])

    return result
