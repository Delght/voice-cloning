"""TTS Microservice — Text-to-Speech via fish-speech and VieNeu-TTS.

Hosts two TTS engines behind separate endpoints:
    POST /tts/fish-speech  — multilingual, zero-shot voice cloning
    POST /tts/vieneu       — Vietnamese-optimized, optional voice cloning

Run:
    uvicorn services.tts.app:app --port 8002
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Engine selection via environment variable.
# TTS_ENGINES=vieneu      (default — lightweight, Vietnamese-optimized)
# TTS_ENGINES=fish        (only fish-speech — multilingual, heavier)
# TTS_ENGINES=fish,vieneu (both — uses more RAM)
_enabled = os.environ.get("TTS_ENGINES", "vieneu").lower().split(",")
_enabled = [e.strip() for e in _enabled if e.strip()]

fish_engine = None
vieneu_engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global fish_engine, vieneu_engine

    if "fish" in _enabled:
        from .fish_engine import FishSpeechEngine

        fish_engine = FishSpeechEngine()
        fish_engine.load()

    if "vieneu" in _enabled:
        from .vieneu_engine import VieNeuEngine

        vieneu_engine = VieNeuEngine()
        vieneu_engine.load()

    yield

    if fish_engine is not None:
        fish_engine.unload()
    if vieneu_engine is not None:
        vieneu_engine.unload()


app = FastAPI(
    title="Voice — TTS Service",
    description="Text-to-Speech with fish-speech (multilingual) and VieNeu-TTS (Vietnamese)",
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
        content={"error": "Inference error", "detail": str(exc)},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    log.exception("Unhandled TTS error")
    return JSONResponse(
        status_code=500,
        content={"error": type(exc).__name__, "detail": str(exc)},
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "engines": {
            "fish_speech": fish_engine.ready if fish_engine else "disabled",
            "vieneu": vieneu_engine.ready if vieneu_engine else "disabled",
        },
    }


@app.post("/tts/fish-speech")
async def tts_fish_speech(
    text: str = Form(..., description="Text to synthesize"),
    ref_audio: UploadFile = File(..., description="Reference audio for voice cloning (10-30s)"),
    ref_text: str = Form("", description="Transcript of reference audio (critical for quality)"),
    temperature: float = Form(0.7),
    top_p: float = Form(0.7),
    repetition_penalty: float = Form(1.2),
    chunk_length: int = Form(200),
):
    """Synthesize speech using fish-speech with zero-shot voice cloning."""
    if fish_engine is None or not fish_engine.ready:
        return JSONResponse(
            status_code=503,
            content={"error": "fish-speech engine not available. Start with TTS_ENGINES=fish"},
        )

    if not ref_text:
        log.warning("No ref_text provided — cloning quality will be degraded.")

    ref_bytes = await ref_audio.read()
    if not ref_bytes:
        return JSONResponse(status_code=400, content={"error": "Empty reference audio file"})

    wav_bytes, sr = fish_engine.synthesize(
        text=text,
        ref_audio_bytes=ref_bytes,
        ref_text=ref_text,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        chunk_length=chunk_length,
    )

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=output.wav"},
    )


@app.post("/tts/vieneu")
async def tts_vieneu(
    text: str = Form(..., description="Text to synthesize (Vietnamese, English, or mixed)"),
    ref_audio: UploadFile | None = File(
        None, description="Optional reference audio for voice cloning (3-5s)"
    ),
    ref_text: str = Form(
        "",
        description=(
            "Optional transcript of reference audio "
            "(improves cloning if supported by the VieNeu SDK)"
        ),
    ),
    temperature: float = Form(1.0),
    top_k: int = Form(50),
    max_chars: int = Form(256, description="Max characters per chunk (reduces truncation)"),
):
    """Synthesize Vietnamese speech using VieNeu-TTS with optional voice cloning."""
    if vieneu_engine is None or not vieneu_engine.ready:
        return JSONResponse(
            status_code=503,
            content={"error": "VieNeu-TTS engine not available. Start with TTS_ENGINES=vieneu"},
        )

    ref_bytes = None
    ref_filename = None
    if ref_audio is not None:
        ref_filename = ref_audio.filename
        ref_bytes = await ref_audio.read()
        if not ref_bytes:
            ref_bytes = None

    wav_bytes, sr = vieneu_engine.synthesize(
        text=text,
        ref_audio_bytes=ref_bytes,
        ref_audio_filename=ref_filename,
        ref_text=ref_text.strip() or None,
        temperature=temperature,
        top_k=top_k,
        max_chars=max_chars,
    )

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=output_vi.wav"},
    )
