"""RVC Microservice: Voice Conversion via Applio.

Run:
    uvicorn services.rvc.app:app --port 8003
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response

from .engine import RVCEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

engine = RVCEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine.load()
    yield
    engine.unload()


app = FastAPI(
    title="Voice - RVC Service",
    description="Voice conversion via Applio RVC - swap voice identity while preserving prosody",
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
        content={"error": "Voice conversion error", "detail": str(exc)},
    )


@app.get("/health")
async def health():
    return {
        "status": "ok" if engine.ready else "loading",
        "available_models": engine.list_models(),
    }


@app.post("/convert-voice")
def convert_voice(
    audio: UploadFile = File(..., description="Input audio file to convert"),
    voice_model: str = Form(
        ..., description="Voice model name (e.g. 'target') or path to .pth file"
    ),
    index_path: str = Form("", description="Path to .index file (optional, improves quality)"),
    pitch: int = Form(0, description="Pitch shift in semitones (+12 = 1 octave up)"),
    f0_method: str = Form("rmvpe", description="F0 extraction: rmvpe, crepe, fcpe, swift"),
    index_rate: float = Form(0.75, description="Index matching rate (0.0-1.0)"),
    protect: float = Form(0.5, description="Consonant protection (0.0-1.0)"),
    clean_audio: bool = Form(False, description="Apply noise reduction before conversion"),
    clean_strength: float = Form(0.5, description="Noise reduction strength (0.0-1.0)"),
):
    """Convert the voice in an audio file to a target voice using a .pth model.

    The original prosody, emotion, and timing are preserved.
    Only the voice identity (timbre) changes.
    """
    audio_bytes = audio.file.read()
    if not audio_bytes:
        return JSONResponse(status_code=400, content={"error": "Empty audio file"})

    try:
        wav_bytes = engine.convert(
            audio_bytes=audio_bytes,
            model_name=voice_model,
            index_path=index_path,
            pitch=pitch,
            f0_method=f0_method,
            index_rate=index_rate,
            protect=protect,
            clean_audio=clean_audio,
            clean_strength=clean_strength,
        )
    except FileNotFoundError as e:
        return JSONResponse(status_code=404, content={"error": str(e)})

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=output_rvc.wav"},
    )
