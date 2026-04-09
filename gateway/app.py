"""API Gateway: reverse proxy for Voice microservices.

Run:
    uvicorn gateway.app:app --port 8000

Endpoints:
    POST /chat              -> STT + LLM + TTS (orchestrator)
    POST /transcribe        -> STT :8001
    POST /tts/fish-speech   -> TTS :8002
    POST /tts/vieneu        -> TTS :8002
    POST /tts/vbv           -> VBV :8005
    POST /convert-voice     -> RVC :8003 (hybrid VBV->RVC or direct)
    POST /llm/chat          -> LLM :8004
    GET  /health            -> all services
    GET  /health/{service}  -> one service
"""

import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from .config import REQUEST_TIMEOUT, ROUTE_MAP, SERVICES
from .orchestrator import PipelineError, apply_rvc_conversion, run_chat_pipeline
from .schemas import GatewayHealth, HealthStatus

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_client: httpx.AsyncClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _client
    _client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT)
    log.info("Gateway ready. Timeout: %.0fs", REQUEST_TIMEOUT)
    yield
    await _client.aclose()
    log.info("Gateway shut down.")


app = FastAPI(
    title="Voice - API Gateway",
    description="Reverse proxy and orchestrator for STT, TTS, VBV, and LLM microservices",
    version="0.1.0",
    lifespan=lifespan,
)


async def _proxy(request: Request, upstream_url: str) -> Response:
    """Forward the incoming request to an upstream service and relay the response."""
    content_type = request.headers.get("content-type", "")

    try:
        body = await request.body()

        resp = await _client.request(
            method=request.method,
            url=upstream_url,
            content=body,
            headers={
                "content-type": content_type,
            },
        )

        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type"),
            headers={
                k: v for k, v in resp.headers.items() if k.lower() in ("content-disposition",)
            },
        )

    except httpx.ConnectError:
        log.error("Cannot connect to upstream: %s", upstream_url)
        return JSONResponse(
            status_code=502,
            content={
                "error": f"Service unavailable: {upstream_url}",
                "detail": "Connection refused. Is the service running?",
            },
        )
    except httpx.TimeoutException:
        log.error("Timeout from upstream: %s", upstream_url)
        return JSONResponse(
            status_code=504,
            content={
                "error": "Gateway timeout",
                "detail": f"Upstream {upstream_url} did not respond in time.",
            },
        )


# --- Orchestrator route ---


@app.post("/chat")
async def chat(
    audio: UploadFile = File(..., description="User voice audio (wav, mp3, flac, etc.)"),
):
    """Voice conversation pipeline: audio -> STT -> LLM -> TTS -> audio."""
    audio_bytes = await audio.read()
    if not audio_bytes:
        return JSONResponse(status_code=400, content={"error": "Empty audio file"})

    try:
        response_wav = await run_chat_pipeline(audio_bytes, _client)
    except PipelineError as e:
        log.error("/chat pipeline error at [%s]: %s", e.stage, e.detail)
        return JSONResponse(
            status_code=e.status_code,
            content={"error": f"Pipeline failed at [{e.stage}]", "detail": e.detail},
        )

    return Response(
        content=response_wav,
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=response.wav"},
    )


# --- Proxy routes ---


@app.post("/transcribe")
async def proxy_transcribe(request: Request):
    return await _proxy(request, ROUTE_MAP["/transcribe"])


@app.post("/tts/vieneu")
async def proxy_tts_vieneu(request: Request):
    return await _proxy(request, ROUTE_MAP["/tts/vieneu"])


@app.post("/tts/fish-speech")
async def proxy_tts_fish(request: Request):
    return await _proxy(request, ROUTE_MAP["/tts/fish-speech"])


@app.post("/tts/vbv")
async def proxy_tts_vbv(request: Request):
    return await _proxy(request, ROUTE_MAP["/tts/vbv"])


@app.post("/convert-voice")
async def convert_voice_endpoint(
    audio: UploadFile = File(None, description="Input audio file to convert"),
    text: str = Form(None, description="Text to synthesize before RVC (VBV)"),
    voice_model: str = Form(..., description="Voice model name (e.g. 'target')"),
    index_path: str = Form("", description="Path to .index file (optional)"),
    pitch: int = Form(0),
    f0_method: str = Form("rmvpe"),
    index_rate: float = Form(0.75),
    protect: float = Form(0.33),
    clean_audio: bool = Form(False),
):
    """Voice conversion pipeline: audio -> RVC or text -> VBV -> RVC."""
    if audio is not None:
        audio_bytes = await audio.read()
    elif text and text.strip():
        try:
            tts_resp = await _client.post(
                ROUTE_MAP["/tts/vbv"], data={"text": text}, timeout=REQUEST_TIMEOUT
            )
            if tts_resp.status_code != 200:
                return JSONResponse(
                    status_code=tts_resp.status_code,
                    content={"error": "VBV TTS failed", "detail": tts_resp.text},
                )
            audio_bytes = tts_resp.content
        except Exception as e:
            return JSONResponse(status_code=502, content={"error": str(e)})
    else:
        return JSONResponse(status_code=400, content={"error": "Must provide either audio or text"})

    try:
        converted_wav = await apply_rvc_conversion(
            audio_bytes=audio_bytes,
            voice_model=voice_model,
            index_path=index_path,
            pitch=pitch,
            f0_method=f0_method,
            index_rate=index_rate,
            protect=protect,
            clean_audio=clean_audio,
            client=_client,
        )
    except PipelineError as e:
        log.error("/convert-voice pipeline error at [%s]: %s", e.stage, e.detail)
        return JSONResponse(
            status_code=e.status_code,
            content={"error": f"Pipeline failed at [{e.stage}]", "detail": e.detail},
        )

    return Response(
        content=converted_wav,
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=converted.wav"},
    )


@app.post("/llm/chat")
async def proxy_llm_chat(request: Request):
    return await _proxy(request, ROUTE_MAP["/llm/chat"])


# --- Health checks ---


async def _check_service(name: str, url: str) -> HealthStatus:
    """Ping a service's /health endpoint."""
    try:
        resp = await _client.get(f"{url}/health", timeout=5.0)
        return HealthStatus(
            service=name,
            url=url,
            status="ok" if resp.status_code == 200 else "degraded",
            detail=resp.json() if resp.status_code == 200 else resp.text,
        )
    except httpx.ConnectError:
        return HealthStatus(service=name, url=url, status="down", detail="Connection refused")
    except httpx.TimeoutException:
        return HealthStatus(
            service=name, url=url, status="timeout", detail="Health check timed out"
        )
    except Exception as e:
        return HealthStatus(service=name, url=url, status="error", detail=str(e))


@app.get("/health", response_model=GatewayHealth)
async def health_all():
    """Aggregate health status from all registered services."""
    statuses = []
    for name, url in SERVICES.items():
        statuses.append(await _check_service(name, url))
    return GatewayHealth(services=statuses)


@app.get("/health/{service}")
async def health_single(service: str):
    """Check health of a specific service by name (stt, tts, vbv, llm)."""
    if service not in SERVICES:
        return JSONResponse(
            status_code=404,
            content={
                "error": f"Unknown service '{service}'",
                "available": list(SERVICES.keys()),
            },
        )
    status = await _check_service(service, SERVICES[service])
    return status
