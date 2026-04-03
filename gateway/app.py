"""API Gateway — reverse proxy for Voice microservices.

Run:
    uvicorn gateway.app:app --port 8000

Endpoints:
    POST /chat              -> orchestrate STT → LLM → TTS pipeline
    POST /transcribe        -> STT :8001
    POST /tts/fish-speech   -> TTS :8002
    POST /tts/vieneu        -> TTS :8002
    POST /convert-voice     -> RVC :8003
    POST /llm/chat          -> LLM :8004
    GET  /health            -> aggregate health from all services
    GET  /health/{service}  -> health check a single service
"""

import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from .config import REQUEST_TIMEOUT, ROUTE_MAP, SERVICES
from .orchestrator import PipelineError, run_chat_pipeline
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
    title="Voice — API Gateway",
    description="Reverse proxy and orchestrator for STT, TTS, RVC, and LLM microservices",
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
    """Voice conversation pipeline: audio → STT → LLM → TTS → audio."""
    audio_bytes = await audio.read()
    if not audio_bytes:
        return JSONResponse(status_code=400, content={"error": "Empty audio file"})

    log.info("/chat: received audio '%s' (%d bytes)", audio.filename, len(audio_bytes))

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


@app.post("/tts/fish-speech")
async def proxy_tts_fish(request: Request):
    return await _proxy(request, ROUTE_MAP["/tts/fish-speech"])


@app.post("/tts/vieneu")
async def proxy_tts_vieneu(request: Request):
    return await _proxy(request, ROUTE_MAP["/tts/vieneu"])


@app.post("/convert-voice")
async def proxy_convert_voice(request: Request):
    return await _proxy(request, ROUTE_MAP["/convert-voice"])


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
    """Check health of a specific service by name (stt, tts, rvc, llm)."""
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
