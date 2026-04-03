"""LLM Microservice: connector to Anything-LLM workspace chat API.

Run:
    uvicorn services.llm.app:app --port 8004

Endpoints:
    POST /chat   {"message": "..."} → {"text": "..."}
    GET  /health → connectivity status to Anything-LLM
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .engine import LLMEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

engine = LLMEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine.load()
    yield
    await engine.unload()


app = FastAPI(
    title="Voice - LLM Service",
    description="HTTP connector to Anything-LLM workspace chat API",
    version="0.1.0",
    lifespan=lifespan,
)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    text: str


@app.get("/health")
async def health():
    return await engine.health_check()


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.message.strip():
        return JSONResponse(status_code=400, content={"error": "Empty message"})

    try:
        response_text = await engine.chat(req.message)
    except RuntimeError as e:
        log.error("LLM error: %s", e)
        return JSONResponse(status_code=503, content={"error": "LLM unavailable", "detail": str(e)})

    return ChatResponse(text=response_text)
