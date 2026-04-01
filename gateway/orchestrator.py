"""Orchestrator — STT → LLM → TTS pipeline for POST /chat.

Pipeline:
    1. POST :8001/transcribe  (STT)  audio bytes  → user text
    2. POST :8004/chat        (LLM)  user text    → response text
    3. POST :8002/tts/vieneu  (TTS)  response text → WAV bytes
"""

import logging

import httpx

from .config import LLM_URL, STT_URL, TTS_URL

log = logging.getLogger(__name__)


class PipelineError(Exception):
    """Raised when a pipeline stage fails."""

    def __init__(self, stage: str, detail: str, status_code: int = 502) -> None:
        super().__init__(f"[{stage}] {detail}")
        self.stage = stage
        self.detail = detail
        self.status_code = status_code


async def run_chat_pipeline(audio_bytes: bytes, client: httpx.AsyncClient) -> bytes:
    """Run the full STT → LLM → TTS pipeline.

    Args:
        audio_bytes: Raw audio from the user.
        client: Shared httpx.AsyncClient from the gateway lifespan.

    Returns:
        WAV bytes of the LLM's spoken response.

    Raises:
        PipelineError: If any stage fails.
    """
    log.info("STT ← %d bytes", len(audio_bytes))
    try:
        stt_resp = await client.post(
            f"{STT_URL}/transcribe",
            files={"audio": ("audio.wav", audio_bytes, "audio/wav")},
        )
    except httpx.ConnectError:
        raise PipelineError("STT", "Cannot connect to STT service (:8001).")
    except httpx.TimeoutException:
        raise PipelineError("STT", "STT service timed out.")

    if stt_resp.status_code != 200:
        raise PipelineError("STT", stt_resp.text[:300], stt_resp.status_code)

    user_text = stt_resp.json().get("text", "").strip()
    if not user_text:
        raise PipelineError("STT", "Empty transcript.", 422)

    log.info("STT → '%s'", user_text[:120])

    log.info("LLM ←")
    try:
        llm_resp = await client.post(f"{LLM_URL}/chat", json={"message": user_text})
    except httpx.ConnectError:
        raise PipelineError("LLM", "Cannot connect to LLM service (:8004).")
    except httpx.TimeoutException:
        raise PipelineError("LLM", "LLM service timed out.")

    if llm_resp.status_code != 200:
        raise PipelineError("LLM", llm_resp.text[:300], llm_resp.status_code)

    response_text = llm_resp.json().get("text", "").strip()
    if not response_text:
        raise PipelineError("LLM", "Empty response.", 502)

    log.info("LLM → '%s'", response_text[:120])

    log.info("TTS ←")
    try:
        tts_resp = await client.post(f"{TTS_URL}/tts/vieneu", data={"text": response_text})
    except httpx.ConnectError:
        raise PipelineError("TTS", "Cannot connect to TTS service (:8002).")
    except httpx.TimeoutException:
        raise PipelineError("TTS", "TTS service timed out.")

    if tts_resp.status_code != 200:
        raise PipelineError("TTS", tts_resp.text[:300], tts_resp.status_code)

    audio_out = tts_resp.content
    if not audio_out:
        raise PipelineError("TTS", "Empty audio response.", 502)

    log.info("TTS → %d bytes", len(audio_out))
    return audio_out
