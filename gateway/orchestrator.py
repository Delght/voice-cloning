"""Pipeline orchestration logic for /chat and /convert-voice endpoints."""

from __future__ import annotations

import re

import httpx

from .config import LLM_URL, RVC_URL, STT_URL, TTS_URL, VBV_URL


class PipelineError(Exception):
    def __init__(self, stage: str, detail: str, status_code: int = 502) -> None:
        super().__init__(f"[{stage}] {detail}")
        self.stage = stage
        self.detail = detail
        self.status_code = status_code


async def run_chat_pipeline(audio_bytes: bytes, client: httpx.AsyncClient) -> bytes:
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

    is_vietnamese = bool(
        re.search(
            r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]",
            response_text.lower(),
        )
    )

    if is_vietnamese:
        try:
            tts_resp = await client.post(
                f"{TTS_URL}/tts/vieneu",
                data={"text": response_text},
            )
        except httpx.ConnectError:
            raise PipelineError("TTS", "Cannot connect to VieNeu TTS service (:8002).")
        except httpx.TimeoutException:
            raise PipelineError("TTS", "VieNeu TTS service timed out.")
    else:
        try:
            tts_resp = await client.post(
                f"{VBV_URL}/tts/vbv",
                data={"text": response_text},
            )
        except httpx.ConnectError:
            raise PipelineError("TTS", "Cannot connect to VBV TTS service (:8005).")
        except httpx.TimeoutException:
            raise PipelineError("TTS", "VBV TTS service timed out.")

    if tts_resp.status_code != 200:
        raise PipelineError("TTS", tts_resp.text[:300], tts_resp.status_code)

    audio_out = tts_resp.content
    if not audio_out:
        raise PipelineError("TTS", "Empty audio response.", 502)

    return audio_out


async def apply_rvc_conversion(
    audio_bytes: bytes,
    voice_model: str,
    index_path: str,
    pitch: int,
    f0_method: str,
    index_rate: float,
    protect: float,
    clean_audio: bool,
    client: httpx.AsyncClient,
) -> bytes:
    try:
        rvc_resp = await client.post(
            f"{RVC_URL}/convert-voice",
            files={"audio": ("audio.wav", audio_bytes, "audio/wav")},
            data={
                "voice_model": voice_model,
                "index_path": index_path,
                "pitch": str(pitch),
                "f0_method": f0_method,
                "index_rate": str(index_rate),
                "protect": str(protect),
                "clean_audio": str(clean_audio).lower(),
            },
            timeout=300.0,
        )
    except httpx.ConnectError:
        raise PipelineError("RVC", "Cannot connect to RVC service (:8003).")
    except httpx.TimeoutException:
        raise PipelineError("RVC", "RVC service timed out.")

    if rvc_resp.status_code != 200:
        raise PipelineError("RVC", rvc_resp.text[:300], rvc_resp.status_code)

    converted_out = rvc_resp.content
    if not converted_out:
        raise PipelineError("RVC", "Empty audio response from RVC.", 502)

    return converted_out
