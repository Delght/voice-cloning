"""POST /chat: STT, LLM, fish-speech TTS (ref audio from env or bundled WAV)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import httpx

from common.fish_ref import default_fish_ref_path
from common.reference_audio import load_reference_as_wav

from .config import LLM_URL, STT_URL, TTS_URL


def _resolve_chat_fish_ref_path() -> Path | None:
    env = (
        os.environ.get("CHAT_FISH_REF_AUDIO") or os.environ.get("VOICE_CHAT_FISH_REF_AUDIO") or ""
    ).strip()
    if env:
        p = Path(env)
        if p.is_file():
            return p
    default_ref = default_fish_ref_path()
    if default_ref.is_file():
        return default_ref
    return None


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

    ref_path = _resolve_chat_fish_ref_path()
    if ref_path is None:
        raise PipelineError(
            "TTS",
            "Missing fish-speech reference (CHAT_FISH_REF_AUDIO, VOICE_CHAT_FISH_REF_AUDIO, "
            "or audio/reference/phuong_anh.wav).",
            503,
        )
    ref_text = (
        os.environ.get("CHAT_FISH_REF_TEXT") or os.environ.get("VOICE_CHAT_FISH_REF_TEXT") or ""
    ).strip()
    try:
        ref_bytes, ref_filename = load_reference_as_wav(ref_path)
    except FileNotFoundError as e:
        raise PipelineError("TTS", f"Reference audio missing: {e}", 503) from e
    except subprocess.CalledProcessError as e:
        raise PipelineError("TTS", f"ffmpeg could not convert reference audio: {e}", 503) from e

    try:
        tts_resp = await client.post(
            f"{TTS_URL}/tts/fish-speech",
            files={"ref_audio": (ref_filename, ref_bytes, "audio/wav")},
            data={"text": response_text, "ref_text": ref_text},
        )
    except httpx.ConnectError:
        raise PipelineError("TTS", "Cannot connect to TTS service (:8002).")
    except httpx.TimeoutException:
        raise PipelineError("TTS", "TTS service timed out.")

    if tts_resp.status_code != 200:
        raise PipelineError("TTS", tts_resp.text[:300], tts_resp.status_code)

    audio_out = tts_resp.content
    if not audio_out:
        raise PipelineError("TTS", "Empty audio response.", 502)

    return audio_out
