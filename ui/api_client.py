"""HTTP client for the Voice Gateway API.

Wraps all Gateway endpoints as simple Python functions.
Used by the Gradio UI - no model imports, just HTTP calls.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import httpx

from common.reference_audio import load_reference_as_wav

GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://localhost:8000")
TIMEOUT = float(os.environ.get("UI_TIMEOUT", "180"))
TTS_TIMEOUT = float(os.environ.get("UI_TTS_TIMEOUT", "900"))

_client = httpx.Client(timeout=TIMEOUT)


class APIError(Exception):
    """Raised when a Gateway call fails."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def _tts_timeout_exc() -> APIError:
    return APIError(
        504,
        f"TTS timed out after {TTS_TIMEOUT:.0f}s. Set UI_TTS_TIMEOUT or shorten the text.",
    )


def _raise_on_error(resp: httpx.Response) -> None:
    if resp.status_code >= 400:
        try:
            body = resp.json()
            detail = body.get("detail") or body.get("error") or resp.text[:300]
        except Exception:
            detail = resp.text[:300]
        raise APIError(resp.status_code, detail)


def _to_wav_bytes(audio_path: str) -> tuple[bytes, str]:
    return load_reference_as_wav(Path(audio_path))


def _save_wav(wav_bytes: bytes) -> str:
    """Save WAV bytes to a temp file and return the path (for Gradio audio output)."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.write(wav_bytes)
    tmp.close()
    return tmp.name


def health() -> dict:
    resp = _client.get(f"{GATEWAY_URL}/health")
    _raise_on_error(resp)
    return resp.json()


def chat(audio_path: str) -> str:
    """POST /chat: full STT->LLM->TTS. For Makefile/scripts; UI uses stepwise calls for progress."""
    wav_bytes, wav_name = _to_wav_bytes(audio_path)
    try:
        resp = _client.post(
            f"{GATEWAY_URL}/chat",
            files={"audio": (wav_name, wav_bytes, "audio/wav")},
            timeout=TTS_TIMEOUT,
        )
    except httpx.ReadTimeout as e:
        raise _tts_timeout_exc() from e
    _raise_on_error(resp)
    return _save_wav(resp.content)


def transcribe(audio_path: str, *, language: str | None = None) -> dict:
    wav_bytes, wav_name = _to_wav_bytes(audio_path)
    data = {}
    if language:
        data["language"] = language
    resp = _client.post(
        f"{GATEWAY_URL}/transcribe",
        data=data or None,
        files={"audio": (wav_name, wav_bytes, "audio/wav")},
    )
    _raise_on_error(resp)
    return resp.json()


def llm_chat(message: str) -> str:
    """POST /llm/chat via gateway - text-only LLM for UI progress flow."""
    resp = _client.post(
        f"{GATEWAY_URL}/llm/chat",
        json={"message": message},
    )
    _raise_on_error(resp)
    return resp.json().get("text", "")


def tts_vieneu(
    *,
    text: str,
    ref_audio_path: str | None = None,
    ref_text: str = "",
    temperature: float = 0.4,
    top_k: int = 50,
    max_chars: int = 256,
) -> str:
    data = {
        "text": text,
        "ref_text": ref_text,
        "temperature": str(temperature),
        "top_k": str(top_k),
        "max_chars": str(max_chars),
    }
    try:
        if ref_audio_path:
            wav_bytes, wav_name = _to_wav_bytes(ref_audio_path)
            resp = _client.post(
                f"{GATEWAY_URL}/tts/vieneu",
                data=data,
                files={"ref_audio": (wav_name, wav_bytes, "audio/wav")},
                timeout=TTS_TIMEOUT,
            )
        else:
            resp = _client.post(f"{GATEWAY_URL}/tts/vieneu", data=data, timeout=TTS_TIMEOUT)
    except httpx.ReadTimeout as e:
        raise _tts_timeout_exc() from e

    _raise_on_error(resp)
    return _save_wav(resp.content)


def tts_fish(
    *,
    text: str,
    ref_audio_path: str | None = None,
    ref_text: str = "",
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.7,
    repetition_penalty: float = 1.2,
    chunk_length: int = 200,
) -> str:
    data = {
        "text": text,
        "ref_text": ref_text,
        "temperature": str(temperature),
        "top_k": str(top_k),
        "top_p": str(top_p),
        "repetition_penalty": str(repetition_penalty),
        "chunk_length": str(chunk_length),
    }
    try:
        if ref_audio_path:
            wav_bytes, wav_name = _to_wav_bytes(ref_audio_path)
            resp = _client.post(
                f"{GATEWAY_URL}/tts/fish-speech",
                data=data,
                files={"ref_audio": (wav_name, wav_bytes, "audio/wav")},
                timeout=TTS_TIMEOUT,
            )
        else:
            resp = _client.post(f"{GATEWAY_URL}/tts/fish-speech", data=data, timeout=TTS_TIMEOUT)
    except httpx.ReadTimeout as e:
        raise _tts_timeout_exc() from e

    _raise_on_error(resp)
    return _save_wav(resp.content)


def tts_vbv(
    *,
    text: str,
    ref_audio_path: str | None = None,
) -> str:
    data = {
        "text": text,
    }
    try:
        if ref_audio_path:
            wav_bytes, wav_name = _to_wav_bytes(ref_audio_path)
            resp = _client.post(
                f"{GATEWAY_URL}/tts/vbv",
                data=data,
                files={"ref_audio": (wav_name, wav_bytes, "audio/wav")},
                timeout=TTS_TIMEOUT,
            )
        else:
            resp = _client.post(f"{GATEWAY_URL}/tts/vbv", data=data, timeout=TTS_TIMEOUT)
    except httpx.ReadTimeout as e:
        raise _tts_timeout_exc() from e
    _raise_on_error(resp)
    return _save_wav(resp.content)


def convert_voice(
    audio_path: str | None,
    voice_model: str,
    index_path: str = "",
    pitch: int = 0,
    f0_method: str = "rmvpe",
    index_rate: float = 0.75,
    protect: float = 0.33,
    clean_audio: bool = False,
    text: str | None = None,
) -> str:
    data = {
        "voice_model": voice_model,
        "index_path": index_path,
        "pitch": str(pitch),
        "f0_method": f0_method,
        "index_rate": str(index_rate),
        "protect": str(protect),
        "clean_audio": str(clean_audio).lower(),
    }
    if text:
        data["text"] = text

    files = {}
    if audio_path:
        wav_bytes, wav_name = _to_wav_bytes(audio_path)
        files = {"audio": (wav_name, wav_bytes, "audio/wav")}

    try:
        resp = _client.post(
            f"{GATEWAY_URL}/convert-voice",
            data=data,
            files=files if files else None,
            timeout=TTS_TIMEOUT,
        )
    except httpx.ReadTimeout as e:
        raise _tts_timeout_exc() from e

    _raise_on_error(resp)
    return _save_wav(resp.content)
