"""HTTP client for the Voice Gateway API.

Wraps all Gateway endpoints as simple Python functions.
Used by the Gradio UI — no model imports, just HTTP calls.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import httpx

GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://localhost:8000")
TIMEOUT = float(os.environ.get("UI_TIMEOUT", "180"))

_client = httpx.Client(timeout=TIMEOUT)


class APIError(Exception):
    """Raised when a Gateway call fails."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def _raise_on_error(resp: httpx.Response) -> None:
    if resp.status_code >= 400:
        try:
            body = resp.json()
            detail = body.get("detail") or body.get("error") or resp.text[:300]
        except Exception:
            detail = resp.text[:300]
        raise APIError(resp.status_code, detail)


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
    with open(audio_path, "rb") as f:
        resp = _client.post(
            f"{GATEWAY_URL}/chat",
            files={"audio": ("audio.wav", f, "audio/wav")},
        )
    _raise_on_error(resp)
    return _save_wav(resp.content)


def transcribe(audio_path: str, *, language: str | None = None) -> dict:
    with open(audio_path, "rb") as f:
        data = {}
        if language:
            data["language"] = language
        resp = _client.post(
            f"{GATEWAY_URL}/transcribe",
            data=data or None,
            files={"audio": ("audio.wav", f, "audio/wav")},
        )
    _raise_on_error(resp)
    return resp.json()


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
    files: dict[str, tuple[str, object, str]] = {}
    if ref_audio_path:
        p = Path(ref_audio_path)
        files["ref_audio"] = (p.name, open(ref_audio_path, "rb"), "application/octet-stream")

    try:
        resp = _client.post(f"{GATEWAY_URL}/tts/vieneu", data=data, files=files or None)
    finally:
        for _, fobj in files.items():
            fobj[1].close()

    _raise_on_error(resp)
    return _save_wav(resp.content)


def tts_fish(
    *,
    text: str,
    ref_audio_path: str,
    ref_text: str = "",
    temperature: float = 0.7,
    top_p: float = 0.7,
    repetition_penalty: float = 1.2,
    chunk_length: int = 200,
) -> str:
    data = {
        "text": text,
        "ref_text": ref_text,
        "temperature": str(temperature),
        "top_p": str(top_p),
        "repetition_penalty": str(repetition_penalty),
        "chunk_length": str(chunk_length),
    }
    p = Path(ref_audio_path)
    with open(ref_audio_path, "rb") as f:
        resp = _client.post(
            f"{GATEWAY_URL}/tts/fish-speech",
            data=data,
            files={"ref_audio": (p.name, f, "application/octet-stream")},
        )
    _raise_on_error(resp)
    return _save_wav(resp.content)


def convert_voice(
    *,
    audio_path: str,
    voice_model: str,
    pitch: int = 0,
    f0_method: str = "rmvpe",
    index_rate: float = 0.75,
    protect: float = 0.5,
    clean_audio: bool = False,
) -> str:
    data = {
        "voice_model": voice_model,
        "pitch": str(pitch),
        "f0_method": f0_method,
        "index_rate": str(index_rate),
        "protect": str(protect),
        "clean_audio": str(clean_audio).lower(),
    }
    with open(audio_path, "rb") as f:
        resp = _client.post(
            f"{GATEWAY_URL}/convert-voice",
            data=data,
            files={"audio": ("audio.wav", f, "audio/wav")},
        )
    _raise_on_error(resp)
    return _save_wav(resp.content)
