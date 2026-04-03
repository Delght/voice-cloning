"""Gradio Voice Chat: STT→LLM→TTS (same stages as `gateway.orchestrator.run_chat_pipeline`).

`POST /chat` runs that pipeline server-side with **VieNeu-only** TTS. This module uses
separate gateway calls (`/transcribe`, `/llm/chat`, `/tts/*`) so the UI can show progress.
When Fish-Speech is selected, TTS differs from `/chat` - keep both code paths in mind when
changing defaults or error handling.

See: `gateway/orchestrator.py`.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

from . import api_client

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_DEFAULT_MORGAN_WAV = os.path.join(_REPO_ROOT, "audio", "output", "morgan_freeman.wav")
VOICE_CHAT_FISH_REF_AUDIO = os.environ.get("VOICE_CHAT_FISH_REF_AUDIO", "").strip()
VOICE_CHAT_FISH_REF_TEXT = os.environ.get("VOICE_CHAT_FISH_REF_TEXT", "").strip()


def resolve_fish_ref_path(uploaded_ref: str | None) -> str | None:
    """Upload wins, then env, then bundled `morgan_freeman.wav` if present."""
    if uploaded_ref and os.path.isfile(uploaded_ref):
        return uploaded_ref
    if VOICE_CHAT_FISH_REF_AUDIO and os.path.isfile(VOICE_CHAT_FISH_REF_AUDIO):
        return VOICE_CHAT_FISH_REF_AUDIO
    if os.path.isfile(_DEFAULT_MORGAN_WAV):
        return _DEFAULT_MORGAN_WAV
    return None


def run_voice_chat(
    audio_path: str | None,
    engine_choice: str,
    fish_ref_audio: str | None,
    fish_ref_text: str,
) -> Iterator[tuple[str, str, str, str | None]]:
    """Yield `(status_md, user_text, ai_text, audio_path)` for Gradio."""
    if audio_path is None:
        yield "", "", "", None
        return

    yield "", "", "", None
    try:
        result = api_client.transcribe(audio_path)
    except api_client.APIError:
        yield "", "", "", None
        return

    user_text = result.get("text", "").strip()
    if not user_text:
        yield "", "", "", None
        return

    yield "", user_text, "", None
    try:
        ai_text = api_client.llm_chat(user_text)
    except api_client.APIError:
        yield "", user_text, "", None
        return

    if not ai_text.strip():
        yield "", user_text, "", None
        return

    yield "", user_text, ai_text, None
    try:
        if "Fish" in engine_choice:
            ref_path = resolve_fish_ref_path(fish_ref_audio)
            if not ref_path:
                yield (
                    "",
                    user_text,
                    ai_text,
                    None,
                )
                return
            ref_txt = (fish_ref_text or "").strip() or VOICE_CHAT_FISH_REF_TEXT
            audio_out = api_client.tts_fish(
                text=ai_text,
                ref_audio_path=ref_path,
                ref_text=ref_txt,
            )
        else:
            audio_out = api_client.tts_vieneu(text=ai_text)
    except api_client.APIError:
        yield "", user_text, ai_text, None
        return

    yield "", user_text, ai_text, audio_out
