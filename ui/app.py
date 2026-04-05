"""Gradio UI for the voice project."""

from __future__ import annotations

import inspect
import os
import re

import gradio as gr

from common.fish_ref import default_fish_ref_path

from . import api_client

TITLE = "Fiona Anne"

ENGINE_VIENEU = "VieNeu-TTS"
ENGINE_FISH = "fish-speech"
ENGINE_CHOICES: tuple[str, ...] = (ENGINE_VIENEU, ENGINE_FISH)
_DEFAULT_ENGINE = ENGINE_FISH

_THEME_BODY_BG_DARK = "#0a1f24"
_THEME_SURFACE_BG_DARK = "#0e2429"
_HEADING_ACCENT = "#22d3ee"

_APP_THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.cyan,
    secondary_hue=gr.themes.colors.teal,
    neutral_hue=gr.themes.colors.slate,
).set(
    body_background_fill_dark=_THEME_BODY_BG_DARK,
    background_fill_primary_dark=_THEME_SURFACE_BG_DARK,
)


def _gradio_theme_kw_for_blocks() -> dict[str, object]:
    """Newer Gradio passes theme to launch(); older versions use Blocks(theme=...)."""
    if "theme" in inspect.signature(gr.Blocks.launch).parameters:
        return {}
    return {"theme": _APP_THEME}


def _gradio_theme_kw_for_launch() -> dict[str, object]:
    if "theme" in inspect.signature(gr.Blocks.launch).parameters:
        return {"theme": _APP_THEME}
    return {}


_DEFAULT_FISH_REF_PATH = str(default_fish_ref_path())
VOICE_CHAT_FISH_REF_AUDIO = os.environ.get("VOICE_CHAT_FISH_REF_AUDIO", "").strip()
VOICE_CHAT_FISH_REF_TEXT = os.environ.get("VOICE_CHAT_FISH_REF_TEXT", "").strip()


def _markdown_to_plain_for_tts(text: str) -> str:
    if not (text or "").strip():
        return ""
    t = text
    t = re.sub(r"```[\s\S]*?```", " ", t)
    t = re.sub(r"`([^`]+)`", r"\1", t)
    t = re.sub(r"^#{1,6}\s*", "", t, flags=re.MULTILINE)
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)
    for _ in range(8):
        nt = re.sub(r"\*\*([^*]+)\*\*", r"\1", t)
        if nt == t:
            break
        t = nt
    t = re.sub(r"(?m)^\s*[*-]\s+", "", t)
    t = re.sub(r"(?m)^\s*\d+\.\s+", "", t)
    t = re.sub(r"(?<!\*)\*(?!\*)([^*\n]+?)(?<!\*)\*(?!\*)", r"\1", t)
    t = re.sub(r"(?m)^\s*([*_-])\s*\1\s*\1+\s*$", "", t)
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _resolve_fish_ref_path(uploaded_ref: str | None) -> str | None:
    if uploaded_ref and os.path.isfile(uploaded_ref):
        return uploaded_ref
    if VOICE_CHAT_FISH_REF_AUDIO and os.path.isfile(VOICE_CHAT_FISH_REF_AUDIO):
        return VOICE_CHAT_FISH_REF_AUDIO
    if os.path.isfile(_DEFAULT_FISH_REF_PATH):
        return _DEFAULT_FISH_REF_PATH
    return None


def _voice_chat_ai_heading_html() -> str:
    return (
        f'<p style="margin:0 0 0.35rem 0;font-weight:600;'
        f'font-size:1.05rem;color:{_HEADING_ACCENT}">'
        "AI response</p>"
    )


def _synthesize_voice_chat_reply(
    ai_text: str,
    fish_ref_audio: str | None,
    fish_ref_text: str,
) -> str:
    ref_path = _resolve_fish_ref_path(fish_ref_audio)
    if not ref_path:
        raise ValueError(
            "No fish-speech reference: upload in the accordion, restore the bundled "
            "default ref, or set VOICE_CHAT_FISH_REF_AUDIO."
        )
    ref_txt = (fish_ref_text or "").strip() or VOICE_CHAT_FISH_REF_TEXT
    return api_client.tts_fish(
        text=ai_text,
        ref_audio_path=ref_path,
        ref_text=ref_txt,
    )


def on_chat(
    audio_path: str | None,
    fish_ref_audio: str | None,
    fish_ref_text: str,
):
    if audio_path is None:
        yield "Error: record or upload an audio file first.", "", "", "", None
        return

    yield "Transcribing...", "", "", "", None
    try:
        result = api_client.transcribe(audio_path)
    except api_client.APIError as e:
        yield f"Error at STT: {e.detail}", "", "", "", None
        return

    user_text = result.get("text", "").strip()
    if not user_text:
        yield "Error: could not transcribe audio. Try speaking more clearly.", "", "", "", None
        return

    yield "Thinking...", user_text, "", "", None
    try:
        ai_text = api_client.llm_chat(user_text)
    except api_client.APIError as e:
        yield f"Error at LLM: {e.detail}", user_text, "", "", None
        return

    if not ai_text.strip():
        yield "Error: AI returned an empty response.", user_text, "", "", None
        return

    yield "", user_text, ai_text, ai_text, None


def on_chat_speak(
    ai_raw_markdown: str,
    fish_ref_audio: str | None,
    fish_ref_text: str,
):
    plain = _markdown_to_plain_for_tts(ai_raw_markdown)
    if not plain.strip():
        raise gr.Error("No AI reply yet. Send a voice message first, then click Speak.")

    try:
        return _synthesize_voice_chat_reply(plain, fish_ref_audio, fish_ref_text)
    except ValueError as e:
        raise gr.Error(str(e)) from e
    except api_client.APIError as e:
        raise gr.Error(str(e.detail)) from e


def on_tts(
    text: str,
    engine: str,
    ref_audio: str | None,
    ref_text: str,
    temperature: float,
    top_k: int,
    max_chars: int,
    top_p: float,
    rep_penalty: float,
    chunk_len: int,
):
    if not text.strip():
        raise gr.Error("Enter some text to synthesize.")

    try:
        if engine == ENGINE_VIENEU:
            return api_client.tts_vieneu(
                text=text,
                ref_audio_path=ref_audio,
                ref_text=ref_text,
                temperature=temperature,
                top_k=int(top_k),
                max_chars=int(max_chars),
            )

        if ref_audio is None:
            raise gr.Error(f"{ENGINE_FISH} requires a reference audio file.")
        return api_client.tts_fish(
            text=text,
            ref_audio_path=ref_audio,
            ref_text=ref_text,
            temperature=temperature,
            top_p=float(top_p),
            repetition_penalty=float(rep_penalty),
            chunk_length=int(chunk_len),
        )
    except api_client.APIError as e:
        raise gr.Error(f"TTS error: {e.detail}") from e


def on_ref_audio_upload(audio_path: str | None):
    """Auto-transcribe reference audio to fill ref_text."""
    if audio_path is None:
        return ""
    try:
        result = api_client.transcribe(audio_path)
        text = result.get("text", "").strip()
        if text:
            return text

        result_vi = api_client.transcribe(audio_path, language="vi")
        return result_vi.get("text", "").strip()
    except api_client.APIError:
        return ""


def on_engine_change(engine: str):
    is_fish = engine == ENGINE_FISH
    ref_label = (
        "Reference audio (required)" if is_fish else "Reference audio (optional, for voice cloning)"
    )
    return [
        gr.update(label=ref_label),
        gr.update(visible=True),
        gr.update(value=0.7 if is_fish else 1.0),  # temperature default
        gr.update(visible=not is_fish),  # top_k
        gr.update(visible=not is_fish),  # max_chars
        gr.update(visible=is_fish),  # top_p
        gr.update(visible=is_fish),  # repetition penalty
        gr.update(visible=is_fish),  # chunk length
    ]


def on_convert(
    audio_path: str | None,
    voice_model: str,
    pitch: int,
    f0_method: str,
    index_rate: float,
    protect: float,
    clean_audio: bool,
):
    if audio_path is None:
        raise gr.Error("Upload an audio file first.")
    if not voice_model.strip():
        raise gr.Error("Enter a voice model name (e.g. 'target').")
    try:
        return api_client.convert_voice(
            audio_path=audio_path,
            voice_model=voice_model.strip(),
            pitch=int(pitch),
            f0_method=f0_method,
            index_rate=float(index_rate),
            protect=float(protect),
            clean_audio=bool(clean_audio),
        )
    except api_client.APIError as e:
        raise gr.Error(f"RVC error: {e.detail}") from e


def on_transcribe(audio_path: str | None):
    if audio_path is None:
        raise gr.Error("Upload or record an audio file first.")
    try:
        result = api_client.transcribe(audio_path)
    except api_client.APIError as e:
        raise gr.Error(f"STT error: {e.detail}") from e

    text = result.get("text", "")
    lang = result.get("language", "?")
    prob = result.get("language_probability", 0)
    duration = result.get("duration", 0)
    segments = result.get("segments", [])

    seg_lines = [f"[{s['start']:.1f}s - {s['end']:.1f}s] {s['text']}" for s in segments]
    seg_text = "\n".join(seg_lines) if seg_lines else "(no segments)"
    info = f"**Language:** {lang} ({prob:.0%}) | **Duration:** {duration:.1f}s"
    return text, info, seg_text


def on_health_check():
    try:
        data = api_client.health()
        lines = [f"**Gateway:** {data.get('gateway', '?')}"]
        for svc in data.get("services", []):
            status = svc.get("status", "?")
            icon = {"ok": "OK", "down": "DOWN", "timeout": "TIMEOUT"}.get(status, status)
            lines.append(f"- **{svc['service']}** ({svc.get('url', '')}) - {icon}")
        return "\n".join(lines)
    except Exception as e:
        return f"**Error:** {e}"


def _add_voice_chat_tab() -> None:
    with gr.TabItem("Voice Chat"):
        chat_status = gr.Markdown(value="")
        with gr.Column():
            chat_audio_in = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Your question",
            )
            with gr.Accordion("AI response voice reference, optional override", open=False):
                chat_fish_ref_audio = gr.Audio(
                    sources=["upload"],
                    type="filepath",
                    label="Reference voice (optional)",
                )
                chat_fish_ref_text = gr.Textbox(
                    label="Transcript (optional, auto-fill)",
                    lines=2,
                    placeholder="What the reference clip says - edit if STT is wrong",
                )
            chat_btn = gr.Button("Send", variant="primary", size="md")
            chat_user_text = gr.Textbox(
                label="You said",
                lines=2,
                interactive=False,
            )
            gr.HTML(_voice_chat_ai_heading_html())
            chat_ai_markdown = gr.Markdown(value="")
            chat_ai_raw_state = gr.State("")
            chat_speak_btn = gr.Button(
                "Speak",
                variant="secondary",
                size="sm",
            )
            chat_audio_out = gr.Audio(
                type="filepath",
                label="AI voice",
                interactive=False,
                autoplay=True,
            )
        chat_fish_ref_audio.change(
            fn=on_ref_audio_upload,
            inputs=[chat_fish_ref_audio],
            outputs=[chat_fish_ref_text],
        )
        chat_btn.click(
            on_chat,
            inputs=[
                chat_audio_in,
                chat_fish_ref_audio,
                chat_fish_ref_text,
            ],
            outputs=[
                chat_status,
                chat_user_text,
                chat_ai_markdown,
                chat_ai_raw_state,
                chat_audio_out,
            ],
        )
        chat_speak_btn.click(
            on_chat_speak,
            inputs=[
                chat_ai_raw_state,
                chat_fish_ref_audio,
                chat_fish_ref_text,
            ],
            outputs=[chat_audio_out],
        )


def _add_voice_cloning_tab() -> None:
    with gr.TabItem("Voice Cloning"):
        with gr.Column():
            tts_engine = gr.Dropdown(
                choices=list(ENGINE_CHOICES),
                value=_DEFAULT_ENGINE,
                label="Engine",
            )
            with gr.Accordion("Parameters", open=False):
                tts_temp = gr.Slider(0.1, 1.5, value=0.7, step=0.1, label="Temperature")
                tts_top_k = gr.Slider(1, 100, value=50, step=1, label="Top-K")
                tts_max_chars = gr.Slider(
                    64,
                    512,
                    value=256,
                    step=16,
                    label="Max chars per chunk (VieNeu)",
                )
                tts_top_p = gr.Slider(0.1, 1.0, value=0.7, step=0.05, label="Top-P", visible=False)
                tts_rep = gr.Slider(
                    1.0,
                    2.0,
                    value=1.2,
                    step=0.1,
                    label="Repetition penalty",
                    visible=False,
                )
                tts_chunk = gr.Slider(
                    50, 500, value=200, step=50, label="Chunk length", visible=False
                )
            with gr.Group():
                tts_ref_audio = gr.Audio(
                    sources=["upload"],
                    type="filepath",
                    label="Reference audio (required)",
                )
                tts_ref_text = gr.Textbox(
                    label="Transcript (auto-filled)",
                    placeholder="Auto-filled after upload; edit if wrong.",
                    lines=3,
                    visible=True,
                )
            tts_text = gr.Textbox(
                label="Text to speak",
                placeholder="What the clone should say...",
                lines=4,
            )
            tts_btn = gr.Button("Generate", variant="primary", size="md")
            tts_audio_out = gr.Audio(
                type="filepath",
                label="Output audio",
                interactive=False,
            )

        tts_ref_audio.change(
            fn=on_ref_audio_upload,
            inputs=[tts_ref_audio],
            outputs=[tts_ref_text],
        )
        tts_engine.change(
            fn=on_engine_change,
            inputs=[tts_engine],
            outputs=[
                tts_ref_audio,
                tts_ref_text,
                tts_temp,
                tts_top_k,
                tts_max_chars,
                tts_top_p,
                tts_rep,
                tts_chunk,
            ],
        )
        tts_btn.click(
            fn=on_tts,
            inputs=[
                tts_text,
                tts_engine,
                tts_ref_audio,
                tts_ref_text,
                tts_temp,
                tts_top_k,
                tts_max_chars,
                tts_top_p,
                tts_rep,
                tts_chunk,
            ],
            outputs=[tts_audio_out],
        )


def _add_voice_conversion_tab() -> None:
    with gr.TabItem("Voice Conversion"):
        with gr.Row():
            with gr.Column():
                rvc_audio_in = gr.Audio(sources=["upload"], type="filepath", label="Input audio")
                rvc_model = gr.Textbox(
                    label="Voice model name",
                    placeholder="e.g. target (looks for models/rvc/target.pth)",
                )
                with gr.Accordion("Parameters", open=False):
                    rvc_pitch = gr.Slider(-24, 24, value=0, step=1, label="Pitch (semitones)")
                    rvc_f0 = gr.Dropdown(
                        choices=["rmvpe", "crepe", "fcpe"],
                        value="rmvpe",
                        label="F0 method",
                    )
                    rvc_index = gr.Slider(0.0, 1.0, value=0.75, step=0.05, label="Index rate")
                    rvc_protect = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Protect")
                    rvc_clean = gr.Checkbox(label="Clean audio (noise reduction)", value=False)
                rvc_btn = gr.Button("Convert", variant="primary")
            with gr.Column():
                rvc_audio_out = gr.Audio(
                    type="filepath",
                    label="Converted audio",
                    interactive=False,
                )

        rvc_btn.click(
            fn=on_convert,
            inputs=[
                rvc_audio_in,
                rvc_model,
                rvc_pitch,
                rvc_f0,
                rvc_index,
                rvc_protect,
                rvc_clean,
            ],
            outputs=[rvc_audio_out],
        )


def _add_transcribe_tab() -> None:
    with gr.TabItem("Transcribe"):
        with gr.Row():
            with gr.Column():
                stt_audio_in = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label="Audio input",
                )
                stt_btn = gr.Button("Transcribe", variant="primary")
            with gr.Column():
                stt_text_out = gr.Textbox(label="Transcript", lines=3, interactive=False)
                stt_info = gr.Markdown(label="Info")
                stt_segments = gr.Textbox(
                    label="Segments (with timestamps)",
                    lines=8,
                    interactive=False,
                )
        stt_btn.click(
            on_transcribe,
            inputs=[stt_audio_in],
            outputs=[stt_text_out, stt_info, stt_segments],
        )


def _add_system_status_section() -> None:
    with gr.Accordion("System Status", open=False):
        health_btn = gr.Button("Check services", size="sm")
        health_output = gr.Markdown()
        health_btn.click(fn=on_health_check, inputs=[], outputs=[health_output])


def build_app() -> gr.Blocks:
    with gr.Blocks(title=TITLE, **_gradio_theme_kw_for_blocks()) as app:
        gr.Markdown(f"# {TITLE}")
        with gr.Tabs():
            _add_voice_chat_tab()
            _add_voice_cloning_tab()
            _add_voice_conversion_tab()
            _add_transcribe_tab()
        _add_system_status_section()
    return app


def main() -> None:
    share = os.environ.get("GRADIO_SHARE", "false").lower() == "true"
    build_app().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=share,
        **_gradio_theme_kw_for_launch(),
    )


if __name__ == "__main__":
    main()
