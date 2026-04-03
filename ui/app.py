"""Voice Cloning & AI Assistant: Gradio Web UI.

Run:
    python -m ui.app
    # or: make run_ui
"""

from __future__ import annotations

import os

import gradio as gr

from . import api_client

TITLE = "Voice Cloning & AI Assistant"

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Bundled default for Morgan-style fish-speech ref (place your sample here or upload in UI).
_DEFAULT_MORGAN_WAV = os.path.join(_REPO_ROOT, "audio", "output", "morgan_freeman.wav")
# Optional overrides (same paths as before if you prefer env over UI).
VOICE_CHAT_FISH_REF_AUDIO = os.environ.get("VOICE_CHAT_FISH_REF_AUDIO", "").strip()
VOICE_CHAT_FISH_REF_TEXT = os.environ.get("VOICE_CHAT_FISH_REF_TEXT", "").strip()


def _resolve_fish_ref_path(uploaded_ref: str | None) -> str | None:
    """Upload wins, then env, then bundled morgan_freeman.wav if present."""
    if uploaded_ref and os.path.isfile(uploaded_ref):
        return uploaded_ref
    if VOICE_CHAT_FISH_REF_AUDIO and os.path.isfile(VOICE_CHAT_FISH_REF_AUDIO):
        return VOICE_CHAT_FISH_REF_AUDIO
    if os.path.isfile(_DEFAULT_MORGAN_WAV):
        return _DEFAULT_MORGAN_WAV
    return None


def on_chat(
    audio_path: str | None,
    engine_choice: str,
    fish_ref_audio: str | None,
    fish_ref_text: str,
):
    """STT → LLM → TTS with per-step status (mirrors gateway orchestrator, adds UX)."""
    if audio_path is None:
        yield "Error: record or upload an audio file first.", "", "", None
        return

    yield "Transcribing...", "", "", None
    try:
        result = api_client.transcribe(audio_path)
    except api_client.APIError as e:
        yield f"Error at STT: {e.detail}", "", "", None
        return

    user_text = result.get("text", "").strip()
    if not user_text:
        yield "Error: could not transcribe audio — try speaking more clearly.", "", "", None
        return

    yield "Thinking...", user_text, "", None
    try:
        ai_text = api_client.llm_chat(user_text)
    except api_client.APIError as e:
        yield f"Error at LLM: {e.detail}", user_text, "", None
        return

    if not ai_text.strip():
        yield "Error: AI returned an empty response.", user_text, "", None
        return

    yield f"Generating speech ({engine_choice})...", user_text, ai_text, None
    try:
        if "Fish" in engine_choice:
            ref_path = _resolve_fish_ref_path(fish_ref_audio)
            if not ref_path:
                yield (
                    "Error: fish-speech needs a reference WAV "
                    "(upload one or add audio/output/morgan_freeman.wav).",
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
    except api_client.APIError as e:
        yield f"Error at TTS: {e.detail}", user_text, ai_text, None
        return

    yield "", user_text, ai_text, audio_out


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
        if engine == "VieNeu-TTS":
            return api_client.tts_vieneu(
                text=text,
                ref_audio_path=ref_audio,
                ref_text=ref_text,
                temperature=temperature,
                top_k=int(top_k),
                max_chars=int(max_chars),
            )

        if ref_audio is None:
            raise gr.Error("fish-speech requires a reference audio file.")
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
    is_fish = engine == "fish-speech"
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


def build_app() -> gr.Blocks:
    with gr.Blocks(title=TITLE, theme=gr.themes.Soft()) as app:
        gr.Markdown(f"# {TITLE}")

        with gr.Tabs():
            with gr.TabItem("Voice Chat"):
                chat_status = gr.Markdown(value="")
                with gr.Row():
                    with gr.Column():
                        chat_audio_in = gr.Audio(
                            sources=["microphone", "upload"],
                            type="filepath",
                            label="Your voice",
                        )
                        chat_tts_engine = gr.Radio(
                            choices=["VieNeu (Fastest)", "Fish-Speech (Morgan Freeman)"],
                            value="Fish-Speech (Morgan Freeman)",
                            label="Voice engine (AI reply)",
                        )
                        with gr.Accordion("Fish-speech reference (timbre)", open=False):
                            chat_fish_ref_audio = gr.Audio(
                                sources=["upload"],
                                type="filepath",
                                label="Reference voice (optional)",
                            )
                            chat_fish_ref_text = gr.Textbox(
                                label="Reference transcript (optional, auto-filled if you upload)",
                                lines=2,
                                placeholder="What the reference clip says - edit if STT is wrong",
                            )
                        chat_btn = gr.Button("Send", variant="primary")
                    with gr.Column():
                        chat_user_text = gr.Textbox(
                            label="You said",
                            lines=2,
                            interactive=False,
                        )
                        chat_ai_text = gr.Textbox(
                            label="AI response",
                            lines=3,
                            interactive=False,
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
                        chat_tts_engine,
                        chat_fish_ref_audio,
                        chat_fish_ref_text,
                    ],
                    outputs=[chat_status, chat_user_text, chat_ai_text, chat_audio_out],
                )

            with gr.TabItem("Text-to-Speech"):
                with gr.Row():
                    with gr.Column():
                        tts_text = gr.Textbox(
                            label="Text",
                            placeholder="Nhập text cần đọc...",
                            lines=3,
                        )
                        tts_engine = gr.Dropdown(
                            choices=["VieNeu-TTS", "fish-speech"],
                            value="VieNeu-TTS",
                            label="Engine",
                        )
                        tts_ref_audio = gr.Audio(
                            sources=["upload"],
                            type="filepath",
                            label="Reference audio (optional, for voice cloning)",
                        )
                        tts_ref_text = gr.Textbox(
                            label="Reference text (auto-filled, editable)",
                            placeholder=(
                                "Auto-transcribed when you upload ref audio. "
                                "Edit if STT got it wrong."
                            ),
                            lines=2,
                            visible=True,
                        )

                        with gr.Accordion("Parameters", open=False):
                            tts_temp = gr.Slider(0.1, 1.5, value=1.0, step=0.1, label="Temperature")
                            tts_top_k = gr.Slider(1, 100, value=50, step=1, label="Top-K")
                            tts_max_chars = gr.Slider(
                                64,
                                512,
                                value=256,
                                step=16,
                                label="Max chars per chunk (VieNeu)",
                            )
                            tts_top_p = gr.Slider(
                                0.1, 1.0, value=0.7, step=0.05, label="Top-P", visible=False
                            )
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

                        tts_btn = gr.Button("Generate", variant="primary")
                    with gr.Column():
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

            with gr.TabItem("Voice Conversion"):
                with gr.Row():
                    with gr.Column():
                        rvc_audio_in = gr.Audio(
                            sources=["upload"], type="filepath", label="Input audio"
                        )
                        rvc_model = gr.Textbox(
                            label="Voice model name",
                            placeholder="e.g. target (looks for models/rvc/target.pth)",
                        )
                        with gr.Accordion("Parameters", open=False):
                            rvc_pitch = gr.Slider(
                                -24, 24, value=0, step=1, label="Pitch (semitones)"
                            )
                            rvc_f0 = gr.Dropdown(
                                choices=["rmvpe", "crepe", "fcpe"],
                                value="rmvpe",
                                label="F0 method",
                            )
                            rvc_index = gr.Slider(
                                0.0, 1.0, value=0.75, step=0.05, label="Index rate"
                            )
                            rvc_protect = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Protect")
                            rvc_clean = gr.Checkbox(
                                label="Clean audio (noise reduction)", value=False
                            )
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

        with gr.Accordion("System Status", open=False):
            health_btn = gr.Button("Check services", size="sm")
            health_output = gr.Markdown()
            health_btn.click(fn=on_health_check, inputs=[], outputs=[health_output])

    return app


if __name__ == "__main__":
    share = os.environ.get("GRADIO_SHARE", "false").lower() == "true"
    build_app().launch(server_name="0.0.0.0", server_port=7860, share=share)
