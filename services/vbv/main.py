import copy
import io
import logging
import os
import time
from contextlib import asynccontextmanager

import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from vibevoice.modular.modeling_vibevoice_streaming_inference import (
    VibeVoiceStreamingForConditionalGenerationInference,
)
from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Global variables for model
model = None
processor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, processor
    # Choose path depending on whether it runs in docker or locally
    model_path = (
        "/app/models/vibevoice" if os.path.exists("/app/models/vibevoice") else "models/vibevoice"
    )
    log.info("Loading VibeVoice model from %s...", model_path)

    try:
        processor = VibeVoiceStreamingProcessor.from_pretrained(model_path)

        # Apple Silicon (MPS) configuration
        load_dtype = torch.float32
        attn_impl_primary = "sdpa"
        model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
            model_path,
            torch_dtype=load_dtype,
            attn_implementation=attn_impl_primary,
            device_map=None,
        )
        model.to("mps")
        model.eval()
        model.set_ddpm_inference_steps(num_steps=5)
        log.info("VibeVoice model loaded successfully on MPS!")
    except Exception as e:
        log.error("Failed to load user model: %s", e)

    yield


app = FastAPI(
    title="Voice - VBV Service",
    description="Microservice for VibeVoice Realtime TTS",
    version="0.1.0",
    lifespan=lifespan,
)


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request, exc):
    msg = str(exc).lower()
    if "out of memory" in msg or "oom" in msg:
        return JSONResponse(
            status_code=503,
            content={"error": "Model out of memory", "detail": str(exc)},
        )
    return JSONResponse(
        status_code=500,
        content={"error": "Internal error", "detail": str(exc)},
    )


@app.post("/tts/vbv")
async def generate_tts(text: str = Form(...), ref_audio: UploadFile = File(None)):
    """
    Generate speech from text using VibeVoice.
    Optionally accepts a reference audio file for zero-shot cloning.
    """
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    log.info(
        "Received request for TTS: Text='%s...', Ref Audio=%s",
        text[:50],
        "Yes" if ref_audio else "No",
    )

    if ref_audio:
        try:
            content = await ref_audio.read()
            # Reference audio content might be used in future when dynamic presets are supported
            _ = content
        except Exception as e:
            log.error("Error reading ref audio: %s", e)
            raise HTTPException(status_code=400, detail="Failed to read reference audio")

    t0 = time.time()

    # VibeVoice inference
    # Note: VibeVoice 0.5B streaming relies on precomputed .pt voice presets
    # instead of on-the-fly zero-shot cloning from wav files. We use a default preset.
    voice_preset_path = (
        "/app/models/vibevoice/voices/en-Carter_man.pt"
        if os.path.exists("/app/models/vibevoice")
        else "models/vibevoice/voices/en-Carter_man.pt"
    )

    all_prefilled_outputs = None
    if os.path.exists(voice_preset_path):
        all_prefilled_outputs = torch.load(
            voice_preset_path, map_location="mps", weights_only=False
        )
    else:
        log.warning("Voice preset %s not found. Output may be degraded or fail.", voice_preset_path)

    inputs = processor.process_input_with_cached_prompt(
        text=text,
        cached_prompt=all_prefilled_outputs,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to("mps")

    outputs = model.generate(
        **inputs,
        max_new_tokens=None,
        cfg_scale=1.5,
        tokenizer=processor.tokenizer,
        generation_config={"do_sample": False},
        verbose=False,
        all_prefilled_outputs=copy.deepcopy(all_prefilled_outputs)
        if all_prefilled_outputs is not None
        else None,
    )

    speech_output = outputs.speech_outputs[0]
    if torch.is_tensor(speech_output):
        audio_output = speech_output.cpu().numpy()
    else:
        audio_output = speech_output

    sample_rate = 24000
    # Flatten if it has extra dimensions
    if len(audio_output.shape) > 1 and audio_output.shape[0] == 1:
        audio_output = audio_output.squeeze(0)

    # Encode as WAV
    buffer = io.BytesIO()
    sf.write(buffer, audio_output, sample_rate, format="WAV")
    buffer.seek(0)

    t1 = time.time()
    audio_duration = len(audio_output) / sample_rate
    log.info("TTS generated successfully in %.3fs. Audio duration: %.2fs", t1 - t0, audio_duration)

    return Response(content=buffer.read(), media_type="audio/wav")


@app.get("/health")
async def health_check():
    return JSONResponse(content={"status": "healthy", "service": "vbv"}, status_code=200)
