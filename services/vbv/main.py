import asyncio
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

# Dynamic device detection — repo convention: cuda → mps → cpu; never hardcode
device: str = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

# Global model state
model: VibeVoiceStreamingForConditionalGenerationInference | None = None
processor: VibeVoiceStreamingProcessor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, processor
    model_path = (
        "/app/models/vibevoice" if os.path.exists("/app/models/vibevoice") else "models/vibevoice"
    )
    log.info("Loading VibeVoice model from %s on device=%s...", model_path, device)

    try:
        processor = VibeVoiceStreamingProcessor.from_pretrained(model_path)

        # float16 on CUDA for speed; float32 on MPS/CPU (no float16 support)
        load_dtype = torch.float16 if device == "cuda" else torch.float32
        model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
            model_path,
            torch_dtype=load_dtype,
            attn_implementation="sdpa",
            device_map=None,
        )
        model.to(device)
        model.eval()
        model.set_ddpm_inference_steps(num_steps=5)
        log.info("VibeVoice model loaded successfully on %s!", device)
    except Exception as e:
        log.error("Failed to load VibeVoice model: %s", e)
        # model/processor remain None; /health will report not-ready

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

    VibeVoice 0.5B uses precomputed .pt voice presets instead of on-the-fly WAV cloning.
    The ref_audio parameter is accepted for API compatibility but is not applied to inference.
    To change voice identity, swap the preset at models/vibevoice/voices/.
    """
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="VBV model is not loaded yet.")

    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    log.info(
        "TTS request: text='%s...', ref_audio=%s",
        text[:50],
        "provided (not used — preset-based)" if ref_audio else "none",
    )

    t0 = time.time()

    voice_preset_path = (
        "/app/models/vibevoice/voices/en-Carter_man.pt"
        if os.path.exists("/app/models/vibevoice")
        else "models/vibevoice/voices/en-Carter_man.pt"
    )

    all_prefilled_outputs = None
    if os.path.exists(voice_preset_path):
        all_prefilled_outputs = torch.load(
            voice_preset_path, map_location=device, weights_only=False
        )
    else:
        log.warning("Voice preset %s not found. Output may be degraded.", voice_preset_path)

    inputs = processor.process_input_with_cached_prompt(
        text=text,
        cached_prompt=all_prefilled_outputs,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to(device)

    # Snapshot prefilled outputs before handing off to thread (avoids mutation race)
    prefilled_copy = (
        copy.deepcopy(all_prefilled_outputs) if all_prefilled_outputs is not None else None
    )

    # Offload blocking CPU/GPU inference to a thread — keeps event loop free
    def _do_generate():
        return model.generate(
            **inputs,
            max_new_tokens=None,
            cfg_scale=1.5,
            tokenizer=processor.tokenizer,
            generation_config={"do_sample": False},
            verbose=False,
            all_prefilled_outputs=prefilled_copy,
        )

    outputs = await asyncio.to_thread(_do_generate)

    speech_output = outputs.speech_outputs[0]
    audio_output = speech_output.cpu().numpy() if torch.is_tensor(speech_output) else speech_output

    sample_rate = 24000
    if len(audio_output.shape) > 1 and audio_output.shape[0] == 1:
        audio_output = audio_output.squeeze(0)

    buffer = io.BytesIO()
    sf.write(buffer, audio_output, sample_rate, format="WAV")
    buffer.seek(0)

    elapsed = time.time() - t0
    log.info("TTS done in %.3fs. Audio: %.2fs", elapsed, len(audio_output) / sample_rate)

    return Response(content=buffer.read(), media_type="audio/wav")


@app.get("/health")
async def health_check():
    """Returns healthy only when model+processor are fully loaded."""
    ready = model is not None and processor is not None
    return JSONResponse(
        content={
            "status": "healthy" if ready else "loading",
            "service": "vbv",
            "device": device,
            "model_loaded": ready,
        },
        status_code=200,
    )
