#!/usr/bin/env python3
"""tts_infer.py: Text-to-Speech Inference

Generates speech from text using fish-speech, cloning the voice
from a short reference audio file (zero-shot, no training needed).

Usage:
    python scripts/tts_infer.py --text "Hello world" --ref data/chunks/speech_chunk_0001.wav \
        --ref-text "America is a cutting edge economy, but our immigration system is stuck "\
        "in the past."
    python scripts/tts_infer.py --text "Xin chào" --ref data/chunks/speech_chunk_0001.wav \
        --ref-text "America is a cutting edge economy" --output data/output.wav
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

FISH_SPEECH_ROOT = Path.home() / "fish-speech"
sys.path.insert(0, str(FISH_SPEECH_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

MODEL_DIR = Path("models/fish-speech-1.5")
DECODER_CKPT = MODEL_DIR / "firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
DECODER_CONFIG = "firefly_gan_vq"

# fish-speech decoder runs at 21 tokens/sec.
# ~150 words/min in English = 2.5 words/sec -> ~8 tokens per word.
TOKENS_PER_WORD = 10
MIN_TOKENS = 200
MAX_TOKENS = 2048


def get_device() -> str:
    """Detect best available device - never hardcode."""
    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


def load_engine():
    """Load fish-speech LLaMA + decoder models and return the inference engine."""
    from fish_speech.inference_engine import TTSInferenceEngine
    from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
    from fish_speech.models.vqgan.inference import load_model as load_decoder

    device = get_device()
    precision = torch.bfloat16
    log.info(f"Loading models | device: {device} | precision: bfloat16")

    llama_queue = launch_thread_safe_queue(
        checkpoint_path=str(MODEL_DIR),
        device=device,
        precision=precision,
        compile=False,
    )
    log.info("LLaMA model loaded.")

    decoder = load_decoder(
        config_name=DECODER_CONFIG,
        checkpoint_path=str(DECODER_CKPT),
        device=device,
    )
    log.info("Decoder (Firefly GAN) loaded.")

    return TTSInferenceEngine(
        llama_queue=llama_queue,
        decoder_model=decoder,
        precision=precision,
        compile=False,
    )


def estimate_max_tokens(text: str) -> int:
    """Scale token budget based on input text length to avoid trailing noise."""
    word_count = len(text.split())
    estimated = word_count * TOKENS_PER_WORD
    return max(MIN_TOKENS, min(estimated, MAX_TOKENS))


def synthesize(
    engine,
    text: str,
    ref_audio_path: Path,
    ref_text: str,
    output_path: Path,
    *,
    temperature: float = 0.7,
    top_p: float = 0.7,
    repetition_penalty: float = 1.2,
    chunk_length: int = 200,
) -> None:
    """Run inference and save the result as a WAV file."""
    from fish_speech.utils.file import audio_to_bytes
    from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest

    max_tokens = estimate_max_tokens(text)

    log.info(f"Reference audio : {ref_audio_path}")
    log.info(f"Reference text  : {ref_text[:80]}{'...' if len(ref_text) > 80 else ''}")
    log.info(f"Synth text      : {text[:80]}{'...' if len(text) > 80 else ''}")
    log.info(f"max_new_tokens  : {max_tokens} (auto-scaled from {len(text.split())} words)")

    req = ServeTTSRequest(
        text=text,
        references=[
            ServeReferenceAudio(
                audio=audio_to_bytes(str(ref_audio_path)),
                text=ref_text,
            )
        ],
        format="wav",
        streaming=False,
        max_new_tokens=max_tokens,
        chunk_length=chunk_length,
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
    )

    audio_chunks: list[np.ndarray] = []
    sample_rate: int = 44100

    try:
        for result in engine.inference(req):
            if result.code == "error":
                log.error(f"Engine error: {result.error}")
                return
            if result.code in ("segment", "final") and isinstance(result.audio, tuple):
                sample_rate, chunk = result.audio
                audio_chunks.append(chunk)
    except RuntimeError as e:
        log.error(f"Inference failed (OOM or model error): {e}")
        return

    if not audio_chunks:
        log.error("No audio generated - check input text or reference audio.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    full_audio = np.concatenate(audio_chunks)
    sf.write(str(output_path), full_audio, sample_rate)
    duration = len(full_audio) / sample_rate
    log.info(f"Saved to {output_path} ({duration:.1f}s @ {sample_rate}Hz)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-shot voice cloning TTS with fish-speech.")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument(
        "--ref",
        required=True,
        help="Reference audio (10–30s, clear voice, no BG noise)",
    )
    parser.add_argument(
        "--ref-text",
        default="",
        help="Transcript of the reference audio - critical for cloning quality",
    )
    parser.add_argument("--output", default="data/output.wav", help="Output WAV file")

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (0.1–1.0, lower = more deterministic)",
    )
    parser.add_argument(
        "--top-p", type=float, default=0.7, help="Nucleus sampling threshold (0.1–1.0)"
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.2,
        help="Penalize repeated tokens (0.9–2.0)",
    )
    parser.add_argument(
        "--chunk-length",
        type=int,
        default=200,
        help="Chars per iterative chunk (100–300)",
    )
    args = parser.parse_args()

    ref_path = Path(args.ref)
    output_path = Path(args.output)

    if not ref_path.exists():
        log.error(f"Reference audio not found: {ref_path}")
        return

    if not MODEL_DIR.exists():
        log.error("Missing fish-speech dependency. Check documentation for setup instructions.")
        return

    if not args.ref_text:
        log.warning(
            "No --ref-text provided. Voice cloning quality will be degraded. "
            "Pass the transcript of your reference audio for best results."
        )

    engine = load_engine()
    synthesize(
        engine,
        text=args.text,
        ref_audio_path=ref_path,
        ref_text=args.ref_text,
        output_path=output_path,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        chunk_length=args.chunk_length,
    )


if __name__ == "__main__":
    main()
