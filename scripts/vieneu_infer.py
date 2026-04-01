#!/usr/bin/env python3
"""vieneu_infer.py — Phase 2: Vietnamese TTS Inference

Generates Vietnamese speech from text using VieNeu-TTS,
with optional zero-shot voice cloning from a short reference audio (3–5s).

Supports code-switching between Vietnamese and English in the same sentence.

Usage:
    python scripts/vieneu_infer.py --text "Xin chào, hôm nay trời đẹp quá!"
    python scripts/vieneu_infer.py --text "Xin chào" --ref data/chunks/speech_chunk_0001.wav
    python scripts/vieneu_infer.py --text "Hệ thống sử dụng API gateway" --output data/output_vi.wav
"""

import argparse
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_engine(mode: str = "turbo"):
    """Initialize VieNeu-TTS engine."""
    from vieneu import Vieneu

    log.info(f"Loading VieNeu-TTS | mode: {mode}")
    engine = Vieneu(mode=mode)
    log.info("VieNeu-TTS engine ready.")
    return engine


def synthesize(
    engine,
    text: str,
    ref_audio_path: Path | None,
    output_path: Path,
    *,
    temperature: float = 0.4,
    top_k: int = 50,
) -> None:
    """Run inference and save the result as a WAV file."""

    log.info(f"Text       : {text[:80]}{'...' if len(text) > 80 else ''}")

    voice = None
    if ref_audio_path is not None:
        log.info(f"Reference  : {ref_audio_path}")
        voice = engine.encode_reference(str(ref_audio_path))
        log.info("Voice embedding encoded.")

    try:
        audio: np.ndarray = engine.infer(
            text=text,
            voice=voice,
            temperature=temperature,
            top_k=top_k,
        )
    except RuntimeError as e:
        log.error(f"Inference failed (OOM or model error): {e}")
        return

    if audio is None or len(audio) == 0:
        log.error("No audio generated — check input text.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    engine.save(audio, str(output_path))

    sample_rate = 24000  # VieNeu-TTS outputs 24kHz
    duration = len(audio) / sample_rate
    log.info(f"Saved → {output_path} ({duration:.1f}s @ {sample_rate}Hz)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Vietnamese TTS with optional voice cloning via VieNeu-TTS."
    )
    parser.add_argument(
        "--text",
        required=True,
        help="Text to synthesize (Vietnamese, English, or mixed)",
    )
    parser.add_argument(
        "--ref",
        default=None,
        help="Reference audio for voice cloning (3–5s, clear voice)",
    )
    parser.add_argument("--output", default="data/output_vi.wav", help="Output WAV file")
    parser.add_argument(
        "--mode",
        default="turbo",
        choices=["turbo", "standard"],
        help="Engine mode (turbo=CPU/GGUF, standard=PyTorch)",
    )

    parser.add_argument(
        "--temperature", type=float, default=0.4, help="Sampling temperature (0.1–1.0)"
    )
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    args = parser.parse_args()

    ref_path = Path(args.ref) if args.ref else None
    output_path = Path(args.output)

    if ref_path is not None and not ref_path.exists():
        log.error(f"Reference audio not found: {ref_path}")
        return

    engine = load_engine(mode=args.mode)
    synthesize(
        engine,
        text=args.text,
        ref_audio_path=ref_path,
        output_path=output_path,
        temperature=args.temperature,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
