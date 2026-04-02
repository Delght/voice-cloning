"""fish-speech TTS Engine — model loading and synthesis.

Extracted from scripts/tts_infer.py for use as a long-running service.
"""

import gc
import io
import logging
import os
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

FISH_SPEECH_ROOT = Path(os.environ.get("FISH_SPEECH_ROOT", str(Path.home() / "fish-speech")))
sys.path.insert(0, str(FISH_SPEECH_ROOT))

log = logging.getLogger(__name__)

MODEL_DIR = Path(os.environ.get("FISH_MODEL_DIR", "models/fish-speech-1.5"))
DECODER_CKPT = MODEL_DIR / "firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
DECODER_CONFIG = "firefly_gan_vq"

TOKENS_PER_WORD = 10
MIN_TOKENS = 200
MAX_TOKENS = 4096

SILENCE_BETWEEN_CHUNKS_S = 0.2

# Heuristic guard against occasional early-stop truncation.
# If a generated chunk is much shorter than expected (by word count),
# retry once with more conservative sampling.
MIN_SECONDS_PER_WORD = 0.12
TRUNCATION_RATIO_THRESHOLD = 0.55

# Retry sampling parameters — applied when truncation is detected.
RETRY_MAX_TOKENS_FACTOR = 1.5  # scale up token budget
RETRY_TEMPERATURE_FLOOR = 0.3  # floor to keep output coherent
RETRY_TEMPERATURE_FACTOR = 0.75  # pull temperature toward conservative
RETRY_TOP_P_CEILING = 0.6  # cap top_p to reduce randomness


def _get_device() -> str:
    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


def _estimate_max_tokens(text: str) -> int:
    word_count = len(text.split())
    estimated = word_count * TOKENS_PER_WORD
    return max(MIN_TOKENS, min(estimated, MAX_TOKENS))


def _split_sentences(text: str) -> list[str]:
    """Split text into sentence-like units, keeping punctuation."""
    text = text.strip()
    if not text:
        return []

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)

    parts: list[str] = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        segs = re.split(r"(?<=[.!?])\s+", para)
        parts.extend([s.strip() for s in segs if s and s.strip()])

    return parts


def _pack_chunks(sentences: list[str], max_chars: int) -> list[str]:
    """Pack sentence units into chunks up to max_chars (best-effort)."""
    if max_chars <= 0:
        return [s for s in sentences if s]

    chunks: list[str] = []
    buf = ""
    for s in sentences:
        if not buf:
            buf = s
            continue

        candidate = f"{buf} {s}"
        if len(candidate) <= max_chars:
            buf = candidate
        else:
            chunks.append(buf)
            buf = s

    if buf:
        chunks.append(buf)

    # If any chunk is still too long, hard-split by commas/spaces.
    final: list[str] = []
    for c in chunks:
        if len(c) <= max_chars:
            final.append(c)
            continue

        pieces = re.split(r"(?<=,)\s+|\s+", c)
        b = ""
        for p in pieces:
            p = p.strip()
            if not p:
                continue
            if not b:
                b = p
                continue
            cand = f"{b} {p}"
            if len(cand) <= max_chars:
                b = cand
            else:
                final.append(b)
                b = p
        if b:
            final.append(b)

    return [c.strip() for c in final if c and c.strip()]


def _expected_min_duration_s(text: str) -> float:
    words = len([w for w in text.split() if w.strip()])
    # A very rough lower bound; used only to detect obvious truncation.
    return max(0.35, words * MIN_SECONDS_PER_WORD)


class FishSpeechEngine:
    """Wraps fish-speech for repeated TTS calls."""

    def __init__(self) -> None:
        self._engine = None

    def load(self) -> None:
        # Deferred: fish_speech only exists in the voice_fish conda env.
        # Importing at module level would crash in the default voice env.
        from fish_speech.inference_engine import TTSInferenceEngine  # noqa: PLC0415
        from fish_speech.models.text2semantic.inference import (
            launch_thread_safe_queue,  # noqa: PLC0415
        )
        from fish_speech.models.vqgan.inference import load_model as load_decoder  # noqa: PLC0415

        device = _get_device()
        precision = torch.bfloat16
        log.info("Loading fish-speech | device: %s | precision: bfloat16", device)

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

        self._engine = TTSInferenceEngine(
            llama_queue=llama_queue,
            decoder_model=decoder,
            precision=precision,
            compile=False,
        )
        log.info("fish-speech engine ready.")

    def unload(self) -> None:
        self._engine = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        log.info("fish-speech engine unloaded.")

    @property
    def ready(self) -> bool:
        return self._engine is not None

    def synthesize(
        self,
        text: str,
        ref_audio_bytes: bytes,
        ref_text: str,
        *,
        temperature: float = 0.7,
        top_p: float = 0.7,
        repetition_penalty: float = 1.2,
        chunk_length: int = 200,
    ) -> tuple[bytes, int]:
        """Synthesize speech and return (wav_bytes, sample_rate)."""
        if not self.ready:
            raise RuntimeError("fish-speech engine not loaded.")

        from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest  # noqa: PLC0415

        sample_rate: int = 44100
        full_audio_chunks: list[np.ndarray] = []

        # Text chunking pipeline to avoid truncation ("text loss") on long inputs.
        sentences = _split_sentences(text)
        chunks = _pack_chunks(sentences, max_chars=chunk_length)
        if not chunks:
            raise RuntimeError("Empty text after preprocessing.")

        log.info(
            "Synthesizing: %d chunks | chunk_length=%d | ref_text='%s'",
            len(chunks),
            chunk_length,
            ref_text[:40],
        )

        def _infer_once(
            *,
            t: str,
            max_new_tokens: int,
            temp: float,
            p: float,
        ) -> tuple[np.ndarray, int]:
            req = ServeTTSRequest(
                text=t,
                references=[ServeReferenceAudio(audio=ref_audio_bytes, text=ref_text)],
                format="wav",
                streaming=False,
                max_new_tokens=max_new_tokens,
                chunk_length=chunk_length,
                top_p=p,
                temperature=temp,
                repetition_penalty=repetition_penalty,
            )

            audio_chunks: list[np.ndarray] = []
            sr: int = 44100
            for result in self._engine.inference(req):
                if result.code == "error":
                    raise RuntimeError(f"fish-speech engine error: {result.error}")
                if result.code in ("segment", "final") and isinstance(result.audio, tuple):
                    sr, a = result.audio
                    audio_chunks.append(a)

            if not audio_chunks:
                raise RuntimeError(
                    "No audio generated for a chunk — check input text or reference."
                )

            return np.concatenate(audio_chunks), sr

        for idx, chunk_text in enumerate(chunks):
            max_tokens = _estimate_max_tokens(chunk_text)
            log.info(
                "Chunk %d/%d: chars=%d | max_tokens=%d | text='%s'",
                idx + 1,
                len(chunks),
                len(chunk_text),
                max_tokens,
                chunk_text[:60],
            )

            chunk_audio, sample_rate = _infer_once(
                t=chunk_text,
                max_new_tokens=max_tokens,
                temp=temperature,
                p=top_p,
            )

            # Retry once if the chunk looks obviously truncated.
            dur_s = len(chunk_audio) / sample_rate
            expected_min_s = _expected_min_duration_s(chunk_text)
            if dur_s < expected_min_s * TRUNCATION_RATIO_THRESHOLD:
                log.warning(
                    "Chunk %d/%d seems truncated (%.2fs < %.2fs). Retrying with safer sampling.",
                    idx + 1,
                    len(chunks),
                    dur_s,
                    expected_min_s,
                )
                chunk_audio, sample_rate = _infer_once(
                    t=chunk_text,
                    max_new_tokens=min(MAX_TOKENS, int(max_tokens * RETRY_MAX_TOKENS_FACTOR)),
                    temp=max(RETRY_TEMPERATURE_FLOOR, temperature * RETRY_TEMPERATURE_FACTOR),
                    p=min(RETRY_TOP_P_CEILING, top_p),
                )
            full_audio_chunks.append(chunk_audio)

            if idx < len(chunks) - 1:
                silence = np.zeros(int(sample_rate * SILENCE_BETWEEN_CHUNKS_S), dtype=np.float32)
                full_audio_chunks.append(silence)

        full_audio = np.concatenate(full_audio_chunks)
        buf = io.BytesIO()
        sf.write(buf, full_audio, sample_rate, format="WAV")
        wav_bytes = buf.getvalue()

        duration = len(full_audio) / sample_rate
        log.info(
            "Generated %.1fs audio @ %dHz (%d bytes)",
            duration,
            sample_rate,
            len(wav_bytes),
        )
        return wav_bytes, sample_rate
