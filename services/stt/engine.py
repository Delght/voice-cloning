"""STT Engine — Whisper model loading and transcription.

Extracted from scripts/chunk_audio.py for use as a long-running service.
The model loads once at startup and handles many requests.
"""

import io
import logging

import librosa
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

log = logging.getLogger(__name__)

SAMPLE_RATE = 24_000


class STTEngine:
    """Wraps faster-whisper for repeated transcription calls.

    The model loads once at startup (~500MB–1.5GB in RAM) and stays in memory
    to serve all incoming requests.
    """

    def __init__(self, model_size: str = "large-v3") -> None:
        self.model_size = model_size
        self._model: WhisperModel | None = None

    def load(self) -> None:
        """Load the Whisper model into memory. Call once at service startup."""
        log.info("Loading Whisper model '%s' (CPU/int8)...", self.model_size)
        self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
        log.info("Whisper model loaded.")

    def unload(self) -> None:
        """Release model from memory."""
        self._model = None
        log.info("Whisper model unloaded.")

    @property
    def ready(self) -> bool:
        return self._model is not None

    def transcribe(
        self,
        audio_bytes: bytes,
        beam_size: int = 5,
        language: str | None = None,
    ) -> dict:
        """Transcribe raw audio bytes and return structured result.

        Args:
            audio_bytes: Raw audio file content (wav, mp3, flac, etc.)
            beam_size: Beam search width (higher = more accurate, slower)

        Returns:
            Dict with "text", "language", and "segments" keys.
        """
        if not self.ready:
            raise RuntimeError("STT engine not loaded. Server still starting up.")

        buf = io.BytesIO(audio_bytes)
        try:
            audio_data, sr = sf.read(buf, dtype="float32")
        except Exception:
            audio_data, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)

        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        if sr != SAMPLE_RATE:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=SAMPLE_RATE)

        segments_raw, info = self._model.transcribe(
            audio_data,
            beam_size=beam_size,
            language=language,
            task="transcribe",
        )

        segments = []
        full_text_parts = []
        for seg in segments_raw:
            segments.append(
                {
                    "start": round(seg.start, 3),
                    "end": round(seg.end, 3),
                    "text": seg.text.strip(),
                }
            )
            full_text_parts.append(seg.text.strip())

        return {
            "text": " ".join(full_text_parts),
            "language": info.language,
            "language_probability": round(info.language_probability, 3),
            "duration": round(info.duration, 2),
            "segments": segments,
        }
