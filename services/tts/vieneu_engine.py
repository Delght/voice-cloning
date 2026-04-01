"""VieNeu-TTS Engine — Vietnamese TTS model loading and synthesis.

Extracted from scripts/vieneu_infer.py for use as a long-running service.
"""

import io
import logging

import numpy as np
import soundfile as sf

log = logging.getLogger(__name__)

SAMPLE_RATE = 24_000


class VieNeuEngine:
    """Wraps VieNeu-TTS for repeated synthesis calls.

    Supports optional voice cloning from a short (3-5s) reference audio.
    """

    def __init__(self, mode: str = "turbo") -> None:
        self.mode = mode
        self._engine = None

    def load(self) -> None:
        from vieneu import Vieneu

        log.info("Loading VieNeu-TTS | mode: %s", self.mode)
        self._engine = Vieneu(mode=self.mode)
        log.info("VieNeu-TTS engine ready.")

    def unload(self) -> None:
        self._engine = None
        log.info("VieNeu-TTS engine unloaded.")

    @property
    def ready(self) -> bool:
        return self._engine is not None

    def synthesize(
        self,
        text: str,
        ref_audio_bytes: bytes | None = None,
        *,
        temperature: float = 0.4,
        top_k: int = 50,
    ) -> tuple[bytes, int]:
        """Synthesize Vietnamese speech and return (wav_bytes, sample_rate).

        Args:
            text: Text to synthesize (Vietnamese, English, or mixed).
            ref_audio_bytes: Optional raw bytes of reference audio for voice cloning.
            temperature: Sampling temperature.
            top_k: Top-k sampling parameter.

        Returns:
            Tuple of (WAV file bytes, sample_rate).
        """
        if not self.ready:
            raise RuntimeError("VieNeu-TTS engine not loaded.")

        log.info("Synthesizing (VieNeu): text='%s'", text[:60])

        voice = None
        if ref_audio_bytes is not None:
            import tempfile
            from pathlib import Path

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(ref_audio_bytes)
                tmp_path = tmp.name
            try:
                voice = self._engine.encode_reference(tmp_path)
                log.info("Voice embedding encoded from reference audio.")
            finally:
                Path(tmp_path).unlink(missing_ok=True)

        audio: np.ndarray = self._engine.infer(
            text=text,
            voice=voice,
            temperature=temperature,
            top_k=top_k,
        )

        if audio is None or len(audio) == 0:
            raise RuntimeError("No audio generated — check input text.")

        buf = io.BytesIO()
        sf.write(buf, audio, SAMPLE_RATE, format="WAV")
        wav_bytes = buf.getvalue()

        duration = len(audio) / SAMPLE_RATE
        log.info(
            "Generated %.1fs audio @ %dHz (%d bytes)",
            duration,
            SAMPLE_RATE,
            len(wav_bytes),
        )
        return wav_bytes, SAMPLE_RATE
