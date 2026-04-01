"""VieNeu-TTS Engine — Vietnamese TTS model loading and synthesis.

Uses **standard** mode which supports proper voice cloning via
ref_audio + ref_text (phonemized and encoded into the LLM prompt).

Turbo mode only uses a 128-dim speaker embedding and ignores ref_text
entirely — that's why cloning quality was poor. Standard mode passes
full speech token codes + phonemized ref_text into the prompt, matching
the official VieNeu SDK documentation.

Run:
    uvicorn services.tts.app:app --port 8002
"""

from __future__ import annotations

import io
import logging
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

log = logging.getLogger(__name__)

SAMPLE_RATE = 24_000


class VieNeuEngine:
    """Wraps VieNeu-TTS (standard mode) for repeated synthesis calls.

    Standard mode loads:
      - Backbone: GGUF quantised LLM (via llama-cpp-python, CPU)
      - Codec:    DistillNeuCodec (PyTorch, CPU)

    Voice cloning flow (standard mode):
      1. ref_audio → codec.encode_code() → speech token codes
      2. ref_text  → phonemize() → ref phonemes
      3. Prompt = ref_phonemes + input_phonemes + speech_token_codes
      4. LLM generates new speech tokens conditioned on all of the above
      5. codec.decode_code() → waveform

    This is fundamentally different from turbo mode which only uses a
    128-dim embedding and ignores ref_text.
    """

    def __init__(
        self,
        *,
        backbone_device: str = "cpu",
        codec_device: str = "cpu",
    ) -> None:
        self._backbone_device = backbone_device
        self._codec_device = codec_device
        self._engine = None

    def load(self) -> None:
        from vieneu import Vieneu

        log.info(
            "Loading VieNeu-TTS | mode=standard | backbone=%s | codec=%s",
            self._backbone_device,
            self._codec_device,
        )
        self._engine = Vieneu(
            mode="standard",
            backbone_device=self._backbone_device,
            codec_device=self._codec_device,
        )
        log.info("VieNeu-TTS (standard) engine ready.")

    def unload(self) -> None:
        if self._engine is not None:
            self._engine.close()
        self._engine = None
        log.info("VieNeu-TTS engine unloaded.")

    @property
    def ready(self) -> bool:
        return self._engine is not None

    def synthesize(
        self,
        text: str,
        ref_audio_bytes: bytes | None = None,
        ref_audio_filename: str | None = None,
        ref_text: str | None = None,
        *,
        temperature: float = 1.0,
        top_k: int = 50,
        max_chars: int = 256,
    ) -> tuple[bytes, int]:
        """Synthesize Vietnamese speech and return (wav_bytes, sample_rate).

        Args:
            text: Text to synthesize (Vietnamese, English, or mixed).
            ref_audio_bytes: Raw bytes of reference audio (3-5s) for cloning.
            ref_text: Exact transcript of the reference audio.
                      Critical for cloning quality — must match what is spoken.
            temperature: Sampling temperature (standard mode default = 1.0).
            top_k: Top-k sampling parameter.

        Returns:
            Tuple of (WAV file bytes, sample_rate).
        """
        if not self.ready:
            raise RuntimeError("VieNeu-TTS engine not loaded.")

        log.info("Synthesizing (VieNeu): text='%s'", text[:60])

        audio: np.ndarray

        if ref_audio_bytes is not None:
            suffix = ".wav"
            if ref_audio_filename:
                try:
                    s = Path(ref_audio_filename).suffix
                    if s:
                        suffix = s
                except Exception:
                    pass

            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(ref_audio_bytes)
                tmp_path = tmp.name

            try:
                kwargs: dict = {
                    "text": text,
                    "ref_audio": tmp_path,
                    "temperature": temperature,
                    "top_k": top_k,
                    "max_chars": max_chars,
                }
                if ref_text:
                    kwargs["ref_text"] = ref_text
                    log.info(
                        "Cloning with ref_audio + ref_text (%d chars)",
                        len(ref_text),
                    )
                else:
                    log.warning(
                        "ref_text not provided — cloning quality "
                        "will be degraded. Provide the exact transcript "
                        "of the reference audio for best results."
                    )

                audio = self._engine.infer(**kwargs)
            finally:
                Path(tmp_path).unlink(missing_ok=True)
        else:
            audio = self._engine.infer(
                text=text,
                temperature=temperature,
                top_k=top_k,
                max_chars=max_chars,
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
