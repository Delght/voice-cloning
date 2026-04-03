"""VieNeu-TTS Engine: Vietnamese TTS model loading and synthesis."""

from __future__ import annotations

import gc
import io
import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

log = logging.getLogger(__name__)

SAMPLE_RATE = 24_000

# VieNeu backbone runs via llama-cpp-python (CPU only).
# Codec (NeuCodec) supports cuda - set VIENEU_DEVICE=cuda on Nvidia cloud.
_VIENEU_DEVICE = os.environ.get("VIENEU_DEVICE", "cpu")


class VieNeuEngine:
    """Wraps VieNeu-TTS (standard mode) for repeated synthesis calls."""

    def __init__(
        self,
        *,
        backbone_device: str = "cpu",
        codec_device: str = _VIENEU_DEVICE,
    ) -> None:
        self._backbone_device = backbone_device
        self._codec_device = codec_device
        self._engine = None

    def load(self) -> None:
        from vieneu import Vieneu  # noqa: PLC0415

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
        gc.collect()
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
        """Synthesize Vietnamese speech and return (wav_bytes, sample_rate)."""
        if not self.ready:
            raise RuntimeError("VieNeu-TTS engine not loaded.")

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
                else:
                    log.warning(
                        "ref_text not provided - cloning quality "
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
            raise RuntimeError("No audio generated - check input text.")

        buf = io.BytesIO()
        sf.write(buf, audio, SAMPLE_RATE, format="WAV")
        wav_bytes = buf.getvalue()
        return wav_bytes, SAMPLE_RATE
