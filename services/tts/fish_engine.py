"""fish-speech TTS Engine — model loading and synthesis.

Extracted from scripts/tts_infer.py for use as a long-running service.
"""

import io
import logging
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

FISH_SPEECH_ROOT = Path.home() / "fish-speech"
sys.path.insert(0, str(FISH_SPEECH_ROOT))

log = logging.getLogger(__name__)

MODEL_DIR = Path("models/fish-speech-1.5")
DECODER_CKPT = MODEL_DIR / "firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
DECODER_CONFIG = "firefly_gan_vq"

TOKENS_PER_WORD = 10
MIN_TOKENS = 200
MAX_TOKENS = 2048


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


class FishSpeechEngine:
    """Wraps fish-speech for repeated TTS calls.

    The model loads once at startup (~10-30s, 1.5GB RAM) and stays in memory.
    """

    def __init__(self) -> None:
        self._engine = None

    def load(self) -> None:
        from fish_speech.inference_engine import TTSInferenceEngine
        from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
        from fish_speech.models.vqgan.inference import load_model as load_decoder

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
        """Synthesize speech and return (wav_bytes, sample_rate).

        Args:
            text: Text to synthesize.
            ref_audio_bytes: Raw bytes of the reference audio file.
            ref_text: Transcript of the reference audio (critical for cloning).
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            repetition_penalty: Penalize repeated tokens.
            chunk_length: Characters per iterative chunk.

        Returns:
            Tuple of (WAV file bytes, sample_rate).
        """
        if not self.ready:
            raise RuntimeError("fish-speech engine not loaded.")

        from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest

        max_tokens = _estimate_max_tokens(text)
        log.info(
            "Synthesizing: text='%s' | max_tokens=%d | ref_text='%s'",
            text[:60],
            max_tokens,
            ref_text[:40],
        )

        req = ServeTTSRequest(
            text=text,
            references=[ServeReferenceAudio(audio=ref_audio_bytes, text=ref_text)],
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

        for result in self._engine.inference(req):
            if result.code == "error":
                raise RuntimeError(f"fish-speech engine error: {result.error}")
            if result.code in ("segment", "final") and isinstance(result.audio, tuple):
                sample_rate, chunk = result.audio
                audio_chunks.append(chunk)

        if not audio_chunks:
            raise RuntimeError("No audio generated — check input text or reference.")

        full_audio = np.concatenate(audio_chunks)
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
