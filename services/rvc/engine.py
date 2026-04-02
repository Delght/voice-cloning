"""RVC Engine — Applio voice conversion model loading and inference."""

import gc
import logging
import os
import sys
import tempfile
from pathlib import Path

APPLIO_ROOT = Path(os.environ.get("APPLIO_ROOT", str(Path.home() / "applio")))
sys.path.insert(0, str(APPLIO_ROOT))

log = logging.getLogger(__name__)

MODELS_DIR = Path(os.environ.get("RVC_MODELS_DIR", "models/rvc"))


class RVCEngine:
    """Wraps Applio VoiceConverter for repeated voice conversion calls."""

    def __init__(self) -> None:
        self._vc = None

    def load(self) -> None:
        original_cwd = os.getcwd()
        os.chdir(str(APPLIO_ROOT))
        try:
            from rvc.infer.infer import VoiceConverter  # noqa: PLC0415

            self._vc = VoiceConverter()
            log.info("Applio VoiceConverter loaded.")
        finally:
            os.chdir(original_cwd)

    def unload(self) -> None:
        self._vc = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except ImportError:
            pass
        log.info("Applio VoiceConverter unloaded.")

    @property
    def ready(self) -> bool:
        return self._vc is not None

    def list_models(self) -> list[str]:
        """List available .pth voice models in models/rvc/."""
        if not MODELS_DIR.exists():
            return []
        return [p.stem for p in MODELS_DIR.glob("*.pth")]

    def resolve_model_path(self, model_name: str) -> Path:
        """Resolve model name to .pth file path.

        Accepts:
            - A bare name like "target" -> models/rvc/target.pth
            - A full/relative path like "models/rvc/target.pth"
        """
        path = Path(model_name)
        if path.suffix == ".pth" and path.exists():
            return path.resolve()

        candidate = MODELS_DIR / f"{model_name}.pth"
        if candidate.exists():
            return candidate.resolve()

        raise FileNotFoundError(
            f"Voice model '{model_name}' not found. "
            f"Available: {self.list_models()}. "
            f"Place .pth files in {MODELS_DIR}/"
        )

    def convert(
        self,
        audio_bytes: bytes,
        model_name: str,
        index_path: str = "",
        *,
        pitch: int = 0,
        f0_method: str = "rmvpe",
        index_rate: float = 0.75,
        protect: float = 0.5,
        clean_audio: bool = False,
        clean_strength: float = 0.5,
    ) -> bytes:
        """Convert voice in audio and return WAV bytes."""
        if not self.ready:
            raise RuntimeError("RVC engine not loaded.")

        model_path = self.resolve_model_path(model_name)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as in_tmp:
            in_tmp.write(audio_bytes)
            input_path = in_tmp.name

        output_path = tempfile.mktemp(suffix=".wav")

        original_cwd = os.getcwd()
        os.chdir(str(APPLIO_ROOT))

        try:
            log.info(
                "Converting: model=%s | pitch=%d | f0=%s",
                model_path.name,
                pitch,
                f0_method,
            )
            self._vc.convert_audio(
                audio_input_path=input_path,
                audio_output_path=output_path,
                model_path=str(model_path),
                index_path=index_path,
                pitch=pitch,
                f0_method=f0_method,
                index_rate=index_rate,
                protect=protect,
                clean_audio=clean_audio,
                clean_strength=clean_strength,
                export_format="WAV",
            )

            result_path = Path(output_path)
            if not result_path.exists():
                raise RuntimeError("Conversion produced no output file.")

            wav_bytes = result_path.read_bytes()
            log.info("Conversion complete: %d bytes", len(wav_bytes))
            return wav_bytes

        finally:
            os.chdir(original_cwd)
            Path(input_path).unlink(missing_ok=True)
            Path(output_path).unlink(missing_ok=True)
