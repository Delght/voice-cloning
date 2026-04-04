#!/usr/bin/env python3
"""rvc_infer.py: Voice Conversion with Applio RVC

Converts the voice in an existing audio file to a target voice
using a trained .pth model, preserving the original prosody and emotion.

Usage:
    python scripts/rvc_infer.py --input data/output.wav --model models/rvc/target.pth
    python scripts/rvc_infer.py --input data/output.wav --model models/rvc/target.pth \
        --index models/rvc/target.index --pitch 0 --f0-method rmvpe
"""

import argparse
import logging
import sys
from pathlib import Path

APPLIO_ROOT = Path.home() / "applio"
sys.path.insert(0, str(APPLIO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def convert(
    input_path: Path,
    output_path: Path,
    model_path: Path,
    index_path: str,
    *,
    pitch: int = 0,
    f0_method: str = "rmvpe",
    index_rate: float = 0.75,
    protect: float = 0.5,
    clean_audio: bool = False,
    clean_strength: float = 0.5,
    export_format: str = "WAV",
) -> None:
    """Run Applio voice conversion on a single audio file."""
    import os

    original_cwd = os.getcwd()
    os.chdir(str(APPLIO_ROOT))

    try:
        from rvc.infer.infer import VoiceConverter

        log.info(f"Input audio : {input_path}")
        log.info(f"Voice model : {model_path}")
        log.info(f"Index file  : {index_path or '(none)'}")
        log.info(f"Pitch shift : {pitch} semitones | F0 method: {f0_method}")

        vc = VoiceConverter()
        vc.convert_audio(
            audio_input_path=str(input_path),
            audio_output_path=str(output_path),
            model_path=str(model_path),
            index_path=index_path,
            pitch=pitch,
            f0_method=f0_method,
            index_rate=index_rate,
            protect=protect,
            clean_audio=clean_audio,
            clean_strength=clean_strength,
            export_format=export_format,
        )
        log.info(f"Saved to {output_path}")

    except RuntimeError as e:
        log.error(f"Voice conversion failed (OOM or model error): {e}")
    finally:
        os.chdir(original_cwd)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Voice conversion using Applio RVC - swap voice identity while keeping prosody."
    )
    parser.add_argument("--input", required=True, help="Input audio file to convert")
    parser.add_argument("--model", required=True, help="Path to .pth voice model")
    parser.add_argument(
        "--index", default="", help="Path to .index file (optional, improves quality)"
    )
    parser.add_argument("--output", default="data/output_rvc.wav", help="Output file path")

    parser.add_argument(
        "--pitch",
        type=int,
        default=0,
        help="Pitch shift in semitones (e.g. +12 = 1 octave up, -12 = 1 octave down)",
    )
    parser.add_argument(
        "--f0-method",
        default="rmvpe",
        help="F0 extraction method: rmvpe (default), crepe, fcpe, swift",
    )
    parser.add_argument(
        "--index-rate",
        type=float,
        default=0.75,
        help="Index matching rate (0.0–1.0, higher = more like target voice)",
    )
    parser.add_argument(
        "--protect",
        type=float,
        default=0.5,
        help="Protect consonants/breathing (0.0–1.0, higher = more protection)",
    )
    parser.add_argument(
        "--clean-audio",
        action="store_true",
        help="Apply noise reduction before conversion",
    )
    parser.add_argument(
        "--clean-strength",
        type=float,
        default=0.5,
        help="Noise reduction strength (0.0–1.0)",
    )
    parser.add_argument(
        "--export-format", default="WAV", help="Output format: WAV, MP3, FLAC, OGG, M4A"
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    model_path = Path(args.model).resolve()
    output_path = Path(args.output).resolve()
    index_path = str(Path(args.index).resolve()) if args.index else ""

    if not input_path.exists():
        log.error(f"Input audio not found: {input_path}")
        return

    if not model_path.exists():
        log.error(f"Voice model not found: {model_path}")
        return

    if not APPLIO_ROOT.exists():
        log.error(
            f"Applio not found at {APPLIO_ROOT}. "
            "Run: git clone https://github.com/IAHispano/Applio.git ~/applio"
        )
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    convert(
        input_path=input_path,
        output_path=output_path,
        model_path=model_path,
        index_path=index_path,
        pitch=args.pitch,
        f0_method=args.f0_method,
        index_rate=args.index_rate,
        protect=args.protect,
        clean_audio=args.clean_audio,
        clean_strength=args.clean_strength,
        export_format=args.export_format,
    )


if __name__ == "__main__":
    main()
