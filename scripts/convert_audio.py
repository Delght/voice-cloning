"""Batch-convert audio files (mp3, m4a) to mono 44.1 kHz WAV.

Usage:
    python scripts/convert_audio.py --input audio/input --output audio/output

Intended for preprocessing reference audio before TTS or RVC inference.
"""

import argparse
import subprocess
from pathlib import Path


def convert_audio(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_files = sorted(input_dir.glob("*.mp3")) + sorted(input_dir.glob("*.m4a"))
    if not audio_files:
        print(f"No mp3/m4a files found in {input_dir}")
        return

    errors: list[str] = []
    for f in audio_files:
        out = output_dir / f"{f.stem}.wav"
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(f),
            "-ac",
            "1",
            "-ar",
            "44100",
            "-c:a",
            "pcm_s16le",
            str(out),
        ]
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            print(f"  {f.name} to {out.name}")
        except subprocess.CalledProcessError:
            print(f"  ERROR: {f.name}")
            errors.append(f.name)

    ok = len(audio_files) - len(errors)
    print(f"\nDone: {ok}/{len(audio_files)} converted to {output_dir}")
    if errors:
        print("Failed:", ", ".join(errors))


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch convert mp3/m4a to WAV")
    parser.add_argument(
        "--input", default="audio/input", help="Input directory (default: audio/input)"
    )
    parser.add_argument(
        "--output", default="audio/output", help="Output directory (default: audio/output)"
    )
    args = parser.parse_args()

    convert_audio(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
