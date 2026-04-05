"""Normalize uploaded audio to WAV bytes for multipart HTTP (UI + gateway)."""

from __future__ import annotations

import subprocess
from pathlib import Path

_NON_WAV = {".mp3", ".m4a", ".aac", ".ogg", ".flac", ".opus"}


def load_reference_as_wav(path: Path) -> tuple[bytes, str]:
    p = path.resolve()
    if not p.is_file():
        raise FileNotFoundError(str(p))
    if p.suffix.lower() not in _NON_WAV:
        return p.read_bytes(), p.name
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(p),
            "-ac",
            "1",
            "-ar",
            "44100",
            "-c:a",
            "pcm_s16le",
            "-f",
            "wav",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    return result.stdout, p.stem + ".wav"
