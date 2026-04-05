from __future__ import annotations

from pathlib import Path


def default_fish_ref_path() -> Path:
    root = Path(__file__).resolve().parent.parent
    return root / "audio" / "reference" / "phuong_anh.wav"
