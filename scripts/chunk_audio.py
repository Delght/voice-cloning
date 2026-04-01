#!/usr/bin/env python3
"""chunk_audio.py — Phase 1 Data Preparation Script

Splits a long audio file into 5–15s chunks at silence boundaries,
transcribes each chunk with faster-whisper, and exports a transcript.csv.

Usage:
    python scripts/chunk_audio.py --input data/raw/audio.wav
    python scripts/chunk_audio.py --input data/raw/audio.wav --model medium --top-db 35
"""

import argparse
import csv
import logging
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

DEFAULT_SAMPLE_RATE = 24_000  # Hz — high-fidelity for voice cloning
MIN_CHUNK_DURATION = 5.0  # seconds
MAX_CHUNK_DURATION = 15.0  # seconds
SILENCE_TOP_DB = 30  # dB below peak → treated as silence
MAX_SILENCE_GAP = 0.5  # seconds — merge speech bursts closer than this

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_audio(path: Path, sr: int = DEFAULT_SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Load audio file and resample to target sample rate (mono)."""
    log.info(f"Loading: {path}")
    audio, sample_rate = librosa.load(str(path), sr=sr, mono=True)
    duration = len(audio) / sample_rate
    log.info(f"Duration: {duration:.1f}s | Sample rate: {sample_rate}Hz | Samples: {len(audio):,}")
    return audio, sample_rate


def detect_speech_intervals(
    audio: np.ndarray,
    sr: int,
    top_db: int = SILENCE_TOP_DB,
    max_gap: float = MAX_SILENCE_GAP,
) -> list[tuple[float, float]]:
    """
    Detect non-silent regions and merge nearby ones.

    Args:
        audio:   numpy float32 array
        sr:      sample rate
        top_db:  silence threshold — regions quieter than (peak - top_db dB) = silence
        max_gap: gaps smaller than this (seconds) between speech bursts get merged

    Returns:
        List of (start_sec, end_sec) tuples representing speech regions.
    """
    raw_intervals = librosa.effects.split(audio, top_db=top_db)

    if len(raw_intervals) == 0:
        log.warning("No speech detected. Try lowering --top-db.")
        return []

    # Merge intervals separated by a small gap
    gap_samples = int(max_gap * sr)
    merged: list[list[int]] = [raw_intervals[0].tolist()]

    for start, end in raw_intervals[1:]:
        if start - merged[-1][1] <= gap_samples:
            merged[-1][1] = int(end)  # extend current region
        else:
            merged.append([int(start), int(end)])

    intervals = [(s / sr, e / sr) for s, e in merged]
    log.info(f"Detected {len(intervals)} speech regions after merging")
    return intervals


def build_chunks(
    intervals: list[tuple[float, float]],
    min_dur: float = MIN_CHUNK_DURATION,
    max_dur: float = MAX_CHUNK_DURATION,
) -> list[tuple[float, float]]:
    """
    Accumulate speech intervals into chunks of min_dur–max_dur seconds.

    Strategy:
    - Keep extending the current chunk while total duration <= max_dur
    - When adding the next interval would exceed max_dur, save current chunk and start a new one
    - Discard chunks shorter than min_dur (usually trailing silence at the end)

    Returns:
        List of (start_sec, end_sec) tuples.
    """
    chunks: list[tuple[float, float]] = []
    chunk_start, chunk_end = intervals[0]

    for start, end in intervals[1:]:
        if end - chunk_start <= max_dur:
            chunk_end = end  # extend current chunk
        else:
            if chunk_end - chunk_start >= min_dur:
                chunks.append((chunk_start, chunk_end))
            chunk_start = start
            chunk_end = end

    # Last chunk
    if chunk_end - chunk_start >= min_dur:
        chunks.append((chunk_start, chunk_end))

    log.info(f"Built {len(chunks)} chunks (target: {min_dur}–{max_dur}s each)")
    return chunks


def export_chunks(
    audio: np.ndarray,
    sr: int,
    chunks: list[tuple[float, float]],
    output_dir: Path,
    stem: str,
) -> list[Path]:
    """Slice audio array and write each chunk as a 24-bit PCM WAV file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_paths: list[Path] = []

    for i, (start, end) in enumerate(chunks, 1):
        filename = f"{stem}_chunk_{i:04d}.wav"
        out_path = output_dir / filename

        start_idx = int(start * sr)
        end_idx = int(end * sr)
        sf.write(str(out_path), audio[start_idx:end_idx], sr, subtype="PCM_24")

        chunk_paths.append(out_path)

    log.info(f"Exported {len(chunk_paths)} WAV chunks → {output_dir}/")
    return chunk_paths


def transcribe_chunks(
    chunk_paths: list[Path],
    model_size: str = "large-v3",
) -> list[str]:
    """
    Transcribe audio chunks with faster-whisper.

    Note: faster-whisper uses CTranslate2 engine which does NOT support MPS.
    It runs on CPU on macOS — this is expected and still fast thanks to INT8 quantization.

    Returns:
        List of transcript strings, one per chunk.
    """
    log.info(f"Loading Whisper model '{model_size}' (first run will download weights)...")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    transcripts: list[str] = []
    for i, path in enumerate(chunk_paths, 1):
        log.info(f"Transcribing [{i}/{len(chunk_paths)}]: {path.name}")
        try:
            segments, _ = model.transcribe(str(path), beam_size=5)
            text = " ".join(seg.text.strip() for seg in segments).strip()
        except RuntimeError as e:
            log.error(f"Transcription failed for {path.name}: {e}")
            text = ""

        transcripts.append(text)
        preview = text[:80] + ("..." if len(text) > 80 else "")
        log.info(f"  → {preview}")

    return transcripts


def save_transcript(
    records: list[dict],
    output_path: Path,
) -> None:
    """Write transcript records to a CSV file (LJSpeech-compatible format)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["chunk_id", "filename", "text", "duration", "path"]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    log.info(f"Saved transcript → {output_path} ({len(records)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chunk a long audio file and transcribe each chunk with Whisper."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input audio file (.wav, .mp3, .flac, ...)",
    )
    parser.add_argument(
        "--output",
        default="data/chunks",
        help="Output directory for WAV chunks (default: data/chunks)",
    )
    parser.add_argument(
        "--model",
        default="large-v3",
        help="Whisper model size: tiny | base | small | medium | large-v3 (default: large-v3)",
    )
    parser.add_argument(
        "--top-db",
        type=int,
        default=30,
        help="Silence threshold in dB (default: 30). Raise if too many chunks, lower if too few.",
    )
    parser.add_argument(
        "--min-dur",
        type=float,
        default=5.0,
        help="Minimum chunk duration in seconds (default: 5)",
    )
    parser.add_argument(
        "--max-dur",
        type=float,
        default=15.0,
        help="Maximum chunk duration in seconds (default: 15)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    transcript_path = Path("data/transcript.csv")

    if not input_path.exists():
        log.error(f"Input file not found: {input_path}")
        return

    audio, sr = load_audio(input_path)

    intervals = detect_speech_intervals(audio, sr, top_db=args.top_db)
    if not intervals:
        log.error("No speech detected. Exiting.")
        return

    chunks = build_chunks(intervals, min_dur=args.min_dur, max_dur=args.max_dur)
    if not chunks:
        log.error("No valid chunks built. Try adjusting --min-dur or --top-db.")
        return

    chunk_paths = export_chunks(audio, sr, chunks, output_dir, stem=input_path.stem)
    transcripts = transcribe_chunks(chunk_paths, model_size=args.model)
    records = [
        {
            "chunk_id": f"{input_path.stem}_chunk_{i:04d}",
            "filename": path.name,
            "text": text,
            "duration": round(end - start, 2),
            "path": str(path),
        }
        for i, (path, text, (start, end)) in enumerate(zip(chunk_paths, transcripts, chunks), 1)
    ]
    save_transcript(records, transcript_path)

    log.info("Done.")


if __name__ == "__main__":
    main()
