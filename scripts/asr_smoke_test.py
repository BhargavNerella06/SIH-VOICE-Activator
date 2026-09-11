#!/usr/bin/env python
"""
scripts/asr_smoke_test.py
============================
Minimal ASR-only smoke test: audio in, transcribed text out.

Deliberately exercises ONLY asr.transcriber -- no command interpretation,
no streaming/HTTP, no KWS, no FastAPI. This matches the "ASR backend
only" scope of the task this script was written for: proving
audio -> ASR -> text works in isolation before it is wired to anything
else.

Usage
-----
    python scripts/asr_smoke_test.py --wav path/to/speech.wav
    python scripts/asr_smoke_test.py --mic --duration 3

No microphone audio is recorded by running `--wav` mode; `--mic` mode
requires an actual person to speak (same caveat as
keyword_spotting/training/record_data.py's live-mic mode).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from asr.transcriber import get_transcriber  # noqa: E402
from audio_processing.utils import ensure_mono  # noqa: E402


def _load_wav(path: str) -> tuple[np.ndarray, int]:
    y, sr = sf.read(path)
    return ensure_mono(np.asarray(y, dtype=np.float32)), sr


def _record_from_mic(duration_s: float) -> tuple[np.ndarray, int]:
    from audio_processing.capture import MicCapture, MicCaptureError

    sample_rate = 16000
    mic = MicCapture(sample_rate=sample_rate)
    try:
        mic.start()
    except MicCaptureError as exc:
        print(f"ERROR: could not start microphone: {exc}")
        raise SystemExit(1)

    print(f"Recording {duration_s:.1f}s from the microphone...")
    chunks = []
    start = time.perf_counter()
    try:
        while (time.perf_counter() - start) < duration_s:
            chunks.append(mic.read_chunk(timeout=2.0))
    finally:
        mic.stop()
    waveform = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    return waveform, sample_rate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wav", type=str, default=None, help="WAV file to transcribe.")
    parser.add_argument("--mic", action="store_true", help="Record live from the microphone instead.")
    parser.add_argument("--duration", type=float, default=3.0, help="Seconds to record if --mic is given.")
    args = parser.parse_args()

    if args.mic:
        waveform, sample_rate = _record_from_mic(args.duration)
    elif args.wav:
        waveform, sample_rate = _load_wav(args.wav)
    else:
        parser.error("Provide either --wav <file> or --mic")
        return 2

    if waveform.size == 0:
        print("No audio to transcribe (empty waveform).")
        return 1

    print(f"Loaded {len(waveform) / sample_rate:.2f}s of audio at {sample_rate} Hz")

    transcriber = get_transcriber()
    if not transcriber.is_ready:
        print(f"ASR model not available: {transcriber.load_error}")
        return 1

    t0 = time.perf_counter()
    result = transcriber.transcribe(waveform, sample_rate=sample_rate)
    wall_ms = (time.perf_counter() - t0) * 1000.0

    print()
    print(f"Status     : {result.status}")
    print(f"Transcript : {result.text!r}")
    print(f"Language   : {result.language}")
    print(
        f"Latency    : {result.latency_ms:.1f} ms (inside transcribe()) / "
        f"{wall_ms:.1f} ms (wall clock, incl. this script's overhead)"
    )
    if result.status != "OK":
        print(f"Message    : {result.message}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
