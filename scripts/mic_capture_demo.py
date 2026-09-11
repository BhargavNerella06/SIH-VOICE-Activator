#!/usr/bin/env python
"""
scripts/mic_capture_demo.py
============================
Stage 2 / Milestone 2.1 local demonstration script.

Captures live microphone audio in streaming/chunked form using
audio_processing.capture.MicCapture and reports:
  - sample rate
  - chunk duration
  - number of chunks received
  - elapsed time

This script is intentionally independent of the FastAPI app: it is meant
to be runnable standalone, including later directly on edge hardware.

Usage
-----
    python scripts/mic_capture_demo.py
    python scripts/mic_capture_demo.py --duration 5 --sample-rate 16000 --chunk-size 512
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Allow running as `python scripts/mic_capture_demo.py` without installing
# the package first.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audio_processing.capture import MicCapture, MicCaptureError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=5.0,
                         help="How long to capture, in seconds (default: 5.0)")
    parser.add_argument("--sample-rate", type=int, default=16000,
                         help="Capture sample rate in Hz (default: 16000)")
    parser.add_argument("--chunk-size", type=int, default=512,
                         help="Samples per chunk (default: 512, ~32ms @ 16kHz)")
    args = parser.parse_args()

    chunk_duration_s = args.chunk_size / args.sample_rate
    print(f"Sample rate   : {args.sample_rate} Hz")
    print(f"Chunk size    : {args.chunk_size} samples")
    print(f"Chunk duration: {chunk_duration_s * 1000:.1f} ms")
    print(f"Capturing for : {args.duration:.1f} s ... (Ctrl+C to stop early)")
    print()

    mic = MicCapture(sample_rate=args.sample_rate, chunk_size=args.chunk_size)

    num_chunks = 0
    peak_amplitude = 0.0
    start = time.perf_counter()

    try:
        mic.start()
    except MicCaptureError as exc:
        print(f"ERROR: could not start microphone capture: {exc}")
        return 1

    try:
        while (time.perf_counter() - start) < args.duration:
            try:
                chunk = mic.read_chunk(timeout=2.0)
            except MicCaptureError as exc:
                print(f"ERROR while reading chunk: {exc}")
                break
            num_chunks += 1
            peak_amplitude = max(peak_amplitude, float(np.max(np.abs(chunk))))
    except KeyboardInterrupt:
        print("\nStopped early by user.")
    finally:
        mic.stop()

    elapsed = time.perf_counter() - start

    print()
    print("=== Capture summary ===")
    print(f"Sample rate       : {args.sample_rate} Hz")
    print(f"Chunk duration    : {chunk_duration_s * 1000:.1f} ms")
    print(f"Chunks received   : {num_chunks}")
    print(f"Elapsed time      : {elapsed:.2f} s")
    if num_chunks > 0:
        print(f"Peak amplitude    : {peak_amplitude:.4f} (of 1.0 full scale)")
        expected_chunks = elapsed / chunk_duration_s
        print(f"Expected chunks   : ~{expected_chunks:.0f} (based on elapsed time)")
    else:
        print("No audio chunks were received — check microphone/device availability.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
