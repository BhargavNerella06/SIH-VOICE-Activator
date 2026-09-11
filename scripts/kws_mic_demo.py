#!/usr/bin/env python
"""
scripts/kws_mic_demo.py
=========================
Stage 2.2 local demonstration: microphone -> KWS -> "WAKE DETECTED".

Independent of FastAPI -- run directly on this machine (and, later,
directly on edge hardware).

Usage
-----
    python scripts/kws_mic_demo.py
    python scripts/kws_mic_demo.py --threshold 0.8 --consecutive-frames 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audio_processing.capture import MicCapture, MicCaptureError  # noqa: E402
from keyword_spotting.detector import (  # noqa: E402
    DEFAULT_CONSECUTIVE_FRAMES,
    DEFAULT_THRESHOLD,
    KeywordDetector,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--consecutive-frames", type=int, default=DEFAULT_CONSECUTIVE_FRAMES)
    parser.add_argument("--model-path", type=str, default=None)
    args = parser.parse_args()

    detector = KeywordDetector(
        threshold=args.threshold,
        consecutive_frames=args.consecutive_frames,
        model_path=args.model_path,
    )
    if not detector.is_ready:
        print("ERROR: KWS model is not trained/loaded.")
        print(detector.load_error)
        return 1

    mic = MicCapture(sample_rate=detector.sample_rate)
    try:
        mic.start()
    except MicCaptureError as exc:
        print(f"ERROR: could not start microphone: {exc}")
        return 1

    print(f"Sample rate        : {detector.sample_rate} Hz")
    print(f"Analysis window     : {detector.window_seconds * 1000:.0f} ms")
    print(f"Hop                : {detector.hop_seconds * 1000:.0f} ms")
    print(f"Threshold          : {detector.threshold}")
    print(f"Consecutive frames : {detector.consecutive_frames}")
    print("Listening for 'NOVA'... (Ctrl+C to stop)")
    print()

    try:
        while True:
            chunk = mic.read_chunk(timeout=5.0)
            result = detector.detect_chunk(chunk)
            if result is None:
                continue
            if result.status == "DETECTED":
                print(
                    f"*** WAKE DETECTED *** keyword={result.keyword} "
                    f"confidence={result.confidence:.3f} latency_ms={result.latency_ms:.1f}"
                )
            # NO_DETECTION frames are intentionally not printed (would be too noisy).
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except MicCaptureError as exc:
        print(f"ERROR while reading microphone: {exc}")
        return 1
    finally:
        mic.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
