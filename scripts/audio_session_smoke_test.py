#!/usr/bin/env python
"""
scripts/audio_session_smoke_test.py
======================================
Minimal AudioSessionManager smoke test: synthetic audio in, one
complete command-audio segment out. Deliberately isolated -- does NOT
touch a real microphone, ASR, or the command interpreter (that chain
is a later task). Feeds hand-built synthetic chunks through a fake
wake-word detector (or, optionally, the real KeywordDetector, to prove
it plugs in safely -- it reports NOT_IMPLEMENTED without a trained
NOVA checkpoint, so no real detection happens either way here).

Usage
-----
    python scripts/audio_session_smoke_test.py
    python scripts/audio_session_smoke_test.py --use-real-detector
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from audio_processing.session import AudioSessionManager, AudioSessionState  # noqa: E402

SAMPLE_RATE = 16000


class _FakeKeywordResult:
    def __init__(self, status: str, confidence=None):
        self.status = status
        self.confidence = confidence


class _FireOnceFakeDetector:
    """Fires DETECTED on its 10th call, NO_DETECTION otherwise -- a
    stand-in for a real wake-word detector, used here purely to
    exercise AudioSessionManager's state machine with synthetic data."""

    def __init__(self, detect_on_call: int = 10):
        self.detect_on_call = detect_on_call
        self._calls = 0

    def detect_chunk(self, chunk: np.ndarray):
        idx = self._calls
        self._calls += 1
        if idx == self.detect_on_call:
            return _FakeKeywordResult(status="DETECTED", confidence=0.93)
        return _FakeKeywordResult(status="NO_DETECTION")


def _chunks(seconds: float, level: float, n_chunk_samples: int = 320):
    """Yield fixed-size synthetic chunks covering `seconds` at `level`
    amplitude (0.0 = silence)."""
    total = int(seconds * SAMPLE_RATE)
    produced = 0
    rng = np.random.RandomState(0)
    while produced < total:
        n = min(n_chunk_samples, total - produced)
        if level == 0.0:
            chunk = np.zeros(n, dtype=np.float32)
        else:
            chunk = (rng.randn(n).astype(np.float32) * 0.05 + level)
        yield chunk
        produced += n


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--use-real-detector", action="store_true",
        help="Plug in the real keyword_spotting.detector.KeywordDetector instead of "
             "a fake one, to demonstrate it integrates safely. No trained NOVA "
             "checkpoint exists in this repo, so it will report NOT_IMPLEMENTED and "
             "never actually fire -- this flag proves safe wiring, not detection.",
    )
    args = parser.parse_args()

    if args.use_real_detector:
        from keyword_spotting.detector import KeywordDetector
        detector = KeywordDetector()
        print(f"Real KeywordDetector.is_ready = {detector.is_ready} "
              f"(expected False -- no trained NOVA checkpoint in this repo)")
        print("NOTE: with no trained model, this detector will never fire DETECTED, "
              "so the session below will stay in LISTENING. Re-run without "
              "--use-real-detector to see a full LISTENING -> COMPLETE cycle.")
    else:
        detector = _FireOnceFakeDetector(detect_on_call=10)

    session = AudioSessionManager(
        sample_rate=SAMPLE_RATE,
        detector=detector,
        preroll_seconds=0.5,
        silence_seconds=0.6,
        min_command_seconds=0.3,
        max_command_seconds=5.0,
    )
    session.start()
    print(f"Initial state: {session.state.value}")

    # Simulate: 1s of ambient "silence" (LISTENING), then the detector
    # fires, then 1s of "speech", then enough "silence" to end the command.
    n_chunks_seen = 0
    for chunk in _chunks(seconds=1.0, level=0.0):
        session.process_chunk(chunk)
        n_chunks_seen += 1
        if session.state != AudioSessionState.LISTENING:
            break

    print(f"State after {n_chunks_seen} listening chunks: {session.state.value}")

    if session.state == AudioSessionState.COMMAND_CAPTURE:
        for chunk in _chunks(seconds=1.0, level=0.4):  # "speech"
            session.process_chunk(chunk)
        print(f"State after simulated speech: {session.state.value}")

        for chunk in _chunks(seconds=1.0, level=0.0):  # trailing "silence"
            session.process_chunk(chunk)
            if session.state == AudioSessionState.COMPLETE:
                break
        print(f"State after trailing silence: {session.state.value}")

    print()
    if session.result is not None:
        seg = session.result
        print("=== Command audio segment (ready for asr.transcriber) ===")
        print(f"Sample rate     : {seg.sample_rate} Hz")
        print(f"Sample count    : {seg.waveform.size}")
        print(f"Duration        : {seg.duration_seconds:.2f} s")
        print(f"End reason      : {seg.end_reason}")
        print(f"Wake word ts    : {seg.wake_word_detected_ts}")
        print(f"Command end ts  : {seg.command_end_ts}")
        print()
        print("This segment was NOT passed to ASR or the command interpreter --")
        print("that integration is a separate, later task.")
        return 0
    else:
        print("No command segment produced (session stayed in LISTENING).")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
