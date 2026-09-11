#!/usr/bin/env python
"""
scripts/debug_wake_word_trace.py
====================================
TEMPORARY diagnostic tool for tracing the NOVA wake-word pipeline stage
by stage, live:

    mic chunk -> sliding window buffer -> feature extraction (log-Mel)
      -> TinyKWSNet score -> threshold/debounce -> DETECTED

This does NOT modify keyword_spotting/detector.py, keyword_spotting/
tinykws_backend.py, or keyword_spotting/features.py -- it only wraps
their existing functions at runtime (monkeypatching within this script's
own process) purely to print what's happening at each stage. The real
pipeline logic, threshold, and model are completely untouched.

Run this yourself (not via an automated tool call) so you can see the
live trace and time saying "NOVA" against it -- same reason every other
live-mic script in this project needs to be run interactively.

Usage
-----
    python scripts/debug_wake_word_trace.py
    python scripts/debug_wake_word_trace.py --threshold 0.85   # unchanged default, just explicit
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from audio_processing.capture import MicCapture, MicCaptureError  # noqa: E402
import keyword_spotting.tinykws_backend as tinykws_backend  # noqa: E402
from keyword_spotting.detector import (  # noqa: E402
    DEFAULT_CONSECUTIVE_FRAMES,
    DEFAULT_THRESHOLD,
    KeywordDetector,
)

_score_call_count = 0
_feature_call_count = 0
_real_extract_features = tinykws_backend.extract_features
_real_score = tinykws_backend.TinyKWSNetBackend.score


def _traced_extract_features(waveform, **kwargs):
    """Wraps keyword_spotting.features.extract_features (as imported into
    tinykws_backend's namespace) to print feature stats -- STAGE 3."""
    global _feature_call_count
    _feature_call_count += 1
    feat = _real_extract_features(waveform, **kwargs)
    if _feature_call_count % 1 == 0:  # every window -- this stage is cheap to print
        print(
            f"  [3 FEATURES]   shape={feat.shape}  min={feat.min():.3f}  "
            f"max={feat.max():.3f}  mean={feat.mean():.3f}  "
            f"all_zero={bool(np.all(feat == 0))}"
        )
    return feat


def _traced_score(self, window, sample_rate):
    """Wraps TinyKWSNetBackend.score() to print the raw model probability
    every time it's called -- STAGE 4/5. Confirms the model is actually
    invoked for every analysis window, not just logging a cached value."""
    global _score_call_count
    _score_call_count += 1
    peak = float(np.max(np.abs(window))) if window.size else 0.0
    t0 = time.perf_counter()
    prob = _real_score(self, window, sample_rate)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    print(
        f"  [4 MODEL CALL] #{_score_call_count}  window_peak={peak:.4f}  "
        f"-> NOVA_prob={prob:.4f}  (inference {latency_ms:.2f}ms)"
    )
    return prob


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--consecutive-frames", type=int, default=DEFAULT_CONSECUTIVE_FRAMES)
    parser.add_argument("--model-path", type=str, default=None)
    args = parser.parse_args()

    print("=" * 78)
    print("NOVA wake-word pipeline trace (TEMPORARY diagnostic -- no files modified)")
    print("=" * 78)

    # Install the traces (runtime monkeypatch only, this process only).
    tinykws_backend.extract_features = _traced_extract_features
    tinykws_backend.TinyKWSNetBackend.score = _traced_score

    print("\n--- STAGE 0: model load / detector registration ---")
    detector = KeywordDetector(
        threshold=args.threshold,
        consecutive_frames=args.consecutive_frames,
        model_path=args.model_path,
    )
    print(f"  detector.is_ready       = {detector.is_ready}")
    print(f"  detector.load_error     = {detector.load_error}")
    if not detector.is_ready:
        print("STAGE 0 FAILED: model not loaded. Stopping here.")
        return 1
    print(f"  backend n_mels          = {detector.n_mels}")
    print(f"  backend sample_rate     = {detector.sample_rate} Hz")
    print(f"  backend window_seconds  = {detector.window_seconds}")
    print(f"  hop_seconds             = {detector.hop_seconds}")
    print(f"  threshold               = {detector.threshold}")
    print(f"  consecutive_frames      = {detector.consecutive_frames}")

    print("\n--- STAGE 1: microphone open ---")
    mic = MicCapture(sample_rate=detector.sample_rate)
    try:
        mic.start()
    except MicCaptureError as exc:
        print(f"STAGE 1 FAILED: could not open microphone: {exc}")
        return 1
    print(f"  mic.sample_rate         = {mic.sample_rate} Hz")
    print(f"  mic.chunk_size          = {mic.chunk_size} samples "
          f"({mic.chunk_size / mic.sample_rate * 1000:.1f} ms)")
    print("  NOTE: MicCapture yields float32 samples in [-1, 1] directly from "
          "sounddevice -- there is no PCM16 conversion in this live-mic path "
          "(PCM16 is only used by the HTTP/WebSocket streaming endpoints in "
          "api/routers/stream.py and api/routers/asr_stream.py, a different "
          "code path). 16kHz mono is enforced by MicCapture's own sample_rate "
          "argument, matching the checkpoint's declared sample_rate above.")

    print("\nListening... say 'NOVA' clearly when ready. Ctrl+C to stop and see summary.\n")

    n_chunks = 0
    max_prob_seen = 0.0
    detections = 0
    t_start = time.time()
    try:
        while True:
            chunk = mic.read_chunk(timeout=5.0)
            n_chunks += 1
            if n_chunks % 20 == 0:
                peak = float(np.max(np.abs(chunk)))
                print(f"[1 AUDIO]      chunk #{n_chunks}  samples={len(chunk)}  "
                      f"dtype={chunk.dtype}  peak={peak:.4f}")

            result = detector.detect_chunk(chunk)
            if result is None:
                continue  # buffer still filling, or hop not reached yet

            if result.status == "DETECTED":
                detections += 1
                print(
                    f"  [7 DETECTED!]  *** WAKE WORD FIRED *** "
                    f"confidence={result.confidence:.4f} latency_ms={result.latency_ms:.1f}\n"
                )
    except KeyboardInterrupt:
        pass
    except MicCaptureError as exc:
        print(f"\nERROR while reading microphone: {exc}")
    finally:
        mic.stop()
        tinykws_backend.extract_features = _real_extract_features
        tinykws_backend.TinyKWSNetBackend.score = _real_score

    elapsed = time.time() - t_start
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"Elapsed              : {elapsed:.1f}s")
    print(f"Mic chunks received  : {n_chunks}")
    print(f"Feature extractions  : {_feature_call_count}")
    print(f"Model score() calls  : {_score_call_count}")
    print(f"Detections (>= thr)  : {detections}")
    print(f"Threshold            : {detector.threshold}")
    if n_chunks == 0:
        print("DIAGNOSIS: STAGE 1 (audio) never delivered a single chunk -- microphone problem.")
    elif _score_call_count == 0:
        print("DIAGNOSIS: audio arrived but the model was never invoked -- buffering/hop problem "
              "in KeywordDetector.detect_chunk (window never filled, or hop never reached).")
    elif detections == 0:
        print("DIAGNOSIS: pipeline ran end-to-end (audio -> features -> model -> threshold check) "
              "but no window's NOVA probability reached the threshold. This matches this "
              "checkpoint's known real-speech score range (~0.50-0.76) measured during training "
              "evaluation, against a 0.85 threshold. This is a model-confidence/threshold gap, "
              "NOT a broken pipeline stage.")
    else:
        print("DIAGNOSIS: pipeline works end-to-end and fired DETECTED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
