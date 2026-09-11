"""
tests/unit/test_kws_detector.py
==================================
Tests for keyword_spotting.detector.KeywordDetector.

Threshold/debounce logic is tested by monkeypatching the model-forward
step (_infer_window) with controlled probabilities, so the logic is
verified independently of any actually-trained checkpoint. Tests that
need a "ready" detector without a real model set `det._model` to a
sentinel object -- is_ready only checks `is not None`.
"""

from __future__ import annotations

import numpy as np
import pytest


def _make_detector(
    threshold: float = 0.8,
    consecutive_frames: int = 3,
    window_seconds: float = 0.1,
    hop_seconds: float = 0.1,
    sample_rate: int = 1000,
    model_path: str = "__no_such_checkpoint__.pt",
):
    from keyword_spotting.detector import KeywordDetector
    return KeywordDetector(
        model_path=model_path,
        sample_rate=sample_rate,
        window_seconds=window_seconds,
        hop_seconds=hop_seconds,
        threshold=threshold,
        consecutive_frames=consecutive_frames,
    )


# ---------------------------------------------------------------------------
# Configuration / validation
# ---------------------------------------------------------------------------

def test_rejects_unsupported_keyword():
    from keyword_spotting.detector import KeywordDetector
    with pytest.raises(ValueError):
        KeywordDetector(keywords=["OTHER"])


def test_default_keyword_is_nova():
    det = _make_detector()
    assert det.keywords == ["NOVA"]


# ---------------------------------------------------------------------------
# "Not ready" honesty: no fabricated confidence when no model is loaded
# ---------------------------------------------------------------------------

def test_not_ready_without_checkpoint():
    det = _make_detector()
    assert det.is_ready is False
    assert det.load_error is not None


def test_detect_reports_not_implemented_without_model():
    det = _make_detector()
    result = det.detect(np.zeros(1000, dtype=np.float32), sample_rate=1000)
    assert result.status == "NOT_IMPLEMENTED"
    assert result.confidence is None
    assert result.keyword is None


def test_detect_chunk_reports_not_implemented_without_model():
    det = _make_detector()
    result = det.detect_chunk(np.zeros(100, dtype=np.float32))
    assert result is not None
    assert result.status == "NOT_IMPLEMENTED"
    assert result.confidence is None


# ---------------------------------------------------------------------------
# Streaming buffering behaviour
# ---------------------------------------------------------------------------

def test_detect_chunk_returns_none_until_window_and_hop_satisfied(monkeypatch):
    det = _make_detector(window_seconds=1.0, hop_seconds=1.0, sample_rate=1000)
    det._model = object()  # pretend ready; _infer_window is never reached here
    small_chunk = np.zeros(100, dtype=np.float32)  # window needs 1000 samples
    for _ in range(9):
        assert det.detect_chunk(small_chunk) is None
    # the 10th chunk fills the 1000-sample window and reaches the hop
    called = {"n": 0}

    def fake_infer(window):
        called["n"] += 1
        return 0.0, 1.0

    monkeypatch.setattr(det, "_infer_window", fake_infer)
    result = det.detect_chunk(small_chunk)
    assert result is not None
    assert called["n"] == 1


# ---------------------------------------------------------------------------
# Threshold + consecutive-frame debounce (anti-false-trigger)
# ---------------------------------------------------------------------------

def test_single_high_probability_frame_does_not_trigger(monkeypatch):
    det = _make_detector(threshold=0.8, consecutive_frames=3, window_seconds=0.1, hop_seconds=0.1, sample_rate=1000)
    det._model = object()
    monkeypatch.setattr(det, "_infer_window", lambda window: (0.99, 1.0))

    chunk = np.zeros(100, dtype=np.float32)  # == window == hop at this config
    result = det.detect_chunk(chunk)
    assert result is not None
    assert result.status == "NO_DETECTION", "a single positive frame must not trigger DETECTED"


def test_consecutive_positive_frames_trigger_detection(monkeypatch):
    det = _make_detector(threshold=0.8, consecutive_frames=3, window_seconds=0.1, hop_seconds=0.1, sample_rate=1000)
    det._model = object()

    probs = iter([0.9, 0.9, 0.2, 0.9, 0.9, 0.9])
    monkeypatch.setattr(det, "_infer_window", lambda window: (next(probs), 1.0))

    chunk = np.zeros(100, dtype=np.float32)
    statuses = [det.detect_chunk(chunk).status for _ in range(6)]

    assert statuses == [
        "NO_DETECTION", "NO_DETECTION", "NO_DETECTION",
        "NO_DETECTION", "NO_DETECTION", "DETECTED",
    ]


def test_detected_result_carries_real_confidence_and_latency(monkeypatch):
    det = _make_detector(threshold=0.5, consecutive_frames=1, window_seconds=0.1, hop_seconds=0.1, sample_rate=1000)
    det._model = object()
    monkeypatch.setattr(det, "_infer_window", lambda window: (0.732, 4.2))

    chunk = np.zeros(100, dtype=np.float32)
    result = det.detect_chunk(chunk)

    assert result.status == "DETECTED"
    assert result.keyword == "NOVA"
    assert result.confidence == pytest.approx(0.732)
    assert result.latency_ms == pytest.approx(4.2)


def test_reset_stream_clears_debounce_state(monkeypatch):
    det = _make_detector(threshold=0.5, consecutive_frames=2, window_seconds=0.1, hop_seconds=0.1, sample_rate=1000)
    det._model = object()
    monkeypatch.setattr(det, "_infer_window", lambda window: (0.9, 1.0))

    chunk = np.zeros(100, dtype=np.float32)
    r1 = det.detect_chunk(chunk)
    assert r1.status == "NO_DETECTION"  # 1st of 2 consecutive frames

    det.reset_stream()
    r2 = det.detect_chunk(chunk)
    assert r2.status == "NO_DETECTION", "reset_stream must clear the consecutive-hit counter"


# ---------------------------------------------------------------------------
# Result JSON-shape contract
# ---------------------------------------------------------------------------

def test_to_dict_detected_shape():
    from keyword_spotting.detector import KeywordResult
    result = KeywordResult(status="DETECTED", keyword="NOVA", confidence=0.93, latency_ms=12.3, message="x")
    assert result.to_dict() == {
        "status": "DETECTED", "keyword": "NOVA", "confidence": 0.93, "latency_ms": 12.3,
    }


def test_to_dict_no_detection_shape():
    from keyword_spotting.detector import KeywordResult
    result = KeywordResult(status="NO_DETECTION", latency_ms=5.0, message="x")
    assert result.to_dict() == {"status": "NO_DETECTION"}


# ---------------------------------------------------------------------------
# One-shot detect() resamples to the configured rate
# ---------------------------------------------------------------------------

def test_detect_resamples_when_input_rate_differs(monkeypatch):
    det = _make_detector(threshold=0.9, consecutive_frames=1, window_seconds=0.1, hop_seconds=0.1, sample_rate=1000)
    det._model = object()
    monkeypatch.setattr(det, "_infer_window", lambda window: (0.0, 1.0))

    # Waveform sampled at 2000 Hz, detector configured for 1000 Hz.
    waveform = np.random.randn(2000).astype(np.float32)
    result = det.detect(waveform, sample_rate=2000)
    assert result.status == "NO_DETECTION"  # must not crash on the resample path
