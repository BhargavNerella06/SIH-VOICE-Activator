"""
tests/unit/test_oww_backend.py
==================================
Tests for keyword_spotting.oww_backend.OpenWakeWordBackend.

The real openwakeword.model.Model is stubbed out here (via
object.__new__ + manually assigned attributes, mirroring how
test_kws_detector.py sets det._model to a sentinel) so these tests
exercise the frame-splitting / warm-up-workaround / resample logic in
isolation, without requiring a real trained "nova" ONNX file or
network access to download the shared embedding backbone.

The full happy path (loading a real model, downloading the shared
backbone) is exercised once an actual trained model exists -- see
keyword_spotting/training/calibrate_oww_threshold.py and the live-mic
smoke test, not here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from keyword_spotting.oww_backend import FRAME_SAMPLES, OpenWakeWordBackend


class _FakeOwwModel:
    """Stands in for openwakeword.model.Model in these unit tests."""

    def __init__(self, scores=None):
        self.reset_calls = 0
        self.predict_calls: list[np.ndarray] = []
        self._scores = iter(scores or [])

    def reset(self) -> None:
        self.reset_calls += 1

    def predict(self, frame: np.ndarray) -> dict:
        self.predict_calls.append(frame)
        try:
            score = next(self._scores)
        except StopIteration:
            score = 0.0
        return {"nova": score}


def _make_backend(fake_model: _FakeOwwModel) -> OpenWakeWordBackend:
    backend = object.__new__(OpenWakeWordBackend)
    backend._model = fake_model
    backend._label = "nova"
    return backend


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_missing_model_path_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        OpenWakeWordBackend(Path("__no_such_nova_model__.onnx"))


# ---------------------------------------------------------------------------
# Frame-splitting / warm-up-workaround behaviour
# ---------------------------------------------------------------------------

def test_score_resets_once_and_feeds_one_predict_call_per_frame():
    fake = _FakeOwwModel(scores=[0.0, 0.0, 0.0])
    backend = _make_backend(fake)

    window = np.zeros(FRAME_SAMPLES * 3, dtype=np.float32)
    backend.score(window, sample_rate=16000)

    assert fake.reset_calls == 1, "must reset once per independent window to avoid cross-window bleed"
    assert len(fake.predict_calls) == 3
    assert all(len(f) == FRAME_SAMPLES for f in fake.predict_calls)


def test_score_drops_incomplete_trailing_samples():
    fake = _FakeOwwModel(scores=[0.0, 0.0])
    backend = _make_backend(fake)

    window = np.zeros(FRAME_SAMPLES * 2 + 500, dtype=np.float32)  # 500 leftover samples
    backend.score(window, sample_rate=16000)

    assert len(fake.predict_calls) == 2, "trailing partial frame must be dropped, not padded or fed as-is"


def test_score_returns_max_across_all_frames_in_the_window():
    # Simulates openwakeword's own warm-up guard: early calls return 0.0
    # regardless of audio content, a real score appears once warmed up.
    fake = _FakeOwwModel(scores=[0.0, 0.0, 0.0, 0.0, 0.0, 0.93, 0.4])
    backend = _make_backend(fake)

    window = np.zeros(FRAME_SAMPLES * 7, dtype=np.float32)
    score = backend.score(window, sample_rate=16000)

    assert score == pytest.approx(0.93)


def test_score_returns_zero_without_touching_model_when_window_shorter_than_one_frame():
    fake = _FakeOwwModel(scores=[0.9])  # would be wrong if this were ever consulted
    backend = _make_backend(fake)

    window = np.zeros(FRAME_SAMPLES - 1, dtype=np.float32)
    score = backend.score(window, sample_rate=16000)

    assert score == 0.0
    assert fake.reset_calls == 0
    assert fake.predict_calls == []


def test_score_converts_float_audio_to_int16_pcm():
    fake = _FakeOwwModel(scores=[0.0])
    backend = _make_backend(fake)

    window = np.full(FRAME_SAMPLES, 1.0, dtype=np.float32)  # full-scale positive
    backend.score(window, sample_rate=16000)

    frame = fake.predict_calls[0]
    assert frame.dtype == np.int16
    assert frame[0] == 32767


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def test_score_resamples_when_input_rate_differs(monkeypatch):
    fake = _FakeOwwModel(scores=[0.0])
    backend = _make_backend(fake)

    resampled = np.zeros(FRAME_SAMPLES, dtype=np.float32)
    calls = {}

    def fake_resample(y, orig_sr, target_sr):
        calls["args"] = (orig_sr, target_sr)
        return resampled

    monkeypatch.setattr("keyword_spotting.oww_backend.resample_if_needed", fake_resample)

    window = np.zeros(8000, dtype=np.float32)  # 8 kHz input
    backend.score(window, sample_rate=8000)

    assert calls["args"] == (8000, 16000)
    assert len(fake.predict_calls) == 1


# ---------------------------------------------------------------------------
# label property
# ---------------------------------------------------------------------------

def test_label_property_exposes_loaded_model_label():
    backend = _make_backend(_FakeOwwModel())
    assert backend.label == "nova"
