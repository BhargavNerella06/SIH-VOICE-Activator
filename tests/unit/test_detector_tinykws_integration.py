"""
tests/unit/test_detector_tinykws_integration.py
===================================================
Integration between keyword_spotting.detector.KeywordDetector and the
TinyKWSNet backend (keyword_spotting.tinykws_backend), using a tiny,
locally-constructed, deterministic checkpoint throughout.

IMPORTANT: no real trained NOVA model is used or claimed anywhere in
this file. These tests verify the *wiring* (checkpoint -> backend ->
detector -> streaming/result contract), never detection accuracy on
real speech.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from keyword_spotting.model import TinyKWSNet

SR = 1000
WINDOW_SECONDS = 0.5
HOP_SECONDS = 0.25
N_MELS = 8


def _make_checkpoint(path: Path, **overrides) -> None:
    torch.manual_seed(0)
    model = TinyKWSNet(n_mels=N_MELS, num_classes=2)
    package = {
        "state_dict": model.state_dict(),
        "n_mels": N_MELS,
        "sample_rate": SR,
        "window_seconds": WINDOW_SECONDS,
        "keyword": "NOVA",
    }
    package.update(overrides)
    torch.save(package, str(path))


def _make_detector(model_path: Path, **kwargs):
    from keyword_spotting.detector import KeywordDetector

    # Constructor sample_rate/window_seconds are deliberately "wrong"
    # (mismatched with the checkpoint) in most tests below, to prove
    # they get overridden rather than silently used.
    defaults = dict(
        model_path=str(model_path),
        sample_rate=99999,
        window_seconds=12.34,
        hop_seconds=HOP_SECONDS,
        threshold=0.5,
        consecutive_frames=1,
    )
    defaults.update(kwargs)
    return KeywordDetector(**defaults)


# ---------------------------------------------------------------------------
# Checkpoint loading via KeywordDetector
# ---------------------------------------------------------------------------

def test_detector_loads_tinykws_checkpoint_and_becomes_ready(tmp_path):
    ckpt = tmp_path / "kws_nova_cnn.pt"
    _make_checkpoint(ckpt)

    det = _make_detector(ckpt)

    assert det.is_ready is True
    assert det.load_error is None


def test_detector_selects_tinykws_backend_by_pt_extension(tmp_path):
    from keyword_spotting.tinykws_backend import TinyKWSNetBackend

    ckpt = tmp_path / "kws_nova_cnn.pt"
    _make_checkpoint(ckpt)
    det = _make_detector(ckpt)

    assert isinstance(det._model, TinyKWSNetBackend)


# ---------------------------------------------------------------------------
# Feature configuration propagation (do not silently hard-code)
# ---------------------------------------------------------------------------

def test_detector_adopts_checkpoint_feature_config_overriding_constructor_args(tmp_path):
    ckpt = tmp_path / "kws_nova_cnn.pt"
    _make_checkpoint(ckpt)

    det = _make_detector(ckpt)  # constructed with sample_rate=99999, window_seconds=12.34

    assert det.sample_rate == SR
    assert det.window_seconds == WINDOW_SECONDS
    assert det.n_mels == N_MELS


def test_detector_window_buffer_sized_from_checkpoint_not_constructor_args(tmp_path):
    ckpt = tmp_path / "kws_nova_cnn.pt"
    _make_checkpoint(ckpt)

    det = _make_detector(ckpt)

    expected_window_samples = int(round(SR * WINDOW_SECONDS))
    assert det._window_samples == expected_window_samples
    assert det._buffer.maxlen == expected_window_samples


def test_openwakeword_path_does_not_adopt_any_feature_config(tmp_path):
    """The openWakeWord backend carries no n_mels/window_seconds of its
    own -- sample_rate/window_seconds must stay exactly as constructed,
    unaffected by the TinyKWSNet-specific override path."""
    from keyword_spotting.detector import KeywordDetector

    det = KeywordDetector(
        model_path=str(tmp_path / "does_not_exist.onnx"),
        sample_rate=12345,
        window_seconds=7.0,
    )
    assert det.is_ready is False  # file doesn't exist, but config must be untouched regardless
    assert det.sample_rate == 12345
    assert det.window_seconds == 7.0
    assert det.n_mels is None


# ---------------------------------------------------------------------------
# Streaming chunk accumulation / inference-window creation
# ---------------------------------------------------------------------------

def test_streaming_chunks_accumulate_into_one_inference_window(tmp_path):
    ckpt = tmp_path / "kws_nova_cnn.pt"
    _make_checkpoint(ckpt)
    det = _make_detector(ckpt)

    window_samples = int(round(SR * WINDOW_SECONDS))
    hop_samples = int(round(SR * HOP_SECONDS))
    small_chunk = np.zeros(hop_samples // 5 or 1, dtype=np.float32)

    results = []
    # Feed chunks well past one full window/hop; every non-None result
    # must be a real KeywordResult (i.e. an inference actually ran).
    for _ in range(200):
        r = det.detect_chunk(small_chunk)
        if r is not None:
            results.append(r)
        if len(results) >= 1:
            break

    assert len(results) >= 1
    assert results[0].status in ("DETECTED", "NO_DETECTION")


def test_detect_chunk_returns_none_before_window_is_full(tmp_path):
    ckpt = tmp_path / "kws_nova_cnn.pt"
    _make_checkpoint(ckpt)
    det = _make_detector(ckpt)

    tiny_chunk = np.zeros(1, dtype=np.float32)
    assert det.detect_chunk(tiny_chunk) is None  # far short of a full window


def test_one_shot_detect_runs_end_to_end_on_a_full_waveform(tmp_path):
    ckpt = tmp_path / "kws_nova_cnn.pt"
    _make_checkpoint(ckpt)
    det = _make_detector(ckpt, threshold=1.1)  # unreachable threshold -> deterministic NO_DETECTION

    waveform = np.random.RandomState(0).randn(SR * 2).astype(np.float32) * 0.1
    result = det.detect(waveform, sample_rate=SR)

    assert result.status == "NO_DETECTION"


# ---------------------------------------------------------------------------
# Result format contract (status / confidence / keyword / latency)
# ---------------------------------------------------------------------------

def test_detected_result_has_full_result_shape(tmp_path):
    ckpt = tmp_path / "kws_nova_cnn.pt"
    _make_checkpoint(ckpt)
    # threshold=0 -> guaranteed to fire on the very first inferred window.
    det = _make_detector(ckpt, threshold=0.0, consecutive_frames=1)

    window_samples = int(round(SR * WINDOW_SECONDS))
    result = det.detect_chunk(np.zeros(window_samples, dtype=np.float32))

    assert result is not None
    assert result.status == "DETECTED"
    assert result.keyword == "NOVA"
    assert result.confidence is not None
    assert 0.0 <= result.confidence <= 1.0
    assert result.latency_ms is not None
    assert result.latency_ms >= 0.0
    assert result.to_dict() == {
        "status": "DETECTED",
        "keyword": "NOVA",
        "confidence": result.confidence,
        "latency_ms": result.latency_ms,
    }


# ---------------------------------------------------------------------------
# Missing checkpoint behaviour (no fabricated detection)
# ---------------------------------------------------------------------------

def test_missing_checkpoint_is_not_implemented_not_fabricated(tmp_path):
    det = _make_detector(tmp_path / "kws_nova_cnn.pt")  # never created

    assert det.is_ready is False
    assert det.load_error is not None
    assert "kws_nova_cnn.pt" in det.load_error

    result = det.detect(np.zeros(SR, dtype=np.float32), sample_rate=SR)
    assert result.status == "NOT_IMPLEMENTED"
    assert result.confidence is None
    assert result.keyword is None


def test_default_model_path_points_at_tinykws_checkpoint():
    """DEFAULT_MODEL_PATH is the primary-training-path artifact name --
    this is a naming/config assertion, not a claim that this file exists
    or that NOVA has been trained."""
    from keyword_spotting.detector import DEFAULT_MODEL_PATH

    assert DEFAULT_MODEL_PATH.name == "kws_nova_cnn.pt"
    assert DEFAULT_MODEL_PATH.suffix == ".pt"


# ---------------------------------------------------------------------------
# Malformed/incompatible checkpoint behaviour (surfaced as NOT_IMPLEMENTED,
# never a crash and never a fabricated DETECTED/NO_DETECTION)
# ---------------------------------------------------------------------------

def test_malformed_checkpoint_is_not_implemented_not_a_crash(tmp_path):
    ckpt = tmp_path / "kws_nova_cnn.pt"
    torch.save({"not_a_real_checkpoint": True}, str(ckpt))  # missing all required keys

    det = _make_detector(ckpt)  # must not raise

    assert det.is_ready is False
    assert det.load_error is not None
    assert "missing required key" in det.load_error

    result = det.detect_chunk(np.zeros(10, dtype=np.float32))
    assert result is not None
    assert result.status == "NOT_IMPLEMENTED"


def test_wrong_keyword_checkpoint_is_not_implemented(tmp_path):
    ckpt = tmp_path / "kws_nova_cnn.pt"
    _make_checkpoint(ckpt, keyword="HELLO")

    det = _make_detector(ckpt)

    assert det.is_ready is False
    assert "HELLO" in det.load_error


def test_unrecognized_extension_is_not_implemented(tmp_path):
    ckpt = tmp_path / "kws_nova_cnn.weights"  # neither .pt/.pth nor .onnx
    ckpt.write_bytes(b"whatever")

    det = _make_detector(ckpt)

    assert det.is_ready is False
    assert "Unrecognized" in det.load_error


# ---------------------------------------------------------------------------
# Backward compatibility: openWakeWord support kept intact
# ---------------------------------------------------------------------------

def test_openwakeword_backend_still_importable_and_used_for_onnx_paths():
    """keep openWakeWord support intact -- do not delete it."""
    from keyword_spotting.oww_backend import OpenWakeWordBackend  # must still import cleanly
    from keyword_spotting.detector import _ONNX_SUFFIXES

    assert ".onnx" in _ONNX_SUFFIXES
    assert OpenWakeWordBackend is not None


# ---------------------------------------------------------------------------
# Honesty: a trained checkpoint exists, but that alone is not an accuracy claim
# ---------------------------------------------------------------------------

def test_nova_checkpoint_exists_and_loads_successfully():
    """A real models/kws_nova_cnn.pt now exists (trained on the project's
    small synthetic+real dataset -- see keyword_spotting/training/train.py)
    and KeywordDetector loads it successfully. This documents that fact; it
    is NOT an accuracy claim -- see that checkpoint's own saved test
    metrics and docs/ for the small-dataset caveats."""
    from keyword_spotting.detector import DEFAULT_MODEL_PATH, KeywordDetector

    assert DEFAULT_MODEL_PATH.exists(), (
        "Expected a trained models/kws_nova_cnn.pt to exist at this stage."
    )
    detector = KeywordDetector()
    assert detector.is_ready is True
    assert detector.load_error is None
