"""
tests/unit/test_capture.py
===========================
Tests for audio_processing.capture.MicCapture that do not require
physical microphone hardware.

The real audio backend (sounddevice / PortAudio) is replaced with a
fake module injected into sys.modules, so these tests exercise the
actual chunking/queue/error-handling logic in MicCapture without ever
touching a real device.
"""

from __future__ import annotations

import sys
import types
from typing import Dict, Optional

import numpy as np
import pytest


def _install_fake_sounddevice(
    monkeypatch: pytest.MonkeyPatch,
    callback_holder: Dict[str, object],
    raise_on_construct: bool = False,
    raise_on_start: bool = False,
) -> "types.ModuleType":
    """Install a fake `sounddevice` module that records the InputStream callback."""
    fake_sd = types.ModuleType("sounddevice")

    class FakeInputStream:
        def __init__(self, samplerate, blocksize, channels, dtype, device, callback):
            if raise_on_construct:
                raise RuntimeError("no audio device found")
            self.samplerate = samplerate
            self.blocksize = blocksize
            self.channels = channels
            self.dtype = dtype
            self.device = device
            self.callback = callback
            callback_holder["callback"] = callback

        def start(self):
            if raise_on_start:
                raise RuntimeError("failed to start stream")

        def stop(self):
            pass

        def close(self):
            pass

    fake_sd.InputStream = FakeInputStream  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
    return fake_sd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def test_default_configuration_suits_kws():
    from audio_processing.capture import MicCapture, DEFAULT_SAMPLE_RATE, DEFAULT_CHUNK_SIZE
    mic = MicCapture()
    assert mic.sample_rate == DEFAULT_SAMPLE_RATE == 16000
    assert mic.chunk_size == DEFAULT_CHUNK_SIZE == 512
    assert mic.channels == 1
    assert mic.is_running is False


def test_custom_sample_rate_and_chunk_size():
    from audio_processing.capture import MicCapture
    mic = MicCapture(sample_rate=8000, chunk_size=256, channels=1)
    assert mic.sample_rate == 8000
    assert mic.chunk_size == 256


# ---------------------------------------------------------------------------
# Interface / lifecycle
# ---------------------------------------------------------------------------

def test_read_chunk_before_start_raises_clean_error():
    from audio_processing.capture import MicCapture, MicCaptureError
    mic = MicCapture()
    with pytest.raises(MicCaptureError):
        mic.read_chunk(timeout=0.1)


def test_start_stop_lifecycle(monkeypatch):
    from audio_processing.capture import MicCapture
    holder: Dict[str, object] = {}
    _install_fake_sounddevice(monkeypatch, holder)

    mic = MicCapture(chunk_size=4)
    assert mic.is_running is False
    mic.start()
    assert mic.is_running is True
    mic.stop()
    assert mic.is_running is False


def test_stop_is_idempotent(monkeypatch):
    from audio_processing.capture import MicCapture
    holder: Dict[str, object] = {}
    _install_fake_sounddevice(monkeypatch, holder)

    mic = MicCapture()
    mic.start()
    mic.stop()
    mic.stop()  # must not raise
    assert mic.is_running is False


def test_context_manager_starts_and_stops(monkeypatch):
    from audio_processing.capture import MicCapture
    holder: Dict[str, object] = {}
    _install_fake_sounddevice(monkeypatch, holder)

    with MicCapture(chunk_size=4) as mic:
        assert mic.is_running is True
    assert mic.is_running is False


# ---------------------------------------------------------------------------
# Streaming / chunking behaviour
# ---------------------------------------------------------------------------

def test_read_chunk_returns_shape_and_dtype_from_callback(monkeypatch):
    from audio_processing.capture import MicCapture
    holder: Dict[str, object] = {}
    _install_fake_sounddevice(monkeypatch, holder)

    mic = MicCapture(sample_rate=16000, chunk_size=4)
    mic.start()

    fake_frame = np.array([[0.1], [0.2], [0.3], [-0.4]], dtype=np.float32)
    holder["callback"](fake_frame, 4, None, None)  # simulate PortAudio delivering audio

    chunk = mic.read_chunk(timeout=1.0)
    assert isinstance(chunk, np.ndarray)
    assert chunk.dtype == np.float32
    assert chunk.shape == (4,)
    np.testing.assert_allclose(chunk, [0.1, 0.2, 0.3, -0.4], atol=1e-6)

    mic.stop()


def test_multiple_chunks_are_delivered_in_order(monkeypatch):
    from audio_processing.capture import MicCapture
    holder: Dict[str, object] = {}
    _install_fake_sounddevice(monkeypatch, holder)

    mic = MicCapture(chunk_size=2)
    mic.start()

    for i in range(5):
        frame = np.full((2, 1), fill_value=float(i), dtype=np.float32)
        holder["callback"](frame, 2, None, None)

    for i in range(5):
        chunk = mic.read_chunk(timeout=1.0)
        assert np.allclose(chunk, i)

    mic.stop()


def test_queue_drops_oldest_chunk_when_full(monkeypatch):
    """Consumer falling behind must not block the audio callback or grow memory."""
    from audio_processing.capture import MicCapture
    holder: Dict[str, object] = {}
    _install_fake_sounddevice(monkeypatch, holder)

    mic = MicCapture(chunk_size=1, queue_maxsize=2)
    mic.start()

    for i in range(5):
        holder["callback"](np.array([[float(i)]], dtype=np.float32), 1, None, None)

    # Only the last `queue_maxsize` chunks should remain (3, 4).
    first = mic.read_chunk(timeout=1.0)
    second = mic.read_chunk(timeout=1.0)
    assert float(first[0]) == 3.0
    assert float(second[0]) == 4.0

    mic.stop()


def test_read_chunk_times_out_when_no_data(monkeypatch):
    from audio_processing.capture import MicCapture, MicCaptureError
    holder: Dict[str, object] = {}
    _install_fake_sounddevice(monkeypatch, holder)

    mic = MicCapture()
    mic.start()
    with pytest.raises(MicCaptureError):
        mic.read_chunk(timeout=0.2)
    mic.stop()


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_start_raises_clean_error_when_sounddevice_not_installed(monkeypatch):
    from audio_processing.capture import MicCapture, MicCaptureError
    monkeypatch.setitem(sys.modules, "sounddevice", None)  # forces ImportError on import

    mic = MicCapture()
    with pytest.raises(MicCaptureError):
        mic.start()
    assert mic.is_running is False


def test_start_raises_clean_error_when_device_unavailable(monkeypatch):
    from audio_processing.capture import MicCapture, MicCaptureError
    holder: Dict[str, object] = {}
    _install_fake_sounddevice(monkeypatch, holder, raise_on_construct=True)

    mic = MicCapture()
    with pytest.raises(MicCaptureError):
        mic.start()
    assert mic.is_running is False


def test_start_raises_clean_error_when_stream_fails_to_start(monkeypatch):
    from audio_processing.capture import MicCapture, MicCaptureError
    holder: Dict[str, object] = {}
    _install_fake_sounddevice(monkeypatch, holder, raise_on_start=True)

    mic = MicCapture()
    with pytest.raises(MicCaptureError):
        mic.start()
    assert mic.is_running is False


# ---------------------------------------------------------------------------
# Backward-compatible one-shot record()
# ---------------------------------------------------------------------------

def test_record_backward_compatible(monkeypatch):
    fake_sd = types.ModuleType("sounddevice")

    def fake_rec(frames, samplerate, channels, dtype, device=None):
        return np.full((frames, channels), 0.5, dtype=np.float32)

    fake_sd.rec = fake_rec  # type: ignore[attr-defined]
    fake_sd.wait = lambda: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)

    from audio_processing.capture import MicCapture

    mic = MicCapture(sample_rate=1000, channels=1)
    audio = mic.record(0.1)
    assert audio.shape == (100,)
    assert audio.dtype == np.float32
    assert np.allclose(audio, 0.5)


def test_record_raises_clean_error_without_sounddevice(monkeypatch):
    from audio_processing.capture import MicCapture, MicCaptureError
    monkeypatch.setitem(sys.modules, "sounddevice", None)

    mic = MicCapture()
    with pytest.raises(MicCaptureError):
        mic.record(0.1)
