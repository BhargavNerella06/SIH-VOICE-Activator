"""
tests/unit/test_asr_transcriber.py
====================================
Transcriber is a real, local ASR backend (Stage 3) built on
faster-whisper. These tests cover:
  - graceful degradation (NOT_IMPLEMENTED, never a fabricated
    transcript) when the backend is unavailable
  - the process-wide model cache
  - real end-to-end transcription, gated behind the actual
    faster-whisper dependency and a real Whisper model download

No NOVA-specific claim is made anywhere here: the model is generic
pretrained Whisper, exercised only on synthetic/silent audio or
locally-synthesized speech, never on NOVA wake-word data.
"""

from __future__ import annotations

import numpy as np
import pytest

from asr.transcriber import (
    Transcriber,
    TranscriptionError,
    clear_transcriber_cache,
    get_transcriber,
    peek_cached_transcriber,
    transcribe,
)


def _dummy_waveform(seconds: float = 1.0, sample_rate: int = 16000) -> np.ndarray:
    return np.zeros(int(seconds * sample_rate), dtype=np.float32)


# ---------------------------------------------------------------------------
# Graceful degradation: never fabricate a transcript
# ---------------------------------------------------------------------------

def test_reports_not_implemented_when_backend_missing(monkeypatch):
    """If faster-whisper can't be imported, status must be NOT_IMPLEMENTED,
    never a fake SUCCESS with invented text."""
    import asr.transcriber as mod

    real_import = __import__

    def _fake_import(name, *args, **kwargs):
        if name == "faster_whisper":
            raise ImportError("simulated: faster-whisper not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _fake_import)
    t = mod.Transcriber()
    assert not t.is_ready
    result = t.transcribe(_dummy_waveform(), sample_rate=16000)
    assert result.status == "NOT_IMPLEMENTED"
    assert result.text == ""


def test_reports_not_implemented_when_model_load_raises(monkeypatch):
    """If the model constructor itself raises (e.g. no network on first
    download), status must be NOT_IMPLEMENTED with the real error, not a
    silent crash or a fabricated result."""
    import asr.transcriber as mod

    class _FakeWhisperModel:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("simulated: no network to fetch model weights")

    fake_module = type("fake_faster_whisper", (), {"WhisperModel": _FakeWhisperModel})

    monkeypatch.setitem(__import__("sys").modules, "faster_whisper", fake_module)
    t = mod.Transcriber()
    assert not t.is_ready
    assert "simulated" in (t.load_error or "")
    result = t.transcribe(_dummy_waveform(), sample_rate=16000)
    assert result.status == "NOT_IMPLEMENTED"


# ---------------------------------------------------------------------------
# Process-wide cache (mirrors voice_separation.engine tests)
# ---------------------------------------------------------------------------

def test_get_transcriber_is_cached():
    clear_transcriber_cache()
    assert peek_cached_transcriber() is None

    t1 = get_transcriber()
    t2 = get_transcriber()
    assert t1 is t2
    assert peek_cached_transcriber() is t1
    clear_transcriber_cache()


def test_different_configs_get_different_cache_entries():
    clear_transcriber_cache()
    t1 = get_transcriber(model_name="tiny.en")
    t2 = get_transcriber(model_name="tiny.en", compute_type="float32")
    assert t1 is not t2
    clear_transcriber_cache()


# ---------------------------------------------------------------------------
# Real end-to-end transcription (requires faster-whisper + a downloaded
# model; skipped automatically if unavailable, e.g. no network/CI sandbox)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_transcriber():
    pytest.importorskip("faster_whisper", reason="faster-whisper not installed")
    t = Transcriber()
    if not t.is_ready:
        pytest.skip(f"ASR model not available in this environment: {t.load_error}")
    return t


def test_transcribe_silence_does_not_error(real_transcriber):
    """Silence must not crash the pipeline; status should be OK even if
    text ends up empty (no speech to transcribe)."""
    result = real_transcriber.transcribe(_dummy_waveform(2.0), sample_rate=16000)
    assert result.status == "OK"
    assert isinstance(result.text, str)


def test_transcribe_resamples_non_native_sample_rate(real_transcriber):
    """Input at a sample rate other than 16 kHz must be handled (resampled
    internally), not rejected."""
    waveform = _dummy_waveform(seconds=1.0, sample_rate=8000)
    result = real_transcriber.transcribe(waveform, sample_rate=8000)
    assert result.status == "OK"


# ---------------------------------------------------------------------------
# Simple functional interface: transcribe(audio, sample_rate) -> str
# ---------------------------------------------------------------------------

def test_transcribe_function_raises_when_backend_missing(monkeypatch):
    """The simple str-returning interface must raise a clear error, never
    return a fabricated/empty string silently, when ASR is unavailable."""
    import asr.transcriber as mod

    real_import = __import__

    def _fake_import(name, *args, **kwargs):
        if name == "faster_whisper":
            raise ImportError("simulated: faster-whisper not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _fake_import)
    clear_transcriber_cache()
    with pytest.raises(TranscriptionError):
        mod.transcribe(_dummy_waveform(), sample_rate=16000)
    clear_transcriber_cache()


def test_transcribe_function_raises_transcription_error_on_failure(monkeypatch):
    """A transcription-time failure (status='ERROR') must also surface as
    TranscriptionError, not a silently-wrong string."""
    import asr.transcriber as mod

    class _FakeTranscriber:
        is_ready = True

        def transcribe(self, waveform, sample_rate):
            return mod.TranscriptionResult(status="ERROR", message="simulated failure")

    monkeypatch.setattr(mod, "get_transcriber", lambda **kwargs: _FakeTranscriber())
    with pytest.raises(TranscriptionError, match="simulated failure"):
        mod.transcribe(_dummy_waveform(), sample_rate=16000)


def test_transcribe_function_returns_plain_string_on_success(monkeypatch):
    import asr.transcriber as mod

    class _FakeTranscriber:
        is_ready = True

        def transcribe(self, waveform, sample_rate):
            return mod.TranscriptionResult(status="OK", text="turn on the lights")

    monkeypatch.setattr(mod, "get_transcriber", lambda **kwargs: _FakeTranscriber())
    result = mod.transcribe(_dummy_waveform(), sample_rate=16000)
    assert result == "turn on the lights"
    assert isinstance(result, str)


def test_transcribe_function_uses_the_same_cache(real_transcriber):
    """The simple function must not bypass the process-wide model cache --
    calling it must not trigger a second model load."""
    clear_transcriber_cache()
    text = transcribe(_dummy_waveform(1.0), sample_rate=16000)
    assert isinstance(text, str)
    assert peek_cached_transcriber() is not None
    clear_transcriber_cache()
