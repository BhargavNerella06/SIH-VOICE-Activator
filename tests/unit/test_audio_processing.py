"""tests/unit/test_audio_processing.py - Tests for the audio_processing module."""

import numpy as np
import pytest


def test_ensure_mono_stereo():
    from audio_processing.utils import ensure_mono
    stereo = np.ones((1000, 2), dtype=np.float32)
    mono = ensure_mono(stereo)
    assert mono.ndim == 1
    assert len(mono) == 1000


def test_ensure_mono_already_mono():
    from audio_processing.utils import ensure_mono
    mono = np.ones(1000, dtype=np.float32)
    result = ensure_mono(mono)
    assert result.ndim == 1


def test_normalize_audio():
    from audio_processing.utils import normalize_audio
    y = np.array([0.0, 0.5, -1.0, 2.0], dtype=np.float32)
    norm = normalize_audio(y)
    assert np.max(np.abs(norm)) <= 1.0 + 1e-6


def test_normalize_audio_silent():
    from audio_processing.utils import normalize_audio
    y = np.zeros(100, dtype=np.float32)
    result = normalize_audio(y)
    assert np.allclose(result, 0.0)


def test_audio_preprocessor_process():
    from audio_processing.preprocess import AudioPreprocessor
    proc = AudioPreprocessor(target_sr=8000)
    y = np.random.randn(16000).astype(np.float32)  # 1s @ 16kHz
    out, sr = proc.process(y, 16000)
    assert sr == 8000
    assert out.dtype == np.float32
    assert out.ndim == 1


def test_resample_if_needed_changes_length():
    from audio_processing.utils import resample_if_needed
    y = np.random.randn(16000).astype(np.float32)  # 1s @ 16kHz
    out = resample_if_needed(y, 16000, 8000)
    assert abs(len(out) - 8000) <= 10  # ~1s @ 8kHz, allow resampler edge slack


def test_resample_if_needed_noop_when_same_rate():
    from audio_processing.utils import resample_if_needed
    y = np.random.randn(8000).astype(np.float32)
    out = resample_if_needed(y, 8000, 8000)
    assert out is y


def test_vad_is_speech():
    from audio_processing.vad import VoiceActivityDetector
    vad = VoiceActivityDetector(threshold=0.01)
    loud = np.random.randn(1000).astype(np.float32)
    silent = np.zeros(1000, dtype=np.float32)
    assert vad.is_speech(loud) is True
    assert vad.is_speech(silent) is False
