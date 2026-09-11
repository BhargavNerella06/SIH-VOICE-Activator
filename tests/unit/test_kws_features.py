"""
tests/unit/test_kws_features.py
==================================
Tests for keyword_spotting.features (log-Mel extraction used by both
training and inference).
"""

import numpy as np


def test_fixed_length_waveform_pads_short_input():
    from keyword_spotting.features import fixed_length_waveform
    y = np.ones(8000, dtype=np.float32)
    out = fixed_length_waveform(y, sample_rate=16000, window_seconds=1.0)
    assert len(out) == 16000
    assert np.all(out[8000:] == 0.0)


def test_fixed_length_waveform_trims_long_input():
    from keyword_spotting.features import fixed_length_waveform
    y = np.ones(20000, dtype=np.float32)
    out = fixed_length_waveform(y, sample_rate=16000, window_seconds=1.0)
    assert len(out) == 16000


def test_fixed_length_waveform_noop_when_exact():
    from keyword_spotting.features import fixed_length_waveform
    y = np.ones(16000, dtype=np.float32)
    out = fixed_length_waveform(y, sample_rate=16000, window_seconds=1.0)
    assert len(out) == 16000
    np.testing.assert_array_equal(out, y)


def test_log_mel_spectrogram_shape_and_dtype():
    from keyword_spotting.features import log_mel_spectrogram
    y = np.random.randn(16000).astype(np.float32)
    feat = log_mel_spectrogram(y, sample_rate=16000, n_mels=40)
    assert feat.dtype == np.float32
    assert feat.shape[0] == 40
    assert feat.shape[1] > 0


def test_extract_features_shape_for_default_window():
    from keyword_spotting.features import (
        DEFAULT_N_MELS,
        DEFAULT_SAMPLE_RATE,
        DEFAULT_WINDOW_SECONDS,
        extract_features,
    )
    y = np.zeros(int(DEFAULT_SAMPLE_RATE * DEFAULT_WINDOW_SECONDS), dtype=np.float32)
    feat = extract_features(y, sample_rate=DEFAULT_SAMPLE_RATE)
    assert feat.shape[0] == DEFAULT_N_MELS


def test_extract_features_deterministic():
    from keyword_spotting.features import extract_features
    y = np.random.RandomState(0).randn(16000).astype(np.float32)
    f1 = extract_features(y, sample_rate=16000)
    f2 = extract_features(y, sample_rate=16000)
    np.testing.assert_allclose(f1, f2)


def test_extract_features_handles_short_and_long_input_the_same_shape():
    from keyword_spotting.features import extract_features
    short = np.random.randn(4000).astype(np.float32)
    long = np.random.randn(32000).astype(np.float32)
    f_short = extract_features(short, sample_rate=16000)
    f_long = extract_features(long, sample_rate=16000)
    assert f_short.shape == f_long.shape
