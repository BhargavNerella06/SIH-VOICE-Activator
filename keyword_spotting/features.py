"""
keyword_spotting.features
==========================
Log-Mel spectrogram feature extraction, shared between training and
inference so the two paths can never drift apart.

Defaults are chosen to be lightweight and edge/MCU-friendly: a small
number of Mel bins and a short, fixed analysis window keep the feature
tensor (and therefore the model) tiny.
"""

from __future__ import annotations

import numpy as np
import librosa

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_N_MELS = 40
DEFAULT_WINDOW_MS = 25.0
DEFAULT_HOP_MS = 10.0
DEFAULT_WINDOW_SECONDS = 1.0  # fixed analysis-window length fed to the model


def fixed_length_waveform(
    waveform: np.ndarray,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
) -> np.ndarray:
    """Zero-pad or trim `waveform` to exactly `window_seconds` at `sample_rate`."""
    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    target_len = int(round(sample_rate * window_seconds))
    if len(waveform) == target_len:
        return waveform
    if len(waveform) > target_len:
        return waveform[:target_len]
    pad = np.zeros(target_len - len(waveform), dtype=np.float32)
    return np.concatenate([waveform, pad])


def log_mel_spectrogram(
    waveform: np.ndarray,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    n_mels: int = DEFAULT_N_MELS,
    window_ms: float = DEFAULT_WINDOW_MS,
    hop_ms: float = DEFAULT_HOP_MS,
) -> np.ndarray:
    """
    Compute a log-Mel spectrogram of `waveform`.

    Returns
    -------
    np.ndarray, shape (n_mels, n_frames), dtype float32
    """
    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    n_fft = max(32, int(round(sample_rate * window_ms / 1000.0)))
    hop_length = max(16, int(round(sample_rate * hop_ms / 1000.0)))

    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )
    ref = mel.max() if mel.max() > 0 else 1.0
    log_mel = librosa.power_to_db(mel, ref=ref)
    return log_mel.astype(np.float32)


def extract_features(
    waveform: np.ndarray,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    n_mels: int = DEFAULT_N_MELS,
) -> np.ndarray:
    """
    End-to-end feature extraction for one fixed-length analysis window:
    pad/trim to `window_seconds`, then compute a log-Mel spectrogram.

    Returns
    -------
    np.ndarray, shape (n_mels, n_frames), dtype float32
    """
    fixed = fixed_length_waveform(waveform, sample_rate=sample_rate, window_seconds=window_seconds)
    return log_mel_spectrogram(fixed, sample_rate=sample_rate, n_mels=n_mels)
