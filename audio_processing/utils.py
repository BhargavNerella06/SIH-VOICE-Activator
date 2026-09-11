"""
audio_processing.utils
======================
Low-level audio helpers.
Ported verbatim from structure/demo/utils.py (that file is left unchanged).
"""

import os

import librosa
import numpy as np
import soundfile as sf


def ensure_mono(y: np.ndarray) -> np.ndarray:
    """Convert multi-channel audio to mono by averaging channels."""
    if y.ndim == 1:
        return y
    return np.mean(y, axis=1)


def resample_if_needed(
    y: np.ndarray, orig_sr: int, target_sr: int
) -> np.ndarray:
    """Resample y from orig_sr to target_sr if they differ."""
    if orig_sr == target_sr:
        return y
    return librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)


def normalize_audio(y: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Peak-normalise audio to [-1, 1]."""
    m = np.max(np.abs(y))
    if m < eps:
        return y
    return y / (m + eps)


def save_wav_int16(path: str, y: np.ndarray, sr: int) -> None:
    """Normalise then write a 16-bit PCM WAV file."""
    y = normalize_audio(y)
    y_int16 = (y * 32767).astype("int16")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    sf.write(path, y_int16, sr, subtype="PCM_16")
