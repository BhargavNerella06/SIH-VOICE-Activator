"""
audio_processing.preprocess
============================
AudioPreprocessor - reads, mono-converts, resamples, and normalises audio
before it is passed to the separation engine.
"""

from __future__ import annotations

import numpy as np
import soundfile as sf

from .utils import ensure_mono, normalize_audio, resample_if_needed


class AudioPreprocessor:
    """
    Prepare raw audio (file or array) for Conv-TasNet inference.

    Parameters
    ----------
    target_sr : int
        Sample rate the downstream model expects (default 8000, matching
        the Conv-TasNet training configuration).
    """

    def __init__(self, target_sr: int = 8000) -> None:
        self.target_sr = target_sr

    def load_file(self, path: str) -> tuple[np.ndarray, int]:
        """
        Load a WAV file and return (mono_float32_array, sample_rate).
        Resamples to target_sr and normalises.
        """
        y, sr = sf.read(path)
        return self.process(y, sr)

    def process(
        self, y: np.ndarray, sr: int
    ) -> tuple[np.ndarray, int]:
        """
        Convert to mono float32, resample, and peak-normalise.

        Returns
        -------
        (processed_waveform, effective_sample_rate)
        """
        y = ensure_mono(y).astype(np.float32)
        if sr != self.target_sr:
            y = resample_if_needed(y, sr, self.target_sr)
            sr = self.target_sr
        y = normalize_audio(y)
        return y, sr
