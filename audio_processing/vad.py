"""
audio_processing.vad
=====================
VoiceActivityDetector - energy-based stub for Milestone 1.

Returns True (voice detected) for any non-silent frame.
A proper implementation (WebRTC VAD / Silero) is planned for a later
milestone.
"""

from __future__ import annotations

import numpy as np


class VoiceActivityDetector:
    """
    Milestone 1 stub: energy-threshold VAD.

    Parameters
    ----------
    threshold : float
        RMS energy below this is classified as silence.
    """

    def __init__(self, threshold: float = 0.01) -> None:
        self.threshold = threshold

    def is_speech(self, frame: np.ndarray) -> bool:
        """Return True if the frame contains speech (energy > threshold)."""
        rms = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))
        return rms >= self.threshold
