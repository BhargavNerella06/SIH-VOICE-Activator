"""
audio_processing
================
Audio capture, preprocessing, and VAD utilities.

Ported from structure/demo/utils.py and structure/demo/speaker_detection.py.
The originals in structure/demo/ are left untouched.
"""
from .utils import ensure_mono, normalize_audio, resample_if_needed, save_wav_int16
from .preprocess import AudioPreprocessor
from .capture import MicCapture, MicCaptureError
from .session import (
    AudioSessionError,
    AudioSessionManager,
    AudioSessionState,
    CommandAudioSegment,
)

__all__ = [
    "ensure_mono",
    "normalize_audio",
    "resample_if_needed",
    "save_wav_int16",
    "AudioPreprocessor",
    "MicCapture",
    "MicCaptureError",
    "AudioSessionManager",
    "AudioSessionState",
    "AudioSessionError",
    "CommandAudioSegment",
]
