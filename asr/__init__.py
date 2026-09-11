"""
asr - local speech-to-text (faster-whisper). See asr.transcriber for
the full interface, backend rationale, and honesty notes.
"""
from .transcriber import (
    Transcriber,
    TranscriptionError,
    TranscriptionResult,
    get_transcriber,
    transcribe,
)

__all__ = [
    "Transcriber",
    "TranscriptionError",
    "TranscriptionResult",
    "get_transcriber",
    "transcribe",
]
