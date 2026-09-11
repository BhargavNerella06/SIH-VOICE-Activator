"""
audio_processing.session
===========================
AudioSessionManager - laptop-side orchestration between raw microphone
chunks and one complete command-audio segment ready for the existing
ASR interface.

    microphone chunks -> pre-roll buffer + KWS detector (LISTENING)
      -> wake word detected -> command-audio capture (COMMAND_CAPTURE)
      -> silence / min / max duration -> one finished segment (COMPLETE)

This module does NOT call ASR or the command interpreter -- it only
produces a `CommandAudioSegment` (a waveform + sample_rate, exactly the
shape `asr.transcriber.transcribe()`/`get_transcriber().transcribe()`
already accept). Wiring that segment into ASR and then
`command.interpreter.CommandInterpreter` is deliberately a separate,
later task. It also does not read from a microphone itself -- a caller
(a live mic loop via `audio_processing.capture.MicCapture`, or a test
feeding synthetic chunks) drives it by calling `process_chunk()`
repeatedly; this keeps the class itself free of any blocking I/O.

Reused, UNMODIFIED, existing infrastructure
----------------------------------------------
- `audio_processing.ring_buffer.AudioRingBuffer` for the pre-roll buffer.
- `audio_processing.vad.VoiceActivityDetector` for silence detection.
- `audio_processing.capture.MicCapture` is the intended real-world chunk
  source (see scripts/audio_session_smoke_test.py), but is never
  imported here -- this module only consumes numpy arrays, regardless
  of where they came from.

KWS decoupling
--------------
Same duck-typed contract already established by `api.streaming.
StreamSession` for exactly this reason: `detector` is any object
exposing `detect_chunk(chunk) -> Optional[result]`, where a non-None
result's `.status == "DETECTED"` signals a wake-word event, and
`.confidence` is read opportunistically if present. This module never
imports `keyword_spotting` -- a hand-written fake detector in tests
today, the real (TinyKWSNet-backed) `KeywordDetector` later, with zero
code change here. `detector=None` (the default) means "never detects
anything" -- useful for exercising pre-roll/buffering logic in
isolation without any wake-word concept at all.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional

import numpy as np

from .ring_buffer import AudioRingBuffer
from .vad import VoiceActivityDetector

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_RATE = 16000
# Raised from 0.5s -- measured (scripts/debug_command_capture.py) real-mic
# wake-to-DETECTED latency for the TinyKWSNet backend (1.0s analysis window
# + consecutive-frame debounce) can exceed 0.5s on its own, before even
# counting the user's natural pause after saying "NOVA". At 0.5s, the
# captured segment's own peak amplitude came back ~0.004 (silence) because
# capture was ending before real speech ever arrived. 1.0s gives more
# trailing context margin against that latency.
DEFAULT_PREROLL_SECONDS = 1.0
DEFAULT_SILENCE_SECONDS = 0.8
# Raised from 0.3s -- this is the exact bug this class's own docstring
# already named ("guards against a single quiet chunk right after the wake
# word ending the command instantly"): 0.3s is shorter than a normal pause
# after saying "NOVA", so DEFAULT_SILENCE_SECONDS worth of that pause alone
# could satisfy silence-termination before the user ever started the
# command. 1.2s comfortably covers "NOVA, <brief pause>, <command phrase>".
DEFAULT_MIN_COMMAND_SECONDS = 1.2
DEFAULT_MAX_COMMAND_SECONDS = 8.0
DEFAULT_VAD_THRESHOLD = 0.01


class AudioSessionState(str, Enum):
    """
    IDLE -> LISTENING -> COMMAND_CAPTURE -> PROCESSING -> COMPLETE
                                                              |
                                                           reset()
                                                              |
                                                              v
                                                          LISTENING

    ERROR is reachable from LISTENING or COMMAND_CAPTURE if a fed chunk
    can't be interpreted as audio at all (not from a detector merely
    failing to detect -- see AudioSessionManager._run_detector).

    Named distinctly from api.streaming.SessionState (a different,
    HTTP-streaming-oriented state machine with different state names)
    to avoid confusion if both are ever imported in the same scope.
    """
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    COMMAND_CAPTURE = "COMMAND_CAPTURE"
    PROCESSING = "PROCESSING"
    COMPLETE = "COMPLETE"
    ERROR = "ERROR"


class AudioSessionError(RuntimeError):
    """Raised for invalid session usage (wrong lifecycle state)."""


@dataclass
class CommandAudioSegment:
    """One complete command-audio segment, ready for the ASR interface."""
    waveform: np.ndarray
    sample_rate: int
    duration_seconds: float
    end_reason: str                        # "silence" | "max_duration"
    wake_word_detected_ts: Optional[float] = None
    command_end_ts: Optional[float] = None


class AudioSessionManager:
    """
    Parameters
    ----------
    sample_rate : int
        Sample rate of the audio that will be fed via process_chunk().
    detector : optional duck-typed KWS detector (see module docstring).
        None (default) means the session will stay in LISTENING forever
        (no wake word is ever recognized) -- useful for isolated
        pre-roll/buffering tests.
    preroll_seconds : float
        How much trailing audio the pre-roll ring buffer retains,
        recovered into the command segment at the moment of detection.
    silence_seconds : float
        Trailing silence duration (per VoiceActivityDetector) required,
        after at least `min_command_seconds` of command audio has been
        captured, to end the command by inferring the user has stopped
        speaking.
    min_command_seconds : float
        Minimum command-audio duration before silence-based termination
        is even considered -- guards against a single quiet chunk right
        after the wake word ending the command instantly.
    max_command_seconds : float
        Hard cap on command-audio duration: forces the segment to
        finalize once reached, regardless of silence. This is the
        "timeout" mechanism for a command that never goes quiet (e.g. a
        stuck-open mic, or genuinely long speech).
    vad_threshold : float
        RMS energy threshold passed to VoiceActivityDetector.

    All chunk processing (`process_chunk`) is synchronous, pure
    computation (no I/O, no sleeping, no locks) -- safe to call from a
    tight mic-reading loop without stalling it.
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        detector: Optional[Any] = None,
        preroll_seconds: float = DEFAULT_PREROLL_SECONDS,
        silence_seconds: float = DEFAULT_SILENCE_SECONDS,
        min_command_seconds: float = DEFAULT_MIN_COMMAND_SECONDS,
        max_command_seconds: float = DEFAULT_MAX_COMMAND_SECONDS,
        vad_threshold: float = DEFAULT_VAD_THRESHOLD,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")
        if min_command_seconds < 0:
            raise ValueError(f"min_command_seconds must be >= 0, got {min_command_seconds}")
        if max_command_seconds <= 0:
            raise ValueError(f"max_command_seconds must be positive, got {max_command_seconds}")
        if min_command_seconds > max_command_seconds:
            raise ValueError("min_command_seconds must not exceed max_command_seconds")
        if silence_seconds < 0:
            raise ValueError(f"silence_seconds must be >= 0, got {silence_seconds}")

        self.sample_rate = sample_rate
        self.detector = detector
        self.preroll_seconds = preroll_seconds
        self.silence_seconds = silence_seconds
        self.min_command_seconds = min_command_seconds
        self.max_command_seconds = max_command_seconds
        self.vad_threshold = vad_threshold

        self.preroll = AudioRingBuffer(capacity_seconds=preroll_seconds, sample_rate=sample_rate)
        self.vad = VoiceActivityDetector(threshold=vad_threshold)

        self.state = AudioSessionState.IDLE
        self.wake_word_detected_ts: Optional[float] = None
        self.wake_word_confidence: Optional[float] = None
        self.detector_error: Optional[str] = None
        self.error_message: Optional[str] = None
        self.result: Optional[CommandAudioSegment] = None

        self._command_chunks: List[np.ndarray] = []
        self._command_samples: int = 0
        self._silence_elapsed_seconds: float = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Transition IDLE -> LISTENING. Raises if not currently IDLE."""
        if self.state != AudioSessionState.IDLE:
            raise AudioSessionError(f"start() requires state IDLE, got {self.state.value}")
        self.state = AudioSessionState.LISTENING

    def reset(self) -> None:
        """
        Return to LISTENING for another wake-word/command cycle, from
        COMPLETE or ERROR. The pre-roll buffer is NOT cleared -- it is a
        continuously-fed ring buffer, not part of what a cycle "uses up".

        Also resets the detector's own streaming state (sliding-window
        buffer + consecutive-hit debounce counter) if it exposes
        `reset_stream()` -- keyword_spotting.detector.KeywordDetector
        does. Without this, stale "NOVA-scoring" audio still sitting in
        the detector's own 1-second analysis window at the moment of the
        previous DETECTED would immediately satisfy the debounce again
        the instant LISTENING resumes, firing a spurious re-detection
        before any genuinely new audio (esp. the user's next utterance)
        has had a chance to be heard -- observed live as a tight
        wake -> empty-capture loop every ~1.4s. Duck-typed and optional
        so detectors without this method (e.g. tests'/orchestrator's
        MockWakeTrigger) are unaffected.
        """
        if self.state not in (AudioSessionState.COMPLETE, AudioSessionState.ERROR):
            raise AudioSessionError(
                f"reset() requires COMPLETE or ERROR, got {self.state.value}"
            )
        self._command_chunks = []
        self._command_samples = 0
        self._silence_elapsed_seconds = 0.0
        self.wake_word_detected_ts = None
        self.wake_word_confidence = None
        self.error_message = None
        self.result = None
        if hasattr(self.detector, "reset_stream"):
            self.detector.reset_stream()
        self.state = AudioSessionState.LISTENING

    # ------------------------------------------------------------------
    # Chunk processing
    # ------------------------------------------------------------------

    def process_chunk(self, chunk: np.ndarray) -> AudioSessionState:
        """
        Feed one audio chunk to the session. Only valid while LISTENING
        or COMMAND_CAPTURE.

        Raises
        ------
        AudioSessionError
            If called while IDLE, PROCESSING, COMPLETE, or ERROR -- a
            caller lifecycle bug (e.g. forgetting to call start(), or
            not calling reset() after COMPLETE), not a runtime audio
            anomaly.

        A chunk that can't be interpreted as numeric audio (e.g. not
        array-like at all) does NOT raise -- it transitions the session
        to ERROR instead, so a live mic-reading loop can keep running
        and just check `.state == AudioSessionState.ERROR` rather than
        wrapping every call in try/except. An empty chunk (zero samples)
        is a harmless no-op, not an error.

        Returns
        -------
        AudioSessionState
            The state after processing this chunk.
        """
        if self.state not in (AudioSessionState.LISTENING, AudioSessionState.COMMAND_CAPTURE):
            raise AudioSessionError(
                f"process_chunk() requires LISTENING or COMMAND_CAPTURE, got {self.state.value}"
            )

        try:
            arr = np.asarray(chunk, dtype=np.float32).reshape(-1)
        except Exception as exc:
            self.state = AudioSessionState.ERROR
            self.error_message = f"Invalid audio chunk: {exc}"
            logger.error(self.error_message)
            return self.state

        if arr.size == 0:
            return self.state

        if self.state == AudioSessionState.LISTENING:
            self._process_listening_chunk(arr)
        else:
            self._process_command_chunk(arr)
        return self.state

    def _run_detector(self, chunk: np.ndarray) -> Optional[Any]:
        """
        Run the attached detector on one chunk. Returns its result only
        for a genuine wake-word detection (`.status == "DETECTED"`),
        else None. Any exception the detector raises is caught and
        recorded on `detector_error` rather than propagating -- a buggy
        or untrained detector must never crash the session (mirrors
        api.streaming.StreamSession._run_detector).
        """
        if self.detector is None:
            return None
        try:
            result = self.detector.detect_chunk(chunk)
        except Exception as exc:  # noqa: BLE001 - deliberately broad, see docstring
            self.detector_error = str(exc)
            logger.warning(
                "KWS detector raised during detect_chunk(); treating this "
                "chunk as no detection: %s", exc,
            )
            return None
        if result is not None and getattr(result, "status", None) == "DETECTED":
            return result
        return None

    def _process_listening_chunk(self, chunk: np.ndarray) -> None:
        # Every chunk feeds the pre-roll buffer unconditionally -- it is
        # an independent consumer of the stream, same as the detector.
        self.preroll.write(chunk)

        detection = self._run_detector(chunk)
        if detection is None:
            return

        self.wake_word_detected_ts = time.time()
        self.wake_word_confidence = getattr(detection, "confidence", None)
        # Seed command-audio capture from the pre-roll snapshot. The
        # chunk that just triggered detection was already written into
        # `preroll` above, so it is included here exactly once -- it
        # must NOT also be appended separately below, or it would be
        # duplicated at the detection boundary (same invariant as
        # api.streaming.StreamSession).
        seed = self.preroll.read()
        self._command_chunks = [seed]
        self._command_samples = int(seed.size)
        self._silence_elapsed_seconds = 0.0
        self.state = AudioSessionState.COMMAND_CAPTURE

    def _process_command_chunk(self, chunk: np.ndarray) -> None:
        self._command_chunks.append(chunk)
        self._command_samples += int(chunk.size)
        chunk_seconds = chunk.size / float(self.sample_rate)

        if self.vad.is_speech(chunk):
            self._silence_elapsed_seconds = 0.0
        else:
            self._silence_elapsed_seconds += chunk_seconds

        elapsed_seconds = self._command_samples / float(self.sample_rate)

        if elapsed_seconds >= self.max_command_seconds:
            self._finalize(end_reason="max_duration")
        elif (
            elapsed_seconds >= self.min_command_seconds
            and self._silence_elapsed_seconds >= self.silence_seconds
        ):
            self._finalize(end_reason="silence")

    def _finalize(self, end_reason: str) -> None:
        self.state = AudioSessionState.PROCESSING
        waveform = (
            np.concatenate(self._command_chunks)
            if self._command_chunks
            else np.zeros(0, dtype=np.float32)
        )
        self.result = CommandAudioSegment(
            waveform=waveform,
            sample_rate=self.sample_rate,
            duration_seconds=waveform.size / float(self.sample_rate),
            end_reason=end_reason,
            wake_word_detected_ts=self.wake_word_detected_ts,
            command_end_ts=time.time(),
        )
        self.state = AudioSessionState.COMPLETE
