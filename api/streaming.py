"""
api.streaming
==============
Transport-layer session/buffer bookkeeping for the Stage 3 development-
machine audio streaming path.

Deliberately separate from ASR and command interpretation
------------------------------------------------------------
This module only knows how to accumulate PCM audio chunks into a
buffer and record when things happened. It never imports
asr.transcriber or command.interpreter, and it has no idea what a
"transcript" or an "action" is -- those are the FastAPI router's job
(api/routers/stream.py), which calls into this module purely for
session/timing bookkeeping and calls the existing, unmodified ASR and
command modules separately once a stream is finalized.

The reason this separation matters for Stage 3 specifically: today the
client sending chunks is a Python script simulating a microphone
(scripts/stream_demo.py). Later, an ESP32 device will send the same
byte format to the same HTTP endpoints. As long as the wire format
(raw little-endian PCM16 chunks, see audio_processing.pcm) doesn't
change, nothing in this module or in api/routers/stream.py needs to
change either -- only the client changes. Keeping session/transport
bookkeeping free of ASR-specific logic is what makes that true.

KWS integration (this stage)
------------------------------
`StreamSession` can optionally be given a wake-word detector -- any
object with a `detect_chunk(chunk) -> Optional[result]` method, where
`result` (when not None) exposes a `.status` attribute equal to
"DETECTED" to signal a wake-word event, and optionally a `.confidence`
attribute. This is exactly the shape of
`keyword_spotting.detector.KeywordDetector.detect_chunk()`'s return
value (`Optional[KeywordResult]`) -- but this module does NOT import
`keyword_spotting` anywhere. It only relies on that duck-typed shape,
so it works identically with a real KeywordDetector, a fake/mock test
double, or no detector at all (`detector=None`, the default -- see
`SessionState` below for what that means for the resulting behaviour).

IMPORTANT: `keyword_spotting.detector.KeywordDetector` today runs on
openWakeWord's frozen general-purpose embedding backbone, which is
detector *infrastructure* -- it is NOT evidence that a real NOVA model
has been trained or validated. `keyword_spotting/model.py`'s TinyKWSNet
remains untrained. No detection performed through this integration
implies anything about real NOVA accuracy. See
docs/STAGE_3_STREAMING.md.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from audio_processing.ring_buffer import AudioRingBuffer

logger = logging.getLogger(__name__)

# How much trailing audio the pre-roll ring buffer retains. See
# audio_processing/ring_buffer.py's module docstring for why this
# exists; this stage starts consuming it at a real detection event
# (see SessionState below), still without changing this constant.
DEFAULT_PREROLL_SECONDS = 0.5


class SessionState(str, Enum):
    """
    Explicit stream lifecycle states.

    WAITING_FOR_WAKE_WORD -> WAKE_WORD_DETECTED -> COLLECTING_COMMAND -> FINALIZED

    A session created WITHOUT a detector (detector=None, e.g. today's
    dev-testing/demo usage before any wake-word gating existed) starts
    directly in COLLECTING_COMMAND: there is nothing to wait for, so
    every chunk is collected for ASR exactly as before this stage. This
    is what keeps /stop usable for development testing without a
    trained NOVA model (see api/routers/stream.py -- unchanged this
    stage, no detector is attached by default).
    """
    WAITING_FOR_WAKE_WORD = "WAITING_FOR_WAKE_WORD"
    WAKE_WORD_DETECTED = "WAKE_WORD_DETECTED"
    COLLECTING_COMMAND = "COLLECTING_COMMAND"
    FINALIZED = "FINALIZED"


class StreamSessionError(RuntimeError):
    """Raised for invalid session operations (unknown id, wrong lifecycle state)."""


@dataclass
class ChunkTiming:
    """Timing metadata for one received chunk."""
    seq: int
    n_samples: int
    receive_ts: float                    # server wall-clock time.time() on arrival
    capture_ts: Optional[float] = None   # client-reported: when audio was captured
    send_ts: Optional[float] = None      # client-reported: when the HTTP request was sent


@dataclass
class StreamSession:
    """
    Server-side state for one in-progress (or finalized) audio stream.

    `assembly` accumulates the audio that will eventually go to ASR;
    `preroll` is a fixed-size ring buffer fed every chunk unconditionally,
    only ever holding the most recent `DEFAULT_PREROLL_SECONDS` -- see
    the module docstring above.

    detector : optional wake-word detector (duck-typed, see module
        docstring). When None (the default), the session starts -- and
        stays -- in COLLECTING_COMMAND: every chunk goes straight into
        `assembly`, exactly as before this stage's KWS integration
        existed. When a detector is given, the session starts in
        WAITING_FOR_WAKE_WORD and gates `assembly` on a real detection
        event (see add_chunk).
    """
    session_id: str
    sample_rate: int
    created_ts: float = field(default_factory=time.time)
    detector: Optional[Any] = None
    assembly: List[np.ndarray] = field(default_factory=list)
    preroll: AudioRingBuffer = field(init=False)
    chunk_timings: List[ChunkTiming] = field(default_factory=list)
    closed: bool = False
    state: SessionState = field(init=False)

    # Wake-word detection bookkeeping.
    wake_word_detected_ts: Optional[float] = None
    wake_word_confidence: Optional[float] = None
    detector_error: Optional[str] = None

    # Stage timestamps, filled in by the router as the stream is finalized.
    stop_received_ts: Optional[float] = None
    asr_start_ts: Optional[float] = None
    transcript_ready_ts: Optional[float] = None
    command_parsed_ts: Optional[float] = None

    def __post_init__(self) -> None:
        self.preroll = AudioRingBuffer(
            capacity_seconds=DEFAULT_PREROLL_SECONDS, sample_rate=self.sample_rate
        )
        self.state = (
            SessionState.COLLECTING_COMMAND
            if self.detector is None
            else SessionState.WAITING_FOR_WAKE_WORD
        )

    @property
    def total_samples(self) -> int:
        return sum(len(c) for c in self.assembly)

    @property
    def audio_duration_seconds(self) -> float:
        return self.total_samples / float(self.sample_rate)

    def _run_detector(self, chunk: np.ndarray) -> Optional[Any]:
        """
        Run the attached detector on one chunk and return its result
        object only if it represents a genuine wake-word detection
        (`result.status == "DETECTED"`); otherwise returns None. Any
        exception raised by the detector is caught and recorded on
        `detector_error` rather than propagating -- a buggy or untrained
        detector must never crash the streaming session (Stage 3
        KWS-integration requirement).
        """
        if self.detector is None:
            return None
        try:
            result = self.detector.detect_chunk(chunk)
        except Exception as exc:  # noqa: BLE001 - deliberately broad, see docstring
            self.detector_error = str(exc)
            logger.warning(
                "KWS detector raised during detect_chunk() for session %s; "
                "treating this chunk as no detection: %s",
                self.session_id, exc,
            )
            return None
        if result is not None and getattr(result, "status", None) == "DETECTED":
            return result
        return None

    def add_chunk(
        self,
        chunk: np.ndarray,
        capture_ts: Optional[float] = None,
        send_ts: Optional[float] = None,
    ) -> ChunkTiming:
        if self.closed:
            raise StreamSessionError(
                f"Session {self.session_id} is already closed; cannot accept more chunks."
            )
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        timing = ChunkTiming(
            seq=len(self.chunk_timings),
            n_samples=int(len(chunk)),
            receive_ts=time.time(),
            capture_ts=capture_ts,
            send_ts=send_ts,
        )
        self.chunk_timings.append(timing)
        # Every chunk feeds the pre-roll buffer unconditionally, regardless
        # of detection state -- it is an independent consumer of the
        # stream, same as the detector (see this module's docstring).
        self.preroll.write(chunk)

        if self.state == SessionState.WAITING_FOR_WAKE_WORD:
            detection = self._run_detector(chunk)
            if detection is not None:
                self.wake_word_detected_ts = time.time()
                self.wake_word_confidence = getattr(detection, "confidence", None)
                self.state = SessionState.WAKE_WORD_DETECTED
                # Seed the command-audio assembly from the pre-roll
                # snapshot. The chunk that just triggered detection was
                # already written into `preroll` above, so it is included
                # here exactly once -- it must NOT also be appended below,
                # or it would be duplicated at the detection boundary.
                self.assembly = [self.preroll.read()]
                self.state = SessionState.COLLECTING_COMMAND
            # else: still waiting -- this chunk contributed only to the
            # pre-roll buffer, not to the ASR assembly.
        else:
            # COLLECTING_COMMAND (with or without a detector): accumulate
            # every subsequent chunk for ASR, preserving post-detection
            # command audio for as long as the client keeps streaming.
            self.assembly.append(chunk)

        return timing

    def assembled_waveform(self) -> np.ndarray:
        if not self.assembly:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self.assembly)


# ---------------------------------------------------------------------------
# Process-wide session store (mirrors the cache pattern used by
# voice_separation.engine / keyword_spotting.detector / asr.transcriber)
# ---------------------------------------------------------------------------

_sessions: Dict[str, StreamSession] = {}
_sessions_lock = threading.Lock()


def create_session(sample_rate: int, detector: Optional[Any] = None) -> StreamSession:
    """
    Create a new session. `detector` is optional (default None) and, if
    given, must be a duck-typed wake-word detector -- see this module's
    docstring. Passing None (the default) preserves the exact pre-KWS-
    integration behaviour: every chunk is collected for ASR immediately,
    with no gating on a detection event. api/routers/stream.py does not
    pass a detector today -- see docs/STAGE_3_STREAMING.md for why.
    """
    session = StreamSession(
        session_id=uuid.uuid4().hex, sample_rate=sample_rate, detector=detector
    )
    with _sessions_lock:
        _sessions[session.session_id] = session
    return session


def get_session(session_id: str) -> StreamSession:
    with _sessions_lock:
        session = _sessions.get(session_id)
    if session is None:
        raise StreamSessionError(f"No such stream session: {session_id!r}")
    return session


def close_session(session_id: str) -> StreamSession:
    session = get_session(session_id)
    session.closed = True
    session.state = SessionState.FINALIZED
    return session


def drop_session(session_id: str) -> None:
    """Remove a session from the store entirely. Safe to call if already gone."""
    with _sessions_lock:
        _sessions.pop(session_id, None)


def clear_all_sessions() -> None:
    """Drop every session. Mainly useful for tests."""
    with _sessions_lock:
        _sessions.clear()
