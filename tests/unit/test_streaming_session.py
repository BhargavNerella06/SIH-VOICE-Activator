"""
tests/unit/test_streaming_session.py
=======================================
api.streaming: session/buffer/timing bookkeeping, tested directly
(no HTTP, no FastAPI TestClient). This is the "transport" layer that
must stay independent of ASR/command interpretation -- these tests
confirm it works with nothing but raw numpy chunks.

The second half of this file covers the Stage 3 KWS-integration
milestone: feeding chunks to a wake-word detector as an independent
stream consumer, pre-roll retrieval at detection, and the resulting
state machine. A fake/mock detector is used throughout -- these tests
deliberately never touch keyword_spotting.detector.KeywordDetector or
any real (untrained) NOVA model; see this file's FakeDetector below and
docs/STAGE_3_STREAMING.md for why that's the point, not a shortcut.
"""

import time

import numpy as np
import pytest

from api.streaming import (
    SessionState,
    StreamSessionError,
    clear_all_sessions,
    close_session,
    create_session,
    drop_session,
    get_session,
)


@pytest.fixture(autouse=True)
def _clean_sessions():
    clear_all_sessions()
    yield
    clear_all_sessions()


def test_create_session_returns_unique_ids():
    a = create_session(sample_rate=16000)
    b = create_session(sample_rate=16000)
    assert a.session_id != b.session_id
    assert a.sample_rate == 16000


def test_get_session_returns_the_same_instance():
    session = create_session(sample_rate=16000)
    fetched = get_session(session.session_id)
    assert fetched is session


def test_get_unknown_session_raises():
    with pytest.raises(StreamSessionError):
        get_session("does-not-exist")


def test_add_chunk_accumulates_into_assembly_in_order():
    session = create_session(sample_rate=16000)
    session.add_chunk(np.array([1, 2, 3], dtype=np.float32))
    session.add_chunk(np.array([4, 5], dtype=np.float32))
    np.testing.assert_array_equal(session.assembled_waveform(), [1, 2, 3, 4, 5])
    assert session.total_samples == 5


def test_add_chunk_records_sequence_and_timing():
    session = create_session(sample_rate=16000)
    t1 = session.add_chunk(np.zeros(10, dtype=np.float32), capture_ts=1.0, send_ts=1.1)
    t2 = session.add_chunk(np.zeros(5, dtype=np.float32), capture_ts=2.0, send_ts=2.1)
    assert t1.seq == 0
    assert t2.seq == 1
    assert t1.n_samples == 10
    assert t2.n_samples == 5
    assert t1.capture_ts == 1.0
    assert t1.send_ts == 1.1
    assert t1.receive_ts >= t1.send_ts  # server always receives after client sends (same clock here)
    assert len(session.chunk_timings) == 2


def test_audio_duration_matches_total_samples_over_sample_rate():
    session = create_session(sample_rate=1000)
    session.add_chunk(np.zeros(500, dtype=np.float32))
    session.add_chunk(np.zeros(250, dtype=np.float32))
    assert session.audio_duration_seconds == pytest.approx(0.75)


def test_preroll_buffer_is_fed_alongside_assembly():
    """The pre-roll ring buffer must receive every chunk too (architecture
    slot for a future wake-word event), independent of the assembly buffer."""
    session = create_session(sample_rate=16000)
    session.add_chunk(np.ones(100, dtype=np.float32))
    assert session.preroll.filled_samples == 100


def test_empty_session_has_zero_duration_and_empty_waveform():
    session = create_session(sample_rate=16000)
    assert session.total_samples == 0
    assert session.audio_duration_seconds == 0.0
    np.testing.assert_array_equal(session.assembled_waveform(), np.zeros(0, dtype=np.float32))


def test_add_chunk_after_close_raises():
    session = create_session(sample_rate=16000)
    session.add_chunk(np.zeros(10, dtype=np.float32))
    close_session(session.session_id)
    with pytest.raises(StreamSessionError):
        session.add_chunk(np.zeros(10, dtype=np.float32))


def test_close_session_is_reflected_on_the_session_object():
    session = create_session(sample_rate=16000)
    assert session.closed is False
    close_session(session.session_id)
    assert session.closed is True
    assert session.state == SessionState.FINALIZED


def test_session_without_detector_starts_in_collecting_command():
    """No detector => old (pre-KWS-integration) behaviour: nothing to wait
    for, so /stop stays usable for dev testing without a trained model."""
    session = create_session(sample_rate=16000)
    assert session.detector is None
    assert session.state == SessionState.COLLECTING_COMMAND


def test_drop_session_removes_it_from_the_store():
    session = create_session(sample_rate=16000)
    drop_session(session.session_id)
    with pytest.raises(StreamSessionError):
        get_session(session.session_id)


def test_drop_session_is_safe_if_already_gone():
    drop_session("never-existed")  # must not raise


# ---------------------------------------------------------------------------
# KWS integration: fake/mock detector test doubles
#
# These deliberately duck-type keyword_spotting.detector.KeywordDetector's
# detect_chunk() contract (Optional[result] with a `.status` and optional
# `.confidence`) without importing keyword_spotting at all -- see
# api/streaming.py's module docstring for why that decoupling matters.
# ---------------------------------------------------------------------------

class _FakeKeywordResult:
    def __init__(self, status: str, confidence=None):
        self.status = status
        self.confidence = confidence


class FakeDetector:
    """
    Fires a DETECTED result on the call whose 0-based index equals
    `detect_on_call`. Every other call returns NO_DETECTION. Records
    every chunk it was given, so tests can assert chunks actually reach
    the detector as an independent stream consumer.
    """

    def __init__(self, detect_on_call=None, confidence: float = 0.97):
        self.detect_on_call = detect_on_call
        self.confidence = confidence
        self.calls = []

    def detect_chunk(self, chunk: np.ndarray):
        idx = len(self.calls)
        self.calls.append(np.asarray(chunk).copy())
        if self.detect_on_call is not None and idx == self.detect_on_call:
            return _FakeKeywordResult(status="DETECTED", confidence=self.confidence)
        return _FakeKeywordResult(status="NO_DETECTION")


class NeverDetects(FakeDetector):
    def __init__(self):
        super().__init__(detect_on_call=None)


class RaisingDetector:
    """Simulates a buggy/misconfigured detector that raises on every call."""

    def __init__(self, message: str = "simulated detector failure"):
        self.message = message
        self.calls = 0

    def detect_chunk(self, chunk: np.ndarray):
        self.calls += 1
        raise RuntimeError(self.message)


# ---------------------------------------------------------------------------
# Chunks reaching the KWS consumer
# ---------------------------------------------------------------------------

def test_chunks_reach_the_detector_as_an_independent_consumer():
    detector = NeverDetects()
    session = create_session(sample_rate=16000, detector=detector)
    for i in range(4):
        session.add_chunk(np.full(10, float(i), dtype=np.float32))
    assert len(detector.calls) == 4
    # The detector's copy of each chunk must match what was sent.
    for i, call in enumerate(detector.calls):
        np.testing.assert_array_equal(call, np.full(10, float(i), dtype=np.float32))


def test_detector_receives_chunks_regardless_of_preroll_also_receiving_them():
    """Pre-roll and the detector are independent consumers of the same
    chunk -- feeding one must not prevent or alter what the other sees."""
    detector = NeverDetects()
    session = create_session(sample_rate=16000, detector=detector)
    session.add_chunk(np.ones(50, dtype=np.float32))
    assert len(detector.calls) == 1
    assert session.preroll.filled_samples == 50


# ---------------------------------------------------------------------------
# No detection -> no wake event
# ---------------------------------------------------------------------------

def test_no_detection_leaves_session_waiting_with_no_wake_event():
    detector = NeverDetects()
    session = create_session(sample_rate=16000, detector=detector)
    for _ in range(5):
        session.add_chunk(np.zeros(10, dtype=np.float32))
    assert session.state == SessionState.WAITING_FOR_WAKE_WORD
    assert session.wake_word_detected_ts is None
    assert session.wake_word_confidence is None
    # Nothing has been gated into the ASR assembly yet.
    assert session.assembled_waveform().size == 0


# ---------------------------------------------------------------------------
# Mocked detection -> wake event
# ---------------------------------------------------------------------------

def test_mocked_detection_fires_wake_event_and_transitions_state():
    detector = FakeDetector(detect_on_call=1, confidence=0.88)
    session = create_session(sample_rate=16000, detector=detector)
    session.add_chunk(np.zeros(10, dtype=np.float32))   # call 0: no detection
    assert session.state == SessionState.WAITING_FOR_WAKE_WORD
    assert session.wake_word_detected_ts is None

    before = time.time()
    session.add_chunk(np.zeros(10, dtype=np.float32))   # call 1: DETECTED
    after = time.time()

    assert session.state == SessionState.COLLECTING_COMMAND
    assert session.wake_word_detected_ts is not None
    assert before <= session.wake_word_detected_ts <= after
    assert session.wake_word_confidence == 0.88


# ---------------------------------------------------------------------------
# Pre-roll retrieval at detection / post-detection preservation / no
# duplication at the detection boundary
# ---------------------------------------------------------------------------

def test_preroll_is_retrieved_into_assembly_at_the_moment_of_detection():
    # capacity_seconds(0.5) * sample_rate(10) = 5-sample pre-roll capacity.
    detector = FakeDetector(detect_on_call=2)
    session = create_session(sample_rate=10, detector=detector)

    session.add_chunk(np.array([1.0, 2.0], dtype=np.float32))  # call 0
    session.add_chunk(np.array([3.0, 4.0], dtype=np.float32))  # call 1
    session.add_chunk(np.array([5.0, 6.0], dtype=np.float32))  # call 2: DETECTED

    # 6 samples written total, capacity 5 -> oldest sample (1.0) dropped.
    np.testing.assert_array_equal(session.assembled_waveform(), [2.0, 3.0, 4.0, 5.0, 6.0])


def test_post_detection_audio_is_preserved_without_duplicating_the_boundary_chunk():
    detector = FakeDetector(detect_on_call=2)
    session = create_session(sample_rate=10, detector=detector)

    session.add_chunk(np.array([1.0, 2.0], dtype=np.float32))   # call 0
    session.add_chunk(np.array([3.0, 4.0], dtype=np.float32))   # call 1
    session.add_chunk(np.array([5.0, 6.0], dtype=np.float32))   # call 2: DETECTED, seeds [2,3,4,5,6]
    session.add_chunk(np.array([7.0, 8.0], dtype=np.float32))   # post-detection: appended directly

    result = session.assembled_waveform()
    np.testing.assert_array_equal(result, [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    # The detecting chunk's samples (5.0, 6.0) appear exactly once each.
    assert list(result).count(5.0) == 1
    assert list(result).count(6.0) == 1


def test_multiple_chunks_before_and_after_detection():
    detector = FakeDetector(detect_on_call=3)
    session = create_session(sample_rate=16000, detector=detector)  # large capacity: nothing evicted

    for i in range(3):  # calls 0, 1, 2: no detection
        session.add_chunk(np.full(4, float(i), dtype=np.float32))
    session.add_chunk(np.full(4, 3.0, dtype=np.float32))  # call 3: DETECTED
    for i in range(4, 6):  # post-detection
        session.add_chunk(np.full(4, float(i), dtype=np.float32))

    assert session.state == SessionState.COLLECTING_COMMAND
    expected = np.concatenate([np.full(4, float(i), dtype=np.float32) for i in range(6)])
    np.testing.assert_array_equal(session.assembled_waveform(), expected)
    # Detector stopped being consulted once a detection fired (single
    # wake-word-per-session model): only calls 0-3 ever reached it.
    assert len(detector.calls) == 4


# ---------------------------------------------------------------------------
# Finalization after detection / before detection
# ---------------------------------------------------------------------------

def test_finalization_after_detection_preserves_collected_command_audio():
    detector = FakeDetector(detect_on_call=0)
    session = create_session(sample_rate=16000, detector=detector)
    session.add_chunk(np.full(10, 1.0, dtype=np.float32))  # DETECTED immediately
    session.add_chunk(np.full(10, 2.0, dtype=np.float32))  # post-detection

    close_session(session.session_id)

    assert session.state == SessionState.FINALIZED
    assert session.closed is True
    assert session.assembled_waveform().size == 20  # nothing lost by finalizing


def test_session_stopped_before_detection_has_empty_command_audio():
    detector = NeverDetects()
    session = create_session(sample_rate=16000, detector=detector)
    session.add_chunk(np.ones(10, dtype=np.float32))
    session.add_chunk(np.ones(10, dtype=np.float32))

    close_session(session.session_id)

    assert session.state == SessionState.FINALIZED
    assert session.wake_word_detected_ts is None
    # No wake word ever fired, so no command audio was ever gated in --
    # ASR (at the router level) will see an empty waveform, exactly like
    # the existing "zero chunks" case already covered for no-detector
    # sessions in tests/integration/test_stream_api.py.
    assert session.assembled_waveform().size == 0


# ---------------------------------------------------------------------------
# Detector errors are handled safely
# ---------------------------------------------------------------------------

def test_detector_exception_does_not_crash_add_chunk():
    detector = RaisingDetector("boom")
    session = create_session(sample_rate=16000, detector=detector)

    timing = session.add_chunk(np.zeros(10, dtype=np.float32))  # must not raise

    assert timing.n_samples == 10
    assert detector.calls == 1
    assert session.detector_error is not None
    assert "boom" in session.detector_error
    assert session.state == SessionState.WAITING_FOR_WAKE_WORD
    assert session.wake_word_detected_ts is None
    assert session.assembled_waveform().size == 0


def test_detector_exception_on_every_call_keeps_session_usable():
    detector = RaisingDetector()
    session = create_session(sample_rate=16000, detector=detector)
    for _ in range(3):
        session.add_chunk(np.zeros(10, dtype=np.float32))

    assert detector.calls == 3
    assert session.state == SessionState.WAITING_FOR_WAKE_WORD
    # Pre-roll still works even though the detector never succeeds.
    assert session.preroll.filled_samples == 30

    # The session can still be finalized cleanly afterwards.
    close_session(session.session_id)
    assert session.state == SessionState.FINALIZED
