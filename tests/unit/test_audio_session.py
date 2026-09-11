"""
tests/unit/test_audio_session.py
===================================
Tests for audio_processing.session.AudioSessionManager, using only
synthetic audio and fake/mock KWS detectors -- no microphone, no real
NOVA model, no training. A safety check against the real (untrained)
KeywordDetector is included separately to prove the duck-typed
integration works without any special-casing, without claiming any
NOVA detection accuracy.
"""

from __future__ import annotations

import numpy as np
import pytest

from audio_processing.session import (
    AudioSessionError,
    AudioSessionManager,
    AudioSessionState,
    CommandAudioSegment,
)


# ---------------------------------------------------------------------------
# Fake detector test doubles (duck-typed, mirrors
# tests/unit/test_streaming_session.py's FakeDetector/NeverDetects/
# RaisingDetector -- kept local here to avoid cross-test-file coupling)
# ---------------------------------------------------------------------------

class _FakeKeywordResult:
    def __init__(self, status: str, confidence=None):
        self.status = status
        self.confidence = confidence


class FakeDetector:
    def __init__(self, detect_on_call=None, confidence: float = 0.9):
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
    def __init__(self, message: str = "simulated detector failure"):
        self.message = message
        self.calls = 0

    def detect_chunk(self, chunk: np.ndarray):
        self.calls += 1
        raise RuntimeError(self.message)


SR = 1000  # small sample rate keeps test data tiny and durations easy to reason about


def _silence(n: int) -> np.ndarray:
    return np.zeros(n, dtype=np.float32)


def _speech(n: int, level: float = 0.5) -> np.ndarray:
    return np.full(n, level, dtype=np.float32)


def _make_session(**overrides) -> AudioSessionManager:
    defaults = dict(
        sample_rate=SR,
        detector=NeverDetects(),
        preroll_seconds=0.5,
        silence_seconds=0.3,
        min_command_seconds=0.1,
        max_command_seconds=2.0,
        vad_threshold=0.01,
    )
    defaults.update(overrides)
    return AudioSessionManager(**defaults)


# ---------------------------------------------------------------------------
# State transitions
# ---------------------------------------------------------------------------

def test_initial_state_is_idle():
    session = _make_session()
    assert session.state == AudioSessionState.IDLE


def test_start_transitions_idle_to_listening():
    session = _make_session()
    session.start()
    assert session.state == AudioSessionState.LISTENING


def test_start_twice_raises():
    session = _make_session()
    session.start()
    with pytest.raises(AudioSessionError):
        session.start()


def test_process_chunk_before_start_raises():
    session = _make_session()
    with pytest.raises(AudioSessionError):
        session.process_chunk(_silence(10))


def test_full_state_sequence_listening_to_complete():
    detector = FakeDetector(detect_on_call=1)
    session = _make_session(detector=detector, silence_seconds=0.1, min_command_seconds=0.05)
    session.start()
    assert session.state == AudioSessionState.LISTENING

    session.process_chunk(_silence(50))  # call 0: no detection
    assert session.state == AudioSessionState.LISTENING

    session.process_chunk(_silence(50))  # call 1: DETECTED
    assert session.state == AudioSessionState.COMMAND_CAPTURE

    session.process_chunk(_speech(100))
    assert session.state == AudioSessionState.COMMAND_CAPTURE

    # enough trailing silence to end the command
    session.process_chunk(_silence(200))
    assert session.state == AudioSessionState.COMPLETE
    assert session.result is not None


def test_reset_returns_to_listening_from_complete():
    detector = FakeDetector(detect_on_call=0)
    session = _make_session(detector=detector, silence_seconds=0.05, min_command_seconds=0.01)
    session.start()
    session.process_chunk(_silence(20))       # DETECTED immediately
    session.process_chunk(_silence(100))      # enough silence to finalize
    assert session.state == AudioSessionState.COMPLETE

    session.reset()
    assert session.state == AudioSessionState.LISTENING
    assert session.result is None
    assert session.wake_word_detected_ts is None


def test_reset_from_listening_raises():
    session = _make_session()
    session.start()
    with pytest.raises(AudioSessionError):
        session.reset()


def test_process_chunk_after_complete_raises():
    detector = FakeDetector(detect_on_call=0)
    session = _make_session(detector=detector, silence_seconds=0.05, min_command_seconds=0.01)
    session.start()
    session.process_chunk(_silence(20))
    session.process_chunk(_silence(100))
    assert session.state == AudioSessionState.COMPLETE
    with pytest.raises(AudioSessionError):
        session.process_chunk(_silence(10))


# ---------------------------------------------------------------------------
# Pre-roll behavior
# ---------------------------------------------------------------------------

def test_preroll_is_retrieved_into_command_audio_at_detection():
    # capacity_seconds(0.5) * sample_rate(10) = 5-sample pre-roll capacity.
    detector = FakeDetector(detect_on_call=2)
    session = AudioSessionManager(
        sample_rate=10, detector=detector, preroll_seconds=0.5,
        silence_seconds=10.0, min_command_seconds=0.0, max_command_seconds=10.0,
    )
    session.start()
    session.process_chunk(np.array([1.0, 2.0], dtype=np.float32))  # call 0
    session.process_chunk(np.array([3.0, 4.0], dtype=np.float32))  # call 1
    session.process_chunk(np.array([5.0, 6.0], dtype=np.float32))  # call 2: DETECTED

    assert session.state == AudioSessionState.COMMAND_CAPTURE
    # 6 samples written, capacity 5 -> oldest (1.0) dropped.
    np.testing.assert_array_equal(np.concatenate(session._command_chunks), [2.0, 3.0, 4.0, 5.0, 6.0])


def test_no_detection_means_preroll_fills_but_no_command_capture():
    session = _make_session(detector=NeverDetects())
    session.start()
    for _ in range(5):
        session.process_chunk(_silence(50))
    assert session.state == AudioSessionState.LISTENING
    assert session.preroll.filled_samples == 250


def test_post_detection_audio_preserved_without_duplicating_boundary_chunk():
    detector = FakeDetector(detect_on_call=2)
    session = AudioSessionManager(
        sample_rate=10, detector=detector, preroll_seconds=0.5,
        silence_seconds=10.0, min_command_seconds=0.0, max_command_seconds=10.0,
    )
    session.start()
    session.process_chunk(np.array([1.0, 2.0], dtype=np.float32))   # call 0
    session.process_chunk(np.array([3.0, 4.0], dtype=np.float32))   # call 1
    session.process_chunk(np.array([5.0, 6.0], dtype=np.float32))   # call 2: DETECTED -> seeds [2,3,4,5,6]
    session.process_chunk(np.array([7.0, 8.0], dtype=np.float32))   # post-detection, appended directly

    combined = np.concatenate(session._command_chunks)
    np.testing.assert_array_equal(combined, [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    assert list(combined).count(5.0) == 1
    assert list(combined).count(6.0) == 1


# ---------------------------------------------------------------------------
# Command capture after wake event
# ---------------------------------------------------------------------------

def test_command_capture_accumulates_multiple_chunks():
    detector = FakeDetector(detect_on_call=0)
    session = _make_session(detector=detector, silence_seconds=5.0, min_command_seconds=0.0, max_command_seconds=5.0)
    session.start()
    session.process_chunk(_silence(10))  # DETECTED
    assert session.state == AudioSessionState.COMMAND_CAPTURE
    for _ in range(4):
        session.process_chunk(_speech(50))
    assert session.state == AudioSessionState.COMMAND_CAPTURE
    assert session._command_samples == 10 + 4 * 50


# ---------------------------------------------------------------------------
# Silence-based termination
# ---------------------------------------------------------------------------

def test_silence_ends_command_after_speech():
    detector = FakeDetector(detect_on_call=0)
    session = _make_session(
        detector=detector, silence_seconds=0.2, min_command_seconds=0.05, max_command_seconds=5.0,
    )
    session.start()
    session.process_chunk(_silence(10))   # DETECTED, seeds tiny pre-roll
    session.process_chunk(_speech(100))   # speech resets silence counter
    assert session.state == AudioSessionState.COMMAND_CAPTURE

    session.process_chunk(_silence(100))  # 0.1s silence -- not yet enough (need 0.2s)
    assert session.state == AudioSessionState.COMMAND_CAPTURE

    session.process_chunk(_silence(150))  # cumulative silence now >= 0.2s
    assert session.state == AudioSessionState.COMPLETE
    assert session.result.end_reason == "silence"


def test_speech_resets_silence_counter():
    detector = FakeDetector(detect_on_call=0)
    session = _make_session(
        detector=detector, silence_seconds=0.2, min_command_seconds=0.0, max_command_seconds=5.0,
    )
    session.start()
    session.process_chunk(_silence(10))    # DETECTED
    session.process_chunk(_silence(150))   # 0.15s silence (not yet enough)
    session.process_chunk(_speech(50))     # speech interrupts -- silence counter resets
    session.process_chunk(_silence(150))   # another 0.15s -- still not enough on its own
    assert session.state == AudioSessionState.COMMAND_CAPTURE
    session.process_chunk(_silence(60))    # now cumulative since last speech >= 0.2s
    assert session.state == AudioSessionState.COMPLETE


# ---------------------------------------------------------------------------
# Minimum / maximum command duration
# ---------------------------------------------------------------------------

def test_min_command_duration_prevents_instant_termination():
    """Even with immediate silence, the command must not end before
    min_command_seconds worth of audio has been captured."""
    detector = FakeDetector(detect_on_call=0)
    session = _make_session(
        detector=detector, silence_seconds=0.05, min_command_seconds=0.5, max_command_seconds=5.0,
    )
    session.start()
    session.process_chunk(_silence(10))   # DETECTED
    # Feed plenty of silence -- silence_seconds (0.05s) is satisfied almost
    # immediately, but min_command_seconds (0.5s = 500 samples) is not.
    session.process_chunk(_silence(100))
    assert session.state == AudioSessionState.COMMAND_CAPTURE
    session.process_chunk(_silence(100))
    assert session.state == AudioSessionState.COMMAND_CAPTURE
    # Now push past min_command_seconds -- should finalize on this call.
    session.process_chunk(_silence(400))
    assert session.state == AudioSessionState.COMPLETE
    assert session.result.duration_seconds >= 0.5


def test_max_command_duration_forces_termination_even_with_continuous_speech():
    detector = FakeDetector(detect_on_call=0)
    session = _make_session(
        detector=detector, silence_seconds=100.0, min_command_seconds=0.0, max_command_seconds=0.3,
    )
    session.start()
    session.process_chunk(_silence(10))  # DETECTED
    for _ in range(10):
        if session.state == AudioSessionState.COMPLETE:
            break
        session.process_chunk(_speech(50))  # never silent -- only max_duration can end this
    assert session.state == AudioSessionState.COMPLETE
    assert session.result.end_reason == "max_duration"


def test_construction_rejects_min_greater_than_max():
    with pytest.raises(ValueError):
        AudioSessionManager(min_command_seconds=5.0, max_command_seconds=1.0)


def test_construction_rejects_non_positive_sample_rate():
    with pytest.raises(ValueError):
        AudioSessionManager(sample_rate=0)


def test_construction_rejects_non_positive_max_duration():
    with pytest.raises(ValueError):
        AudioSessionManager(max_command_seconds=0)


# ---------------------------------------------------------------------------
# Empty / invalid audio chunks
# ---------------------------------------------------------------------------

def test_empty_chunk_while_listening_is_a_harmless_noop():
    session = _make_session()
    session.start()
    state = session.process_chunk(np.zeros(0, dtype=np.float32))
    assert state == AudioSessionState.LISTENING
    assert session.preroll.filled_samples == 0


def test_empty_chunk_while_command_capture_is_a_harmless_noop():
    detector = FakeDetector(detect_on_call=0)
    session = _make_session(detector=detector, silence_seconds=5.0, min_command_seconds=0.0)
    session.start()
    session.process_chunk(_silence(10))  # DETECTED
    samples_before = session._command_samples
    state = session.process_chunk(np.zeros(0, dtype=np.float32))
    assert state == AudioSessionState.COMMAND_CAPTURE
    assert session._command_samples == samples_before


def test_invalid_chunk_transitions_to_error_not_raise():
    session = _make_session()
    session.start()

    class _Unconvertible:
        def __array__(self):
            raise TypeError("cannot convert")

    state = session.process_chunk(_Unconvertible())
    assert state == AudioSessionState.ERROR
    assert session.error_message is not None


def test_error_state_rejects_further_chunks():
    session = _make_session()
    session.start()

    class _Unconvertible:
        def __array__(self):
            raise TypeError("cannot convert")

    session.process_chunk(_Unconvertible())
    assert session.state == AudioSessionState.ERROR
    with pytest.raises(AudioSessionError):
        session.process_chunk(_silence(10))


def test_reset_from_error_returns_to_listening():
    session = _make_session()
    session.start()

    class _Unconvertible:
        def __array__(self):
            raise TypeError("cannot convert")

    session.process_chunk(_Unconvertible())
    assert session.state == AudioSessionState.ERROR
    session.reset()
    assert session.state == AudioSessionState.LISTENING
    assert session.error_message is None


# ---------------------------------------------------------------------------
# Detector errors handled safely (not a session-level ERROR)
# ---------------------------------------------------------------------------

def test_detector_exception_does_not_crash_or_error_the_session():
    detector = RaisingDetector("boom")
    session = _make_session(detector=detector)
    session.start()
    state = session.process_chunk(_silence(50))
    assert state == AudioSessionState.LISTENING  # not ERROR -- a detector hiccup, not invalid audio
    assert session.detector_error is not None
    assert "boom" in session.detector_error


# ---------------------------------------------------------------------------
# Resulting segment shape/sample-rate/sample-count
# ---------------------------------------------------------------------------

def test_result_segment_has_expected_sample_rate():
    detector = FakeDetector(detect_on_call=0)
    session = _make_session(detector=detector, silence_seconds=0.05, min_command_seconds=0.01, sample_rate=8000)
    session.start()
    session.process_chunk(_silence(10))
    session.process_chunk(_silence(1000))
    assert session.state == AudioSessionState.COMPLETE
    assert session.result.sample_rate == 8000


def test_result_segment_sample_count_matches_captured_audio():
    detector = FakeDetector(detect_on_call=0)
    session = _make_session(detector=detector, silence_seconds=0.05, min_command_seconds=0.0)
    session.start()
    session.process_chunk(_silence(10))   # seeds pre-roll (10 samples) into command audio
    session.process_chunk(_speech(40))
    session.process_chunk(_silence(60))   # enough silence (0.06s >= 0.05s) to finalize
    assert session.state == AudioSessionState.COMPLETE

    expected_samples = 10 + 40 + 60
    assert session.result.waveform.size == expected_samples
    assert session.result.duration_seconds == pytest.approx(expected_samples / SR)


def test_result_is_a_command_audio_segment_with_all_fields():
    detector = FakeDetector(detect_on_call=0, confidence=0.77)
    session = _make_session(detector=detector, silence_seconds=0.05, min_command_seconds=0.0)
    session.start()
    session.process_chunk(_silence(10))
    session.process_chunk(_silence(100))
    assert isinstance(session.result, CommandAudioSegment)
    assert session.wake_word_confidence == 0.77
    assert session.result.wake_word_detected_ts == session.wake_word_detected_ts
    assert session.result.command_end_ts is not None
    assert session.result.waveform.dtype == np.float32


# ---------------------------------------------------------------------------
# Modularity: the real (trained) KeywordDetector plugs in with zero
# special-casing (no NOVA detection accuracy claim of any kind)
# ---------------------------------------------------------------------------

def test_real_keyword_detector_integrates_safely_without_special_casing():
    from keyword_spotting.detector import KeywordDetector

    detector = KeywordDetector()  # loads the project's real trained checkpoint
    assert detector.is_ready is True  # confirms it actually loaded, not skipped

    session = _make_session(detector=detector, sample_rate=detector.sample_rate)
    session.start()
    for _ in range(5):
        state = session.process_chunk(np.random.randn(320).astype(np.float32) * 0.01)
    # Low-amplitude random noise is not the NOVA wake word -- the real
    # detector must not fabricate a detection on it, so the session stays
    # in LISTENING -- never crash, never a spurious DETECTED.
    assert state == AudioSessionState.LISTENING
    assert session.detector_error is None
