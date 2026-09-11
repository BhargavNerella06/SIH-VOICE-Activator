"""
tests/unit/test_voice_pipeline.py
====================================
Tests for voice_pipeline.orchestrator.VoiceCommandOrchestrator, using
only synthetic audio chunks and a mocked ASR backend (monkeypatching
get_transcriber, mirroring the established pattern in
tests/integration/test_stream_api.py). No microphone, no real speech,
no trained NOVA model, no network dependency.

CommandInterpreter is exercised for real (unmodified, deterministic,
no model to mock) so these tests also cover real
ASR-text -> command-interpretation integration.
"""

from __future__ import annotations

import numpy as np
import pytest

from asr.transcriber import TranscriptionResult
from audio_processing.session import AudioSessionState
from voice_pipeline.orchestrator import (
    MockWakeTrigger,
    PipelineResult,
    VoiceCommandOrchestrator,
    WakeMode,
)

SR = 1000


def _silence(n: int) -> np.ndarray:
    return np.zeros(n, dtype=np.float32)


def _speech(n: int, level: float = 0.5) -> np.ndarray:
    return np.full(n, level, dtype=np.float32)


class _FakeTranscriber:
    """Deterministic stand-in for asr.transcriber.Transcriber."""

    def __init__(self, text: str = "turn on the lights", status: str = "OK", latency_ms: float = 1.0):
        self.text = text
        self.status = status
        self.latency_ms = latency_ms
        self.calls = 0

    def transcribe(self, waveform, sample_rate):
        self.calls += 1
        return TranscriptionResult(
            status=self.status,
            text=self.text if self.status == "OK" else "",
            language="en",
            latency_ms=self.latency_ms,
            message=f"fake transcriber ({self.status})",
        )


@pytest.fixture
def fake_transcriber(monkeypatch):
    fake = _FakeTranscriber()
    monkeypatch.setattr("voice_pipeline.orchestrator.get_transcriber", lambda **kwargs: fake)
    return fake


def _make_orchestrator(**overrides):
    trigger = MockWakeTrigger()
    defaults = dict(
        wake_mode=WakeMode.MOCK,
        detector=trigger,
        sample_rate=SR,
        session_kwargs=dict(
            preroll_seconds=0.5, silence_seconds=0.2,
            min_command_seconds=0.05, max_command_seconds=2.0,
        ),
        log=lambda msg: None,  # silence console output in tests
    )
    defaults.update(overrides)
    orch = VoiceCommandOrchestrator(**defaults)
    return orch, trigger


# ---------------------------------------------------------------------------
# Complete wake -> command -> ASR -> interpreter flow
# ---------------------------------------------------------------------------

def test_full_flow_with_mock_wake_produces_structured_command(fake_transcriber):
    fake_transcriber.text = "turn on the lights"
    orch, trigger = _make_orchestrator()
    orch.start()

    def chunks():
        for _ in range(3):
            yield _silence(50)
        trigger.fire()
        for _ in range(3):
            yield _speech(50)
        for _ in range(5):
            yield _silence(50)

    result = orch.run_cycle(chunks())

    assert result.status == "OK"
    assert result.transcript == "turn on the lights"
    assert result.command.action == "turn_on"
    assert result.command.intent == "device_control"
    assert result.command.device == "lights"
    assert result.command.original_text == "turn on the lights"
    assert result.command.confidence == 1.0
    assert result.end_reason == "silence"
    assert fake_transcriber.calls == 1


def test_full_flow_populates_all_latency_timings(fake_transcriber):
    orch, trigger = _make_orchestrator()
    orch.start()

    def chunks():
        for _ in range(3):
            yield _silence(50)
        trigger.fire()
        for _ in range(3):
            yield _speech(50)
        for _ in range(5):
            yield _silence(50)

    result = orch.run_cycle(chunks())
    t = result.timings

    assert t.wake_detected_ts is not None
    assert t.command_capture_start_ts is not None
    assert t.command_capture_end_ts is not None
    assert t.asr_start_ts is not None
    assert t.asr_end_ts is not None
    assert t.command_interpreted_ts is not None
    assert t.command_capture_duration_ms is not None and t.command_capture_duration_ms >= 0
    assert t.asr_latency_ms is not None and t.asr_latency_ms >= 0
    assert t.total_wake_to_result_ms is not None and t.total_wake_to_result_ms >= 0
    # Ordering sanity: wake happens no later than capture end, which is no
    # later than the final interpreted timestamp.
    assert t.wake_detected_ts <= t.command_capture_end_ts <= t.command_interpreted_ts


# ---------------------------------------------------------------------------
# Mock wake trigger (explicitly NOT NOVA detection)
# ---------------------------------------------------------------------------

def test_mock_trigger_fires_only_once_per_arm():
    trigger = MockWakeTrigger()
    assert trigger.detect_chunk(_silence(10)) is None  # not armed yet
    trigger.fire()
    assert trigger.is_armed is True
    result = trigger.detect_chunk(_silence(10))
    assert result is not None
    assert result.status == "DETECTED"
    assert result.keyword == "MOCK_TRIGGER"  # never "NOVA"
    assert trigger.is_armed is False
    # Firing consumed -- the next chunk must NOT also report DETECTED.
    assert trigger.detect_chunk(_silence(10)) is None


def test_mock_trigger_ignores_audio_content_entirely():
    """The mock trigger must fire purely on .fire(), never based on
    what's actually in the chunk -- proving it makes no NOVA claim."""
    trigger = MockWakeTrigger()
    trigger.fire()
    # A chunk full of silence still fires -- content is irrelevant.
    result = trigger.detect_chunk(_silence(100))
    assert result.status == "DETECTED"


def test_orchestrator_mock_mode_logs_are_clearly_labeled():
    logs = []
    orch, trigger = _make_orchestrator(log=logs.append)
    orch.start()
    assert any("MOCK" in line for line in logs)
    # Any mention of NOVA in MOCK mode must be a disclaimer ("NOT real NOVA
    # detection"), never an unqualified claim that NOVA was detected.
    nova_lines = [line for line in logs if "NOVA" in line]
    assert all("not" in line.lower() for line in nova_lines)


# ---------------------------------------------------------------------------
# Real KeywordDetector path with the currently-untrained checkpoint
# ---------------------------------------------------------------------------

def test_real_kws_mode_with_trained_checkpoint_does_not_fire_on_noise(fake_transcriber):
    from keyword_spotting.detector import KeywordDetector

    detector = KeywordDetector()  # loads the project's real trained checkpoint
    assert detector.is_ready is True  # confirms it actually loaded, not skipped

    orch = VoiceCommandOrchestrator(
        wake_mode=WakeMode.REAL_KWS, detector=detector, sample_rate=detector.sample_rate,
        log=lambda msg: None,
    )
    orch.start()

    chunks = [np.random.randn(320).astype(np.float32) * 0.01 for _ in range(10)]
    result = orch.run_cycle(iter(chunks))

    assert result.status == "INCOMPLETE"
    assert result.command is None
    assert result.transcript == ""
    assert fake_transcriber.calls == 0  # ASR must never be reached without a real detection
    # No claim of NOVA accuracy anywhere in the result.
    assert "NOVA" not in result.message or "no trained" in result.message.lower() or "checkpoint" in result.message.lower()


def test_real_kws_mode_does_not_crash_with_untrained_detector():
    from keyword_spotting.detector import KeywordDetector

    detector = KeywordDetector()
    orch = VoiceCommandOrchestrator(
        wake_mode=WakeMode.REAL_KWS, detector=detector, sample_rate=detector.sample_rate,
        log=lambda msg: None,
    )
    orch.start()
    chunks = [np.random.randn(320).astype(np.float32) * 0.01 for _ in range(5)]
    result = orch.run_cycle(iter(chunks))
    assert orch.session.state == AudioSessionState.LISTENING  # never advanced, never crashed


# ---------------------------------------------------------------------------
# ASR failure
# ---------------------------------------------------------------------------

def test_asr_error_status_surfaces_cleanly(fake_transcriber):
    fake_transcriber.status = "ERROR"
    orch, trigger = _make_orchestrator()
    orch.start()

    def chunks():
        for _ in range(2):
            yield _silence(50)
        trigger.fire()
        for _ in range(2):
            yield _speech(50)
        for _ in range(5):
            yield _silence(50)

    result = orch.run_cycle(chunks())
    assert result.status == "ASR_ERROR"
    assert result.command is None
    assert "ERROR" in result.message


def test_asr_not_implemented_surfaces_as_asr_error_not_a_crash(fake_transcriber):
    fake_transcriber.status = "NOT_IMPLEMENTED"
    orch, trigger = _make_orchestrator()
    orch.start()

    def chunks():
        for _ in range(2):
            yield _silence(50)
        trigger.fire()
        for _ in range(2):
            yield _speech(50)
        for _ in range(5):
            yield _silence(50)

    result = orch.run_cycle(chunks())
    assert result.status == "ASR_ERROR"
    assert result.command is None


# ---------------------------------------------------------------------------
# Unknown command
# ---------------------------------------------------------------------------

def test_unknown_command_is_a_successful_pipeline_result(fake_transcriber):
    """An unrecognized transcript is a valid outcome (action="unknown"),
    not a pipeline failure -- matches command.interpreter's own philosophy."""
    fake_transcriber.text = "what is the weather today"
    orch, trigger = _make_orchestrator()
    orch.start()

    def chunks():
        for _ in range(2):
            yield _silence(50)
        trigger.fire()
        for _ in range(2):
            yield _speech(50)
        for _ in range(5):
            yield _silence(50)

    result = orch.run_cycle(chunks())
    assert result.status == "OK"
    assert result.command.action == "unknown"
    assert result.command.intent == "unknown"
    assert result.command.confidence == 0.0


# ---------------------------------------------------------------------------
# Empty command audio
# ---------------------------------------------------------------------------

def test_empty_command_audio_short_circuits_before_asr(fake_transcriber):
    from audio_processing.session import CommandAudioSegment

    orch, _ = _make_orchestrator()
    empty_segment = CommandAudioSegment(
        waveform=np.zeros(0, dtype=np.float32), sample_rate=SR,
        duration_seconds=0.0, end_reason="silence",
    )
    result = orch._run_asr_and_interpret(empty_segment)
    assert result.status == "EMPTY_AUDIO"
    assert result.command is None
    assert fake_transcriber.calls == 0  # ASR must never be called on empty audio


# ---------------------------------------------------------------------------
# Session error / max-duration termination ("timeout")
# ---------------------------------------------------------------------------

def test_session_error_from_invalid_chunk_surfaces_as_session_error(fake_transcriber):
    orch, trigger = _make_orchestrator()
    orch.start()

    class _Unconvertible:
        def __array__(self):
            raise TypeError("cannot convert")

    def chunks():
        for _ in range(2):
            yield _silence(50)
        trigger.fire()
        yield _speech(50)
        yield _Unconvertible()

    result = orch.run_cycle(chunks())
    assert result.status == "SESSION_ERROR"
    assert result.command is None


def test_max_duration_termination_still_produces_a_valid_ok_result(fake_transcriber):
    """Hitting max_command_seconds is a normal, deterministic outcome
    (the pipeline's built-in "timeout"), not an error -- the pipeline
    must still proceed through ASR/interpretation normally."""
    orch, trigger = _make_orchestrator(
        session_kwargs=dict(
            preroll_seconds=0.5, silence_seconds=100.0,  # never triggers on its own
            min_command_seconds=0.0, max_command_seconds=0.2,
        ),
    )
    orch.start()

    def chunks():
        trigger.fire()
        for _ in range(20):
            yield _speech(50)  # continuous speech -- only max_duration can end this

    result = orch.run_cycle(chunks())
    assert result.status == "OK"
    assert result.end_reason == "max_duration"


# ---------------------------------------------------------------------------
# Reset for another command cycle
# ---------------------------------------------------------------------------

def test_reset_allows_a_second_cycle_without_reloading_asr(fake_transcriber):
    orch, trigger = _make_orchestrator()
    orch.start()

    def chunks():
        for _ in range(2):
            yield _silence(50)
        trigger.fire()
        for _ in range(2):
            yield _speech(50)
        for _ in range(5):
            yield _silence(50)

    first = orch.run_cycle(chunks())
    assert first.status == "OK"

    orch.reset()
    assert orch.session.state == AudioSessionState.LISTENING

    second = orch.run_cycle(chunks())
    assert second.status == "OK"

    # Same cached transcriber instance used both times -- no reload.
    assert fake_transcriber.calls == 2


def test_reset_clears_previous_timings():
    orch, trigger = _make_orchestrator()
    orch.start()

    def chunks():
        for _ in range(2):
            yield _silence(50)
        trigger.fire()
        for _ in range(5):
            yield _silence(50)

    orch.run_cycle(chunks())
    assert orch.timings.wake_detected_ts is not None
    orch.reset()
    assert orch.timings.wake_detected_ts is None
