"""
voice_pipeline.orchestrator
==============================
VoiceCommandOrchestrator - wires the existing laptop-demo components
into one end-to-end pipeline:

    MicCapture chunks -> wake detection -> AudioSessionManager
      -> ASR (asr.transcriber) -> CommandInterpreter -> structured result

This module does not reimplement any of those components -- it only
sequences calls to their existing, UNMODIFIED public interfaces:
  - audio_processing.session.AudioSessionManager (chunk buffering,
    pre-roll, wake-gated command capture, silence/duration-based
    end-of-command detection)
  - asr.transcriber.get_transcriber() (cached faster-whisper backend)
  - command.interpreter.CommandInterpreter (deterministic rule-based
    parser)

Two wake-trigger modes (requirement: "the rest of the pipeline must be
identical after the wake event")
--------------------------------------------------------------------------
Both modes are just different objects passed as AudioSessionManager's
existing `detector=` parameter -- nothing downstream of the wake event
knows or cares which one fired.

- WakeMode.REAL_KWS: `keyword_spotting.detector.get_keyword_detector()`,
  the real, TinyKWSNet-backed detector. With no trained
  `models/kws_nova_cnn.pt` checkpoint (the current, honest state of this
  project), it reports `is_ready=False` and never fires `DETECTED` --
  this orchestrator makes NO claim that it does. Once a real checkpoint
  exists, this mode starts working with ZERO changes here.
- WakeMode.MOCK: `MockWakeTrigger` (this module) -- a deterministic,
  manually-armed stand-in that fires on the next chunk after `.fire()`
  is called, completely independent of audio content. THIS IS NOT NOVA
  DETECTION and is never described as such anywhere in this module's
  output or logs -- every message mentioning it says "mock"/"MOCK"
  explicitly.

No microphone I/O of its own
-------------------------------
`VoiceCommandOrchestrator.run_cycle()` consumes chunks from any
iterable of numpy arrays -- a live `MicCapture` read loop (see
scripts/voice_command_demo.py for how that's wired) or a plain list/
generator of synthetic chunks (see tests/unit/test_voice_pipeline.py).
This keeps the orchestrator itself free of blocking I/O and fully
testable without a real microphone.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Optional

import numpy as np

from audio_processing.session import (
    AudioSessionManager,
    AudioSessionState,
    CommandAudioSegment,
    DEFAULT_MAX_COMMAND_SECONDS,
    DEFAULT_MIN_COMMAND_SECONDS,
    DEFAULT_PREROLL_SECONDS,
    DEFAULT_SILENCE_SECONDS,
    DEFAULT_VAD_THRESHOLD,
)
from asr.transcriber import DEFAULT_COMPUTE_TYPE, DEFAULT_DEVICE, DEFAULT_MODEL_NAME, get_transcriber
from command.interpreter import CommandInterpreter, CommandResult

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_RATE = 16000


class WakeMode(str, Enum):
    REAL_KWS = "REAL_KWS"
    MOCK = "MOCK"


# ---------------------------------------------------------------------------
# Mock wake trigger -- development/testing stand-in ONLY, never NOVA detection
# ---------------------------------------------------------------------------

@dataclass
class _MockDetectionResult:
    """Deliberately does NOT reuse keyword_spotting.detector.KeywordResult
    (different module, different meaning) -- duck-typed to the same
    `.status`/`.confidence` shape every KWS-consumer in this project
    already expects (see audio_processing.session.AudioSessionManager's
    `detector` contract)."""
    status: str = "DETECTED"
    keyword: str = "MOCK_TRIGGER"   # deliberately NOT "NOVA"
    confidence: Optional[float] = None


class MockWakeTrigger:
    """
    Deterministic, manually-armed wake-trigger stand-in for development
    and testing while real NOVA recording/training remains postponed.

    THIS IS NOT NOVA DETECTION. It never inspects audio content in any
    way -- `detect_chunk()` returns a DETECTED result on the very next
    call after `.fire()` is invoked, regardless of what the chunk
    actually contains, and None otherwise. It exists purely so the rest
    of this pipeline (AudioSessionManager onward) can be exercised
    identically to how it will run with a real detector, per this
    task's explicit requirement that "the rest of the pipeline must be
    identical after the wake event."

    Duck-types the same `detect_chunk(chunk) -> Optional[result]`
    contract as `keyword_spotting.detector.KeywordDetector` and every
    fake detector used in this project's tests -- swapping this for the
    real detector (or vice versa) requires no other code changes.
    """

    def __init__(self) -> None:
        self._armed = False
        self.fire_count = 0

    def fire(self) -> None:
        """Arm the trigger: the *next* detect_chunk() call reports DETECTED."""
        self._armed = True

    @property
    def is_armed(self) -> bool:
        return self._armed

    def detect_chunk(self, chunk: np.ndarray) -> Optional[_MockDetectionResult]:
        if self._armed:
            self._armed = False
            self.fire_count += 1
            return _MockDetectionResult()
        return None


# ---------------------------------------------------------------------------
# Latency instrumentation
# ---------------------------------------------------------------------------

@dataclass
class PipelineTimings:
    """Wall-clock timestamps (time.time()) for each pipeline stage.

    All *_ms properties return None if the timestamps needed to compute
    them aren't both set yet -- never a fabricated/zero duration.
    """
    wake_detected_ts: Optional[float] = None
    command_capture_start_ts: Optional[float] = None
    command_capture_end_ts: Optional[float] = None
    asr_start_ts: Optional[float] = None
    asr_end_ts: Optional[float] = None
    command_interpreted_ts: Optional[float] = None

    @property
    def command_capture_duration_ms(self) -> Optional[float]:
        if self.command_capture_start_ts is None or self.command_capture_end_ts is None:
            return None
        return (self.command_capture_end_ts - self.command_capture_start_ts) * 1000.0

    @property
    def asr_latency_ms(self) -> Optional[float]:
        if self.asr_start_ts is None or self.asr_end_ts is None:
            return None
        return (self.asr_end_ts - self.asr_start_ts) * 1000.0

    @property
    def total_wake_to_result_ms(self) -> Optional[float]:
        if self.wake_detected_ts is None or self.command_interpreted_ts is None:
            return None
        return (self.command_interpreted_ts - self.wake_detected_ts) * 1000.0


@dataclass
class PipelineResult:
    """
    status : "OK" | "ASR_ERROR" | "EMPTY_AUDIO" | "SESSION_ERROR" | "INCOMPLETE"
        "OK" covers a successfully interpreted command REGARDLESS of
        whether CommandInterpreter recognized it -- action="unknown" is
        a valid, non-error outcome (matches command.interpreter's own
        philosophy). The other statuses mean the pipeline could not
        reach interpretation at all.
    """
    status: str
    transcript: str = ""
    command: Optional[CommandResult] = None
    end_reason: Optional[str] = None   # from CommandAudioSegment: "silence" | "max_duration"
    timings: PipelineTimings = field(default_factory=PipelineTimings)
    message: str = ""


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class VoiceCommandOrchestrator:
    """
    Parameters
    ----------
    wake_mode : WakeMode
        REAL_KWS or MOCK -- informational/logging only; the actual
        behaviour comes entirely from `detector`.
    detector : optional duck-typed KWS detector/trigger.
        If None, one is created automatically based on `wake_mode`:
        `keyword_spotting.detector.get_keyword_detector()` for
        REAL_KWS, or a fresh `MockWakeTrigger()` for MOCK.
    sample_rate : int
        Sample rate of chunks that will be fed to `run_cycle()`.
    session_kwargs : dict, optional
        Extra keyword arguments forwarded to AudioSessionManager
        (preroll_seconds, silence_seconds, min_command_seconds,
        max_command_seconds, vad_threshold) -- see
        docs/AUDIO_SESSION_MANAGER.md for their meaning.
    asr_model_name, asr_device, asr_compute_type : str
        Forwarded to asr.transcriber.get_transcriber() -- the SAME
        process-wide cache every other ASR call site uses, so the
        model is loaded at most once regardless of how many
        orchestrators/cycles use it.
    log : callable(str) -> None, optional
        Called with each console-demo status line (default: print).
        Tests pass a list-appending callable to capture output instead.
    """

    def __init__(
        self,
        wake_mode: WakeMode = WakeMode.MOCK,
        detector: Optional[Any] = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        session_kwargs: Optional[dict] = None,
        asr_model_name: str = DEFAULT_MODEL_NAME,
        asr_device: str = DEFAULT_DEVICE,
        asr_compute_type: str = DEFAULT_COMPUTE_TYPE,
        log: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.wake_mode = wake_mode
        self.sample_rate = sample_rate
        self._log_fn = log or print

        if detector is None:
            if wake_mode == WakeMode.REAL_KWS:
                from keyword_spotting.detector import get_keyword_detector
                detector = get_keyword_detector()
            else:
                detector = MockWakeTrigger()
        self.detector = detector

        session_kwargs = dict(session_kwargs or {})
        session_kwargs.setdefault("preroll_seconds", DEFAULT_PREROLL_SECONDS)
        session_kwargs.setdefault("silence_seconds", DEFAULT_SILENCE_SECONDS)
        session_kwargs.setdefault("min_command_seconds", DEFAULT_MIN_COMMAND_SECONDS)
        session_kwargs.setdefault("max_command_seconds", DEFAULT_MAX_COMMAND_SECONDS)
        session_kwargs.setdefault("vad_threshold", DEFAULT_VAD_THRESHOLD)
        self.session = AudioSessionManager(
            sample_rate=sample_rate, detector=self.detector, **session_kwargs
        )

        # Cached, process-wide -- NOT reloaded per command or per orchestrator.
        self.transcriber = get_transcriber(
            model_name=asr_model_name, device=asr_device, compute_type=asr_compute_type
        )
        self.interpreter = CommandInterpreter()

        self.timings = PipelineTimings()

    def _log(self, message: str) -> None:
        self._log_fn(message)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the underlying AudioSessionManager (IDLE -> LISTENING)."""
        self.session.start()
        self.timings = PipelineTimings()
        mode_note = " (MOCK wake trigger -- NOT real NOVA detection)" if self.wake_mode == WakeMode.MOCK else ""
        self._log(f"Listening...{mode_note}")

    def reset(self) -> None:
        """Reset for another wake -> command cycle, reusing the same
        detector, ASR model, and interpreter (no reload)."""
        self.session.reset()
        self.timings = PipelineTimings()

    # ------------------------------------------------------------------
    # Capture phase: chunks -> one CommandAudioSegment (or an early result)
    # ------------------------------------------------------------------

    def _run_capture_phase(
        self, chunks: Iterable[np.ndarray]
    ) -> "CommandAudioSegment | PipelineResult":
        """
        Consume chunks until AudioSessionManager reaches COMPLETE or
        ERROR, or `chunks` is exhausted first. Returns the finished
        CommandAudioSegment on success, or an early PipelineResult
        describing why capture didn't complete.
        """
        wake_announced = False
        for chunk in chunks:
            state = self.session.process_chunk(chunk)

            if not wake_announced and state == AudioSessionState.COMMAND_CAPTURE:
                wake_announced = True
                self.timings.wake_detected_ts = self.session.wake_word_detected_ts
                self.timings.command_capture_start_ts = time.time()
                mock_note = " (MOCK trigger)" if self.wake_mode == WakeMode.MOCK else ""
                self._log(f"Wake trigger detected{mock_note}")
                self._log("Capturing command...")

            if state == AudioSessionState.COMPLETE:
                self.timings.command_capture_end_ts = time.time()
                return self.session.result  # type: ignore[return-value]

            if state == AudioSessionState.ERROR:
                return PipelineResult(
                    status="SESSION_ERROR",
                    message=self.session.error_message or "Audio session entered an error state.",
                    timings=self.timings,
                )

        return PipelineResult(
            status="INCOMPLETE",
            message=(
                "Audio source was exhausted before a command was captured "
                f"(session state: {self.session.state.value}). If using "
                "WakeMode.REAL_KWS, this is expected while no trained NOVA "
                "checkpoint exists -- see docs/AUDIO_SESSION_MANAGER.md."
            ),
            timings=self.timings,
        )

    # ------------------------------------------------------------------
    # ASR + interpretation phase: one segment -> one PipelineResult
    # ------------------------------------------------------------------

    def _run_asr_and_interpret(self, segment: CommandAudioSegment) -> PipelineResult:
        if segment.waveform.size == 0:
            return PipelineResult(
                status="EMPTY_AUDIO",
                end_reason=segment.end_reason,
                message="Captured command audio was empty; nothing to transcribe.",
                timings=self.timings,
            )

        self._log("Transcribing...")
        self.timings.asr_start_ts = time.time()
        asr_result = self.transcriber.transcribe(segment.waveform, sample_rate=segment.sample_rate)
        self.timings.asr_end_ts = time.time()

        if asr_result.status != "OK":
            return PipelineResult(
                status="ASR_ERROR",
                end_reason=segment.end_reason,
                message=f"ASR failed ({asr_result.status}): {asr_result.message}",
                timings=self.timings,
            )

        self._log(f'Command: "{asr_result.text}"')
        cmd_result = self.interpreter.interpret(asr_result.text)
        self.timings.command_interpreted_ts = time.time()

        self._log(f"Intent: {cmd_result.intent}")
        if cmd_result.device:
            self._log(f"Device: {cmd_result.device}")
        self._log(f"Action: {cmd_result.action}")

        return PipelineResult(
            status="OK",
            transcript=asr_result.text,
            command=cmd_result,
            end_reason=segment.end_reason,
            timings=self.timings,
        )

    # ------------------------------------------------------------------
    # Full cycle
    # ------------------------------------------------------------------

    def run_cycle(self, chunks: Iterable[np.ndarray]) -> PipelineResult:
        """
        Run one full LISTENING -> COMMAND_CAPTURE -> ASR -> interpret
        cycle, consuming chunks from `chunks` (a live MicCapture read
        loop, wrapped by the caller into an iterable -- see
        scripts/voice_command_demo.py -- or synthetic test chunks).

        Call `start()` before this, and `reset()` before calling this
        again for another cycle.
        """
        outcome = self._run_capture_phase(chunks)
        if isinstance(outcome, PipelineResult):
            return outcome
        return self._run_asr_and_interpret(outcome)
