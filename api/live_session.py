"""
api.live_session
====================
Drives LIVE mode: the real mic -> KeywordDetector -> AudioSessionManager
-> ASR -> CommandInterpreter -> device-execution pipeline, running in a
background thread, pushing JSON-serializable events onto a thread-safe
queue for api/routers/live.py to broadcast to connected browser clients.

This module does NOT reimplement any pipeline stage. It only drives the
existing, UNMODIFIED voice_pipeline.orchestrator.VoiceCommandOrchestrator
(WakeMode.REAL_KWS) with a live audio_processing.capture.MicCapture
source, and maps its PipelineResult onto api.device_state's software
device-state layer -- see that module's docstring for why this stands in
for IoT relay execution instead of driving real hardware.

Honesty / graceful degradation
--------------------------------
If the microphone or the trained KWS checkpoint is unavailable, this
degrades to an OFFLINE status (broadcast once) rather than crashing the
server. DEMO mode (entirely client-side in frontend/code.html) does not
depend on this module at all and keeps working regardless.

Threading model
----------------
Mic I/O and orchestrator.run_cycle() (which blocks on ASR inference) are
synchronous/blocking, so they run on a dedicated background thread here,
never on the FastAPI event loop. Events cross to the async world only via
a plain `queue.Queue`, which api/routers/live.py drains from an asyncio
task -- no asyncio API is ever called from this thread.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
from typing import Iterator, Optional

import numpy as np

from api.device_state import execute_device
from audio_processing.session import AudioSessionState
from voice_pipeline.orchestrator import PipelineResult, VoiceCommandOrchestrator, WakeMode

logger = logging.getLogger(__name__)

# Explicit, opt-in override for KeywordDetector's confidence threshold, set
# for this process only via env var -- mirrors scripts/voice_command_demo.py's
# --kws-threshold flag (there's no argparse at FastAPI-server-startup time,
# so an env var is this path's equivalent "explicit flag"). Unset means the
# production default in keyword_spotting/detector.py (0.85) is used
# unchanged; this module never changes that default itself. Measured against
# this project's own 10+10 verified real-recording dataset (see
# keyword_spotting/training/ and scripts/debug_command_capture.py), 0.40 is
# the demo-only threshold that was found to actually fire on live speech.
_KWS_THRESHOLD_ENV_VAR = "NOVA_KWS_THRESHOLD"

_STATUS_FOR_SESSION_STATE = {
    AudioSessionState.LISTENING: ("READY", 'Say "NOVA" to wake'),
    AudioSessionState.COMMAND_CAPTURE: ("DETECTED", "NOVA detected — capturing command..."),
    AudioSessionState.PROCESSING: ("PROCESSING", "Processing captured audio..."),
}


class LiveVoiceSession:
    """Owns the background thread. Call `start()` once (e.g. at FastAPI
    startup) and `stop()` at shutdown. `events` is a thread-safe queue of
    dicts ready to be broadcast verbatim as WebSocket JSON messages."""

    def __init__(self) -> None:
        self.events: "queue.Queue[dict]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()
        self.status = "STOPPED"  # STOPPED | STARTING | READY | OFFLINE
        self.offline_reason: Optional[str] = None

    def _emit(self, event: dict) -> None:
        self.events.put(event)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_flag.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="nova-live-session")
        self._thread.start()

    def stop(self) -> None:
        self._stop_flag.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    # ------------------------------------------------------------------

    def _go_offline(self, reason: str) -> None:
        self.status = "OFFLINE"
        self.offline_reason = reason
        logger.warning("LiveVoiceSession OFFLINE: %s", reason)
        self._emit({"type": "status", "state": "OFFLINE", "message": reason})

    def _run(self) -> None:
        self.status = "STARTING"
        self._emit({"type": "status", "state": "STARTING", "message": "Starting NOVA live pipeline..."})

        try:
            from audio_processing.capture import MicCapture, MicCaptureError
        except Exception as exc:
            self._go_offline(f"Microphone capture module unavailable: {exc}")
            return

        detector = None
        threshold_override = os.environ.get(_KWS_THRESHOLD_ENV_VAR)
        if threshold_override:
            try:
                from keyword_spotting.detector import get_keyword_detector
                detector = get_keyword_detector(threshold=float(threshold_override))
                logger.warning(
                    "%s=%s set: overriding KeywordDetector threshold for this session only "
                    "(production default in keyword_spotting/detector.py is unchanged).",
                    _KWS_THRESHOLD_ENV_VAR, threshold_override,
                )
            except ValueError:
                logger.error(
                    "%s=%r is not a valid float; ignoring and using the default threshold.",
                    _KWS_THRESHOLD_ENV_VAR, threshold_override,
                )

        try:
            orchestrator = VoiceCommandOrchestrator(
                wake_mode=WakeMode.REAL_KWS, detector=detector, log=self._log_line,
            )
        except Exception as exc:
            self._go_offline(f"Could not start the voice pipeline: {exc}")
            return

        if not getattr(orchestrator.detector, "is_ready", True):
            self._emit({
                "type": "status", "state": "READY",
                "message": "KWS model not ready — live wake detection will not fire "
                           "until a trained checkpoint is available.",
            })

        mic = MicCapture(sample_rate=orchestrator.sample_rate)
        try:
            mic.start()
        except MicCaptureError as exc:
            self._go_offline(f"Microphone unavailable: {exc}")
            return

        self.status = "READY"
        self._emit({"type": "status", "state": "READY", "message": 'Say "NOVA" to wake'})

        try:
            orchestrator.start()
            first_cycle = True
            while not self._stop_flag.is_set():
                if not first_cycle:
                    orchestrator.reset()
                first_cycle = False

                result = orchestrator.run_cycle(self._chunk_source(mic, orchestrator))
                if result.status == "INCOMPLETE":
                    # Only happens when we stopped mid-capture (see
                    # _chunk_source) -- the session isn't COMPLETE/ERROR,
                    # so reset() would raise; just stop cleanly instead.
                    break
                self._handle_result(result, orchestrator)
        except MicCaptureError as exc:
            self._go_offline(f"Microphone error during capture: {exc}")
        finally:
            mic.stop()
            if self.status != "OFFLINE":
                self.status = "STOPPED"

    def _chunk_source(self, mic, orchestrator: VoiceCommandOrchestrator) -> Iterator[np.ndarray]:
        last_state = None
        while not self._stop_flag.is_set():
            state = orchestrator.session.state
            if state != last_state:
                last_state = state
                mapped = _STATUS_FOR_SESSION_STATE.get(state)
                if mapped is not None:
                    title, sub = mapped
                    self._emit({"type": "status", "state": title, "message": sub})
            yield mic.read_chunk(timeout=1.0)

    def _handle_result(self, result: PipelineResult, orchestrator: VoiceCommandOrchestrator) -> None:
        t = result.timings
        confidence = orchestrator.session.wake_word_confidence

        if result.status == "OK" and result.command is not None:
            cmd = result.command
            update = execute_device(cmd.device, cmd.action) if cmd.device else None

            self._emit({
                "type": "voice_result",
                "wake_word": "NOVA",
                "command": result.transcript,
                "intent": cmd.intent,
                "device": cmd.device,
                "action": cmd.action,
                "confidence": cmd.confidence,
                "recognized": update is not None,
                "missing_parameters": cmd.missing_parameters,
            })
            if update is not None:
                self._emit({
                    "type": "device_update",
                    "device": update["device"],
                    "state": update["state"],
                })

            self._emit({
                "type": "telemetry",
                "kws_confidence": confidence,
                "capture_ms": t.command_capture_duration_ms,
                "asr_ms": t.asr_latency_ms,
                "iot_ms": 1.0 if update is not None else None,
                "total_ms": t.total_wake_to_result_ms,
            })
            state_title = "EXECUTED" if update is not None else "READY"
            state_msg = f"{update['device']} → {update['state']}" if update is not None else \
                f'Command not recognized: "{result.transcript}"'
            self._emit({"type": "status", "state": state_title, "message": state_msg})
        else:
            self._emit({
                "type": "status", "state": "READY",
                "message": result.message or "No command captured.",
            })
            self._emit({
                "type": "telemetry",
                "kws_confidence": confidence,
                "capture_ms": t.command_capture_duration_ms,
                "asr_ms": t.asr_latency_ms,
                "iot_ms": None,
                "total_ms": t.total_wake_to_result_ms,
            })

    def _log_line(self, line: str) -> None:
        logger.info("[live] %s", line)


_session: Optional[LiveVoiceSession] = None


def get_live_session() -> LiveVoiceSession:
    global _session
    if _session is None:
        _session = LiveVoiceSession()
    return _session
