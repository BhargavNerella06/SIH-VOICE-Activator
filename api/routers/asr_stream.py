"""
api.routers.asr_stream
=========================
WebSocket remote-ASR streaming endpoint -- the "REMOTE ASR SERVER" side
of the SIH target architecture:

    EDGE DEVICE: mic -> KWS -> NOVA -> command audio
      -- wireless/network stream -->
    REMOTE ASR SERVER: ASR -> command interpreter -> structured result

For this development stage, both sides run on the same laptop via
127.0.0.1 (see voice_pipeline/remote_asr_client.py and
scripts/remote_asr_client_demo.py). Nothing in this protocol assumes
that -- the ASR server can later run on a genuinely separate machine
with zero protocol changes; only the client's connection target
(currently 127.0.0.1) would change.

This does NOT run KWS or wake-word detection. Wake detection and
command-audio endpointing (pre-roll, silence/min/max-duration capture)
already happened upstream, on the edge, via
audio_processing.session.AudioSessionManager -- this server only ever
receives already-gated "command audio" and treats it as one complete
utterance to transcribe. It reuses the EXISTING, UNMODIFIED
asr.transcriber (cached) and command.interpreter -- no second ASR or
command-parsing implementation exists here.

Protocol (one WebSocket connection == one command-audio session -- no
separate session-id/store needed, unlike api/streaming.py's HTTP
POST-per-chunk design, since the connection itself IS the session)
------------------------------------------------------------------------
1. Client connects to ws://<host>:<port>/asr/stream
2. Client sends one JSON text message:
       {"type": "start", "sample_rate": 16000, "channels": 1, "format": "pcm16"}
   Server validates it and replies {"type": "ready"}, or
   {"type": "error", "message": ...} followed by closing the connection.
3. Client sends N binary WebSocket frames, each one raw little-endian
   PCM16 chunk -- the SAME wire format audio_processing.pcm and
   api/routers/stream.py already use, so this project has exactly one
   audio-chunk encoding, not two.
4. Client sends one JSON text message: {"type": "end"} -- an EXPLICIT
   command-completion signal. There is no implicit end-of-stream or
   silence-based cutoff here (that already happened upstream, on the
   edge); this server trusts the client's "end" signal completely.
5. Server reconstructs the waveform, runs ASR then command
   interpretation, and replies with one JSON text message:
       {"type": "result", "status": ..., "transcript": ..., "command": {...}, "timings": {...}}
6. Server closes the connection.

A client that disconnects before sending "end", or sends malformed
metadata/audio, is handled cleanly (logged, connection closed) -- never
a server crash and never a fabricated result.

Architecture boundary (per this task's explicit instruction)
------------------------------------------------------------------
EDGE  : KWS + audio capture/session -- not touched by this module.
REMOTE: ASR + command interpretation -- both run HERE, once, server-
        side. Command interpretation is NOT duplicated on the client.
"""

from __future__ import annotations

import json
import logging
import time
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from audio_processing.pcm import pcm16_bytes_to_float32
from asr.transcriber import get_transcriber
from command.interpreter import CommandInterpreter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/asr", tags=["asr-stream"])

SUPPORTED_FORMATS = ("pcm16",)

# WebSocket close code for a protocol violation (RFC 6455 1003: "cannot
# accept the type of data").
_CLOSE_PROTOCOL_ERROR = 1003


async def _send_error_and_close(websocket: WebSocket, message: str) -> None:
    logger.warning("asr_stream protocol error: %s", message)
    try:
        await websocket.send_json({"type": "error", "message": message})
    except Exception:
        pass  # connection may already be gone; nothing more we can do
    try:
        await websocket.close(code=_CLOSE_PROTOCOL_ERROR)
    except Exception:
        pass


def _validate_start_message(raw: dict) -> Optional[str]:
    """Return an error message string if `raw` is not a valid 'start'
    handshake, else None."""
    if raw.get("type") != "start":
        return f"Expected message type 'start', got {raw.get('type')!r}."
    sample_rate = raw.get("sample_rate")
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate <= 0:
        return f"Invalid or missing 'sample_rate': {raw.get('sample_rate')!r} (must be a positive integer)."
    channels = raw.get("channels", 1)
    if channels != 1:
        return f"Only mono audio (channels=1) is supported, got {channels!r}."
    audio_format = raw.get("format", "pcm16")
    if audio_format not in SUPPORTED_FORMATS:
        return f"Unsupported audio format {audio_format!r}; expected one of {SUPPORTED_FORMATS}."
    return None


@router.websocket("/stream")
async def asr_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    session_created_ts = time.time()

    # ------------------------------------------------------------------
    # Step 1: metadata handshake
    # ------------------------------------------------------------------
    try:
        raw = await websocket.receive_json()
    except (json.JSONDecodeError, ValueError) as exc:
        await _send_error_and_close(websocket, f"Expected a JSON 'start' message: {exc}")
        return
    except WebSocketDisconnect:
        logger.info("Client disconnected before sending a 'start' message.")
        return

    error = _validate_start_message(raw)
    if error is not None:
        await _send_error_and_close(websocket, error)
        return

    sample_rate: int = raw["sample_rate"]
    await websocket.send_json({"type": "ready"})

    # ------------------------------------------------------------------
    # Step 2/3: receive binary audio chunks until an 'end' control message
    # ------------------------------------------------------------------
    chunks: List[np.ndarray] = []
    chunk_count = 0
    stream_start_ts: Optional[float] = None
    stream_end_ts: Optional[float] = None

    try:
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                logger.info(
                    "Client disconnected mid-stream (after %d chunk(s), before 'end').",
                    chunk_count,
                )
                return

            data = message.get("bytes")
            if data is not None:
                if stream_start_ts is None:
                    stream_start_ts = time.time()
                try:
                    waveform = pcm16_bytes_to_float32(data)
                except ValueError as exc:
                    await _send_error_and_close(websocket, f"Malformed audio chunk: {exc}")
                    return
                chunks.append(waveform)
                chunk_count += 1
                continue

            text = message.get("text")
            if text is not None:
                try:
                    control = json.loads(text)
                except json.JSONDecodeError as exc:
                    await _send_error_and_close(websocket, f"Malformed control message: {exc}")
                    return
                if control.get("type") == "end":
                    stream_end_ts = time.time()
                    break
                await _send_error_and_close(
                    websocket,
                    f"Unexpected message type {control.get('type')!r} while streaming audio "
                    "(expected binary chunks or a final {'type': 'end'}).",
                )
                return
    except WebSocketDisconnect:
        logger.info(
            "Client disconnected mid-stream (after %d chunk(s), before 'end').", chunk_count,
        )
        return

    # ------------------------------------------------------------------
    # Step 4/5: ASR (existing, cached) + command interpretation (existing)
    # ------------------------------------------------------------------
    waveform = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    timings = {
        "session_created_ts": session_created_ts,
        "stream_start_ts": stream_start_ts,
        "stream_end_ts": stream_end_ts,
        "n_chunks": chunk_count,
        "audio_duration_s": waveform.size / float(sample_rate),
    }

    if waveform.size == 0:
        await _send_result(
            websocket, status="EMPTY_AUDIO", transcript="", command=None, timings=timings,
            message="No audio received; nothing to transcribe.",
        )
        return

    asr_start_ts = time.time()
    asr_result = get_transcriber().transcribe(waveform, sample_rate=sample_rate)
    asr_end_ts = time.time()
    timings["asr_start_ts"] = asr_start_ts
    timings["asr_end_ts"] = asr_end_ts
    timings["asr_latency_ms"] = (asr_end_ts - asr_start_ts) * 1000.0

    if asr_result.status != "OK":
        await _send_result(
            websocket, status="ASR_ERROR", transcript="", command=None, timings=timings,
            message=f"ASR failed ({asr_result.status}): {asr_result.message}",
        )
        return

    cmd_result = CommandInterpreter().interpret(asr_result.text)
    timings["command_interpreted_ts"] = time.time()

    await _send_result(
        websocket,
        status="OK",
        transcript=asr_result.text,
        command={
            "status": cmd_result.status,
            "action": cmd_result.action,
            "parameters": cmd_result.parameters,
            "missing_parameters": cmd_result.missing_parameters,
            "intent": cmd_result.intent,
            "device": cmd_result.device,
            "original_text": cmd_result.original_text,
            "confidence": cmd_result.confidence,
            "message": cmd_result.message,
        },
        timings=timings,
        message="",
    )


async def _send_result(
    websocket: WebSocket, *, status: str, transcript: str, command: Optional[dict],
    timings: dict, message: str,
) -> None:
    try:
        await websocket.send_json({
            "type": "result",
            "status": status,
            "transcript": transcript,
            "command": command,
            "timings": timings,
            "message": message,
        })
    except Exception as exc:
        logger.warning("Could not send result -- client likely disconnected: %s", exc)
        return
    try:
        await websocket.close()
    except Exception:
        pass
