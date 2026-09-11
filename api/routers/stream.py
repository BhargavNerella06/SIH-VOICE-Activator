"""
api.routers.stream
====================
Stage 3 development-machine audio streaming path.

    POST /stream/start              -> open a session, get a session_id
    POST /stream/{id}/chunk         -> push one small raw-PCM16 audio chunk
    POST /stream/{id}/stop          -> finalize: run ASR, then the existing
                                        command interpreter, on the fully
                                        assembled audio

This models, on one development machine:

    microphone/audio source
      -> edge-side chunking + buffering       (audio_processing.pcm,
                                                api.streaming.StreamSession)
      -> streaming audio chunks over HTTP     (this router)
      -> remote ASR server                    (asr.transcriber, unmodified;
                                                today it runs in this same
                                                process -- see
                                                docs/STAGE_3_STREAMING.md for
                                                why that is an honest stand-in
                                                for a real remote call, not a
                                                claim that it *is* remote)
      -> transcript -> existing command interpreter -> structured action

Transport is intentionally the only thing this router does. It buffers
raw PCM chunks (api.streaming.StreamSession) and, once the client calls
/stop, hands the fully assembled waveform to the existing, UNMODIFIED
asr.transcriber and command.interpreter modules exactly the way
api/routers/voice_command.py already does for a single uploaded file.
Nothing here is ASR- or command-specific: an ESP32 client could replace
the Python client simulator (scripts/stream_demo.py) by POSTing the same
raw PCM16 bytes to the same three endpoints, and nothing in this file or
in api/streaming.py would need to change.
"""

from __future__ import annotations

import time
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request

from audio_processing.pcm import pcm16_bytes_to_float32
from api.streaming import (
    StreamSessionError,
    close_session,
    create_session,
    get_session,
)
from asr.transcriber import get_transcriber
from command.interpreter import CommandInterpreter
from api.schemas import (
    ChunkTimingInfo,
    StatusInfo,
    StreamChunkResponse,
    StreamStartResponse,
    StreamStopResponse,
    StreamTiming,
)

router = APIRouter(prefix="/stream", tags=["stream"])

# faster-whisper's native sample rate; see asr/transcriber.py.
_DEFAULT_STREAM_SAMPLE_RATE = 16000


@router.post("/start", response_model=StreamStartResponse)
async def stream_start(
    sample_rate: int = Query(
        _DEFAULT_STREAM_SAMPLE_RATE,
        description="Sample rate of the PCM16 chunks that will be sent, in Hz.",
    ),
):
    """Open a new streaming session and return its session_id."""
    session = create_session(sample_rate=sample_rate)
    return StreamStartResponse(
        session_id=session.session_id,
        sample_rate=session.sample_rate,
        created_ts=session.created_ts,
    )


@router.post("/{session_id}/chunk", response_model=StreamChunkResponse)
async def stream_chunk(
    session_id: str,
    request: Request,
    capture_ts: Optional[float] = Query(
        None, description="Client-reported unix time the audio was captured."
    ),
    send_ts: Optional[float] = Query(
        None, description="Client-reported unix time the HTTP request was sent."
    ),
):
    """
    Push one small raw little-endian PCM16 audio chunk (request body =
    raw bytes, no multipart/WAV wrapping -- see audio_processing.pcm).
    """
    try:
        session = get_session(session_id)
    except StreamSessionError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    body = await request.body()
    if len(body) == 0:
        raise HTTPException(status_code=400, detail="Empty chunk body.")
    try:
        waveform = pcm16_bytes_to_float32(body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    try:
        timing = session.add_chunk(waveform, capture_ts=capture_ts, send_ts=send_ts)
    except StreamSessionError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    return StreamChunkResponse(
        seq=timing.seq,
        n_samples=timing.n_samples,
        receive_ts=timing.receive_ts,
        total_duration_s=session.audio_duration_seconds,
    )


@router.post("/{session_id}/stop", response_model=StreamStopResponse)
async def stream_stop(session_id: str):
    """
    Finalize the stream: assemble every received chunk, run ASR, then
    the existing command interpreter, and return the structured result
    plus a full latency breakdown.
    """
    try:
        session = get_session(session_id)
    except StreamSessionError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    if session.closed:
        raise HTTPException(
            status_code=409, detail=f"Session {session_id} is already finalized."
        )

    session.stop_received_ts = time.time()
    close_session(session_id)

    waveform = session.assembled_waveform()

    session.asr_start_ts = time.time()
    asr_result = get_transcriber().transcribe(waveform, sample_rate=session.sample_rate)
    session.transcript_ready_ts = time.time()

    cmd_result = CommandInterpreter().interpret(asr_result.text)
    session.command_parsed_ts = time.time()

    timing = StreamTiming(
        session_created_ts=session.created_ts,
        chunks=[
            ChunkTimingInfo(
                seq=c.seq, n_samples=c.n_samples, receive_ts=c.receive_ts,
                capture_ts=c.capture_ts, send_ts=c.send_ts,
            )
            for c in session.chunk_timings
        ],
        audio_duration_s=session.audio_duration_seconds,
        n_chunks=len(session.chunk_timings),
        stop_received_ts=session.stop_received_ts,
        asr_start_ts=session.asr_start_ts,
        transcript_ready_ts=session.transcript_ready_ts,
        command_parsed_ts=session.command_parsed_ts,
        asr_latency_ms=(session.transcript_ready_ts - session.asr_start_ts) * 1000.0,
        command_latency_ms=(session.command_parsed_ts - session.transcript_ready_ts) * 1000.0,
        server_total_ms=(session.command_parsed_ts - session.created_ts) * 1000.0,
    )

    return StreamStopResponse(
        asr=StatusInfo(status=asr_result.status, message=asr_result.message),
        transcript=asr_result.text,
        command=StatusInfo(status=cmd_result.status, message=cmd_result.message),
        action=cmd_result.action,
        parameters=cmd_result.parameters,
        missing_parameters=cmd_result.missing_parameters,
        timing=timing,
    )
