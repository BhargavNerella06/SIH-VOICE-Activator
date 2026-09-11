"""
voice_pipeline.remote_asr_client
====================================
RemoteASRClient - the "EDGE" side's client for the remote-ASR streaming
protocol implemented by api/routers/asr_stream.py.

    AudioSessionManager.CommandAudioSegment
      -> RemoteASRClient.send_command_audio()
      -> ws://<host>:<port>/asr/stream (see api/routers/asr_stream.py
         for the full protocol: start handshake, binary PCM16 chunks,
         explicit "end" signal, JSON result)
      -> RemoteASRResult (transcript + command interpretation, both
         already run server-side -- this client does NOT re-run ASR or
         command interpretation locally, per this task's explicit
         edge/remote boundary)

For this development stage, `host` defaults to "127.0.0.1" (the ASR
server runs on the same laptop) -- nothing else about this client
assumes that; pointing it at a different host later requires no other
change here or on the server.

Uses `websockets.sync.client` (already installed via uvicorn[standard]
-- no new dependency) for a plain synchronous client, matching this
project's other demo scripts (stream_demo.py, voice_command_demo.py),
none of which use asyncio.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from audio_processing.pcm import DEFAULT_CHUNK_MS, float32_to_pcm16_bytes, frame_waveform

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_CONNECT_TIMEOUT_S = 5.0
DEFAULT_RESULT_TIMEOUT_S = 30.0


class RemoteASRClientError(RuntimeError):
    """Reserved for programming-usage errors. Network/protocol failures
    are reported via RemoteASRResult.status instead of raising -- see
    send_command_audio()'s docstring."""


@dataclass
class RemoteASRTimings:
    """Client-side wall-clock timestamps. All *_ms properties return
    None if the underlying timestamps aren't both set -- never a
    fabricated duration."""
    client_start_ts: Optional[float] = None
    connected_ts: Optional[float] = None
    stream_start_ts: Optional[float] = None
    stream_end_ts: Optional[float] = None
    result_received_ts: Optional[float] = None

    @property
    def streaming_duration_ms(self) -> Optional[float]:
        """Time spent sending audio chunks (client-side)."""
        if self.stream_start_ts is None or self.stream_end_ts is None:
            return None
        return (self.stream_end_ts - self.stream_start_ts) * 1000.0

    @property
    def server_round_trip_ms(self) -> Optional[float]:
        """Time from finishing sending audio ('end' sent) to receiving
        the server's result. This necessarily bundles network transit
        AND server-side ASR/interpretation time together -- from the
        client's vantage point alone, the two can't be cleanly
        separated (see the server's own `timings` in the result for
        its internal breakdown, e.g. `asr_latency_ms`)."""
        if self.stream_end_ts is None or self.result_received_ts is None:
            return None
        return (self.result_received_ts - self.stream_end_ts) * 1000.0

    @property
    def total_client_ms(self) -> Optional[float]:
        """Wall-clock time from starting the whole operation (including
        connecting) to receiving the final result."""
        if self.client_start_ts is None or self.result_received_ts is None:
            return None
        return (self.result_received_ts - self.client_start_ts) * 1000.0


@dataclass
class RemoteASRResult:
    """
    status : "OK" | "ASR_ERROR" | "EMPTY_AUDIO" | "CONNECTION_ERROR" | "PROTOCOL_ERROR"
        The first three are relayed directly from the server's own
        result (see api/routers/asr_stream.py). CONNECTION_ERROR and
        PROTOCOL_ERROR are client-side: the server was unreachable, the
        connection dropped, or it sent something not matching the
        expected protocol -- never a crash, always a clear status.
    """
    status: str
    transcript: str = ""
    command: Optional[Dict[str, Any]] = None
    server_timings: Dict[str, Any] = field(default_factory=dict)
    client_timings: RemoteASRTimings = field(default_factory=RemoteASRTimings)
    message: str = ""


class RemoteASRClient:
    """
    Parameters
    ----------
    host, port : str, int
        Where the remote ASR server (api/routers/asr_stream.py) is
        listening. Defaults to 127.0.0.1:8000 for this stage's
        localhost development setup.
    connect_timeout : float
        Seconds to wait for the initial WebSocket connection + 'ready'
        handshake before giving up.
    result_timeout : float
        Seconds to wait for the server's final result after sending
        'end' (covers server-side ASR + interpretation time).
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT_S,
        result_timeout: float = DEFAULT_RESULT_TIMEOUT_S,
    ) -> None:
        self.host = host
        self.port = port
        self.connect_timeout = connect_timeout
        self.result_timeout = result_timeout

    @property
    def url(self) -> str:
        return f"ws://{self.host}:{self.port}/asr/stream"

    def send_command_audio(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        chunk_ms: float = DEFAULT_CHUNK_MS,
    ) -> RemoteASRResult:
        """
        Stream one complete command-audio segment (e.g. from
        AudioSessionManager's CommandAudioSegment) to the remote ASR
        server and return its result.

        Never raises for network/protocol failures -- connection
        refused, timeouts, and unexpected server responses all come
        back as a RemoteASRResult with a "CONNECTION_ERROR" or
        "PROTOCOL_ERROR" status and a clear `message`, so a live
        edge-device loop doesn't need to wrap every call in try/except.
        """
        from websockets.exceptions import ConnectionClosed
        from websockets.sync.client import connect

        timings = RemoteASRTimings(client_start_ts=time.time())

        try:
            ws = connect(self.url, open_timeout=self.connect_timeout)
        except Exception as exc:
            return RemoteASRResult(
                status="CONNECTION_ERROR",
                client_timings=timings,
                message=f"Could not connect to remote ASR server at {self.url}: {exc}",
            )
        timings.connected_ts = time.time()

        try:
            with ws:
                ws.send(json.dumps({
                    "type": "start", "sample_rate": sample_rate, "channels": 1, "format": "pcm16",
                }))
                ready = json.loads(ws.recv(timeout=self.connect_timeout))
                if ready.get("type") == "error":
                    return RemoteASRResult(
                        status="PROTOCOL_ERROR", client_timings=timings,
                        message=ready.get("message", "Server rejected the session."),
                    )
                if ready.get("type") != "ready":
                    return RemoteASRResult(
                        status="PROTOCOL_ERROR", client_timings=timings,
                        message=f"Unexpected server response to 'start': {ready}",
                    )

                timings.stream_start_ts = time.time()
                for chunk in frame_waveform(waveform, sample_rate, chunk_ms=chunk_ms):
                    ws.send(float32_to_pcm16_bytes(chunk))
                timings.stream_end_ts = time.time()

                ws.send(json.dumps({"type": "end"}))

                raw_result = ws.recv(timeout=self.result_timeout)
                timings.result_received_ts = time.time()
                result = json.loads(raw_result)
        except TimeoutError as exc:
            return RemoteASRResult(
                status="CONNECTION_ERROR", client_timings=timings,
                message=f"Timed out waiting for the remote ASR server: {exc}",
            )
        except ConnectionClosed as exc:
            return RemoteASRResult(
                status="CONNECTION_ERROR", client_timings=timings,
                message=f"Connection to remote ASR server closed unexpectedly: {exc}",
            )
        except Exception as exc:
            return RemoteASRResult(
                status="CONNECTION_ERROR", client_timings=timings,
                message=f"Error communicating with remote ASR server: {exc}",
            )

        if result.get("type") != "result":
            return RemoteASRResult(
                status="PROTOCOL_ERROR", client_timings=timings,
                message=f"Unexpected message from server (expected a 'result'): {result}",
            )

        return RemoteASRResult(
            status=result.get("status", "PROTOCOL_ERROR"),
            transcript=result.get("transcript", ""),
            command=result.get("command"),
            server_timings=result.get("timings", {}),
            client_timings=timings,
            message=result.get("message", ""),
        )
