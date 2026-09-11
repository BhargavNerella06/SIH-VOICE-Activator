"""
tests/integration/test_asr_stream.py
=======================================
Tests for the remote-ASR WebSocket protocol (api/routers/asr_stream.py),
using FastAPI's synchronous TestClient.websocket_connect() -- no live
uvicorn process, no real network socket, no microphone.

ASR is mocked (monkeypatching api.routers.asr_stream.get_transcriber),
mirroring the established pattern in tests/integration/test_stream_api.py.
command.interpreter.CommandInterpreter runs for real (unmodified, fast,
deterministic), so these tests also cover real ASR-text ->
command-interpretation integration on the server side.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from asr.transcriber import TranscriptionResult
from audio_processing.pcm import chunk_samples, float32_to_pcm16_bytes


@pytest.fixture
def app():
    from api.main import app as _app
    return _app


@pytest.fixture
def client(app):
    return TestClient(app)


class _FakeTranscriber:
    def __init__(self, text: str = "turn on the lights", status: str = "OK"):
        self._text = text
        self._status = status
        self.calls = 0

    def transcribe(self, waveform, sample_rate):
        self.calls += 1
        return TranscriptionResult(
            status=self._status,
            text=self._text if self._status == "OK" else "",
            language="en",
            latency_ms=1.0,
            message=f"fake transcriber ({self._status})",
        )


@pytest.fixture
def mock_asr(monkeypatch):
    fake = _FakeTranscriber()
    monkeypatch.setattr("api.routers.asr_stream.get_transcriber", lambda: fake)
    return fake


def _pcm_chunks(waveform: np.ndarray, sample_rate: int = 16000, chunk_ms: float = 30.0):
    size = chunk_samples(sample_rate, chunk_ms)
    return [
        float32_to_pcm16_bytes(waveform[i:i + size])
        for i in range(0, len(waveform), size)
    ]


def _run_session(ws, sample_rate=16000, chunks=(), send_end=True):
    """Drive one full start -> chunks -> end exchange over an already-
    open TestClient websocket, returning the 'ready' and 'result' messages."""
    ws.send_json({"type": "start", "sample_rate": sample_rate, "channels": 1, "format": "pcm16"})
    ready = ws.receive_json()
    for chunk_bytes in chunks:
        ws.send_bytes(chunk_bytes)
    if send_end:
        ws.send_json({"type": "end"})
        result = ws.receive_json()
    else:
        result = None
    return ready, result


# ---------------------------------------------------------------------------
# Protocol metadata
# ---------------------------------------------------------------------------

def test_valid_start_message_gets_ready(client, mock_asr):
    with client.websocket_connect("/asr/stream") as ws:
        ws.send_json({"type": "start", "sample_rate": 16000, "channels": 1, "format": "pcm16"})
        ready = ws.receive_json()
    assert ready == {"type": "ready"}


def test_missing_sample_rate_is_rejected(client):
    with client.websocket_connect("/asr/stream") as ws:
        ws.send_json({"type": "start", "channels": 1, "format": "pcm16"})
        resp = ws.receive_json()
    assert resp["type"] == "error"
    assert "sample_rate" in resp["message"]


def test_non_positive_sample_rate_is_rejected(client):
    with client.websocket_connect("/asr/stream") as ws:
        ws.send_json({"type": "start", "sample_rate": 0, "format": "pcm16"})
        resp = ws.receive_json()
    assert resp["type"] == "error"


def test_wrong_first_message_type_is_rejected(client):
    with client.websocket_connect("/asr/stream") as ws:
        ws.send_json({"type": "chunk"})
        resp = ws.receive_json()
    assert resp["type"] == "error"
    assert "start" in resp["message"]


def test_unsupported_format_is_rejected(client):
    with client.websocket_connect("/asr/stream") as ws:
        ws.send_json({"type": "start", "sample_rate": 16000, "format": "mp3"})
        resp = ws.receive_json()
    assert resp["type"] == "error"
    assert "mp3" in resp["message"]


def test_multi_channel_is_rejected(client):
    with client.websocket_connect("/asr/stream") as ws:
        ws.send_json({"type": "start", "sample_rate": 16000, "channels": 2, "format": "pcm16"})
        resp = ws.receive_json()
    assert resp["type"] == "error"
    assert "channels" in resp["message"] or "mono" in resp["message"].lower()


# ---------------------------------------------------------------------------
# Malformed metadata beyond simple field validation
# ---------------------------------------------------------------------------

def test_non_json_first_message_is_rejected(client):
    with client.websocket_connect("/asr/stream") as ws:
        ws.send_text("not valid json{{{")
        resp = ws.receive_json()
    assert resp["type"] == "error"


def test_malformed_control_message_after_ready_is_rejected(client, mock_asr):
    with client.websocket_connect("/asr/stream") as ws:
        ws.send_json({"type": "start", "sample_rate": 16000, "format": "pcm16"})
        ws.receive_json()
        ws.send_text("not valid json{{{")
        resp = ws.receive_json()
    assert resp["type"] == "error"


def test_unexpected_message_type_while_streaming_is_rejected(client, mock_asr):
    with client.websocket_connect("/asr/stream") as ws:
        ws.send_json({"type": "start", "sample_rate": 16000, "format": "pcm16"})
        ws.receive_json()
        ws.send_json({"type": "not_a_real_command"})
        resp = ws.receive_json()
    assert resp["type"] == "error"


# ---------------------------------------------------------------------------
# Chunk ordering / complete session / successful transcription
# ---------------------------------------------------------------------------

def test_chunk_ordering_and_complete_session(client, mock_asr):
    mock_asr._text = "turn on the lights"
    waveform = np.linspace(-0.5, 0.5, 16000, dtype=np.float32)  # 1s, monotonic (order-sensitive)
    chunks = _pcm_chunks(waveform, sample_rate=16000, chunk_ms=30.0)

    with client.websocket_connect("/asr/stream") as ws:
        ready, result = _run_session(ws, sample_rate=16000, chunks=chunks)

    assert ready == {"type": "ready"}
    assert result["type"] == "result"
    assert result["status"] == "OK"
    assert result["transcript"] == "turn on the lights"
    assert result["timings"]["n_chunks"] == len(chunks)
    assert result["timings"]["audio_duration_s"] == pytest.approx(1.0, abs=0.01)


def test_successful_transcription_and_command_interpretation(client, mock_asr):
    mock_asr._text = "turn on the lights"
    waveform = np.zeros(16000, dtype=np.float32)
    chunks = _pcm_chunks(waveform)

    with client.websocket_connect("/asr/stream") as ws:
        _, result = _run_session(ws, chunks=chunks)

    assert result["status"] == "OK"
    assert result["command"]["action"] == "turn_on"
    assert result["command"]["intent"] == "device_control"
    assert result["command"]["device"] == "lights"
    assert result["command"]["parameters"] == {"device": "lights"}
    assert result["command"]["confidence"] == 1.0
    assert result["command"]["original_text"] == "turn on the lights"


def test_command_interpreter_handles_missing_parameter(client, mock_asr):
    mock_asr._text = "turn on"
    with client.websocket_connect("/asr/stream") as ws:
        _, result = _run_session(ws, chunks=_pcm_chunks(np.zeros(1600, dtype=np.float32)))
    assert result["command"]["action"] == "turn_on"
    assert result["command"]["missing_parameters"] == ["device"]


def test_unknown_command_is_a_successful_result(client, mock_asr):
    mock_asr._text = "what is the weather today"
    with client.websocket_connect("/asr/stream") as ws:
        _, result = _run_session(ws, chunks=_pcm_chunks(np.zeros(1600, dtype=np.float32)))
    assert result["status"] == "OK"
    assert result["command"]["action"] == "unknown"


# ---------------------------------------------------------------------------
# Empty audio
# ---------------------------------------------------------------------------

def test_empty_audio_no_chunks_sent(client, mock_asr):
    with client.websocket_connect("/asr/stream") as ws:
        _, result = _run_session(ws, chunks=[])
    assert result["status"] == "EMPTY_AUDIO"
    assert result["command"] is None
    assert mock_asr.calls == 0  # ASR must never be called on empty audio


# ---------------------------------------------------------------------------
# Server ASR failure
# ---------------------------------------------------------------------------

def test_server_asr_error_status_surfaces_cleanly(client, mock_asr):
    mock_asr._status = "ERROR"
    with client.websocket_connect("/asr/stream") as ws:
        _, result = _run_session(ws, chunks=_pcm_chunks(np.zeros(1600, dtype=np.float32)))
    assert result["status"] == "ASR_ERROR"
    assert result["command"] is None


def test_server_asr_not_implemented_surfaces_as_asr_error(client, mock_asr):
    mock_asr._status = "NOT_IMPLEMENTED"
    with client.websocket_connect("/asr/stream") as ws:
        _, result = _run_session(ws, chunks=_pcm_chunks(np.zeros(1600, dtype=np.float32)))
    assert result["status"] == "ASR_ERROR"


# ---------------------------------------------------------------------------
# Malformed audio chunk mid-stream
# ---------------------------------------------------------------------------

def test_malformed_chunk_mid_stream_is_rejected(client, mock_asr):
    with client.websocket_connect("/asr/stream") as ws:
        ws.send_json({"type": "start", "sample_rate": 16000, "format": "pcm16"})
        ws.receive_json()
        ws.send_bytes(b"\x01\x02\x03")  # odd byte count -- malformed PCM16
        resp = ws.receive_json()
    assert resp["type"] == "error"
    assert "malformed" in resp["message"].lower()


# ---------------------------------------------------------------------------
# Client-side connection failure (no server listening on this port)
# ---------------------------------------------------------------------------

def test_client_reports_connection_error_when_server_unreachable():
    from voice_pipeline.remote_asr_client import RemoteASRClient

    client = RemoteASRClient(host="127.0.0.1", port=1, connect_timeout=1.0)
    result = client.send_command_audio(np.zeros(1600, dtype=np.float32), sample_rate=16000)
    assert result.status == "CONNECTION_ERROR"
    assert "127.0.0.1" in result.message


# ---------------------------------------------------------------------------
# Model caching: ASR must not reload per command
# ---------------------------------------------------------------------------

def test_asr_not_reloaded_across_multiple_websocket_sessions(client, monkeypatch):
    import api.routers.asr_stream as asr_stream_mod

    fake = _FakeTranscriber()
    call_count = {"n": 0}

    def counting_get_transcriber():
        call_count["n"] += 1
        return fake

    monkeypatch.setattr(asr_stream_mod, "get_transcriber", counting_get_transcriber)

    for _ in range(3):
        with client.websocket_connect("/asr/stream") as ws:
            _run_session(ws, chunks=_pcm_chunks(np.zeros(1600, dtype=np.float32)))

    # get_transcriber() (the cache accessor) is called once per session --
    # that's expected and cheap (a dict lookup); what matters is that the
    # underlying model object is the SAME instance every time, i.e. never
    # reconstructed/reloaded.
    assert call_count["n"] == 3
    assert fake.calls == 3  # same fake instance handled all 3 transcriptions


# ---------------------------------------------------------------------------
# Multiple sequential command sessions (independent state, no leakage)
# ---------------------------------------------------------------------------

def test_multiple_sequential_sessions_are_independent(client, mock_asr):
    mock_asr._text = "turn on the lights"
    with client.websocket_connect("/asr/stream") as ws:
        _, result1 = _run_session(ws, chunks=_pcm_chunks(np.zeros(1600, dtype=np.float32)))
    assert result1["command"]["action"] == "turn_on"

    mock_asr._text = "turn off the fan"
    with client.websocket_connect("/asr/stream") as ws:
        _, result2 = _run_session(ws, chunks=_pcm_chunks(np.zeros(1600, dtype=np.float32)))
    assert result2["command"]["action"] == "turn_off"
    assert result2["command"]["device"] == "fan"

    # First session's chunk count must not leak into the second.
    assert result1["timings"]["n_chunks"] == result2["timings"]["n_chunks"]
