"""
tests/integration/test_stream_api.py
=======================================
HTTP-level tests for the Stage 3 /stream/* endpoints:
start -> chunk (x N) -> stop.

ASR is mocked here (asr.transcriber.get_transcriber is monkeypatched to
return a fake Transcriber-like object with a deterministic transcribe()),
mirroring the existing approach in tests/unit/test_asr_transcriber.py of
never depending on a real model download for deterministic test
behaviour. command.interpreter is exercised for real (unmodified,
fast, no model) so these tests also cover real ASR->command
integration, per the task requirement.
"""

from __future__ import annotations

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

from audio_processing.pcm import chunk_samples, float32_to_pcm16_bytes
from asr.transcriber import TranscriptionResult


# ---------------------------------------------------------------------------
# App / client fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def app():
    from api.main import app as _app
    return _app


@pytest.fixture
def transport(app):
    return ASGITransport(app=app)


class _FakeTranscriber:
    """Deterministic stand-in for asr.transcriber.Transcriber."""

    def __init__(self, text: str = "turn on the lights", status: str = "OK"):
        self._text = text
        self._status = status

    def transcribe(self, waveform, sample_rate):
        return TranscriptionResult(
            status=self._status,
            text=self._text if self._status == "OK" else "",
            language="en",
            latency_ms=1.0,
            message=f"fake transcriber ({self._status})",
        )


@pytest.fixture
def mock_asr(monkeypatch):
    """Patch the transcriber used by api.routers.stream with a fake one.
    Returns the fake instance so a test can customize its text/status."""
    fake = _FakeTranscriber()
    monkeypatch.setattr("api.routers.stream.get_transcriber", lambda: fake)
    return fake


def _pcm_chunks(waveform: np.ndarray, sample_rate: int = 16000, chunk_ms: float = 30.0):
    size = chunk_samples(sample_rate, chunk_ms)
    return [
        float32_to_pcm16_bytes(waveform[i:i + size])
        for i in range(0, len(waveform), size)
    ]


async def _start_session(client, sample_rate=16000) -> str:
    resp = await client.post("/stream/start", params={"sample_rate": sample_rate})
    assert resp.status_code == 200, resp.text
    return resp.json()["session_id"]


# ---------------------------------------------------------------------------
# Stream lifecycle
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stream_start_returns_a_session_id(transport):
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/stream/start", params={"sample_rate": 16000})
    assert resp.status_code == 200
    data = resp.json()
    assert "session_id" in data and data["session_id"]
    assert data["sample_rate"] == 16000


@pytest.mark.asyncio
async def test_stream_chunk_unknown_session_returns_404(transport):
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/stream/does-not-exist/chunk",
            content=float32_to_pcm16_bytes(np.zeros(160, dtype=np.float32)),
        )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_stream_stop_unknown_session_returns_404(transport):
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/stream/does-not-exist/stop")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_stop_twice_returns_409_on_second_call(transport, mock_asr):
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        session_id = await _start_session(client)
        first = await client.post(f"/stream/{session_id}/stop")
        assert first.status_code == 200
        second = await client.post(f"/stream/{session_id}/stop")
    assert second.status_code == 409


@pytest.mark.asyncio
async def test_chunk_after_stop_returns_409(transport, mock_asr):
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        session_id = await _start_session(client)
        stop_resp = await client.post(f"/stream/{session_id}/stop")
        assert stop_resp.status_code == 200
        chunk_resp = await client.post(
            f"/stream/{session_id}/chunk",
            content=float32_to_pcm16_bytes(np.zeros(160, dtype=np.float32)),
        )
    assert chunk_resp.status_code == 409


# ---------------------------------------------------------------------------
# Malformed / incomplete audio
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_empty_chunk_body_returns_400(transport):
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        session_id = await _start_session(client)
        resp = await client.post(f"/stream/{session_id}/chunk", content=b"")
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_odd_byte_count_chunk_returns_400(transport):
    """3 bytes cannot be a whole number of 16-bit PCM samples."""
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        session_id = await _start_session(client)
        resp = await client.post(f"/stream/{session_id}/chunk", content=b"\x01\x02\x03")
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Server receiving multiple chunks
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_single_chunk_accepted_and_reports_duration(transport):
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        session_id = await _start_session(client, sample_rate=16000)
        chunk = float32_to_pcm16_bytes(np.zeros(480, dtype=np.float32))  # 30ms @ 16k
        resp = await client.post(f"/stream/{session_id}/chunk", content=chunk)
    assert resp.status_code == 200
    data = resp.json()
    assert data["seq"] == 0
    assert data["n_samples"] == 480
    assert data["total_duration_s"] == pytest.approx(0.03, abs=1e-6)


@pytest.mark.asyncio
async def test_multiple_chunks_accumulate_across_requests(transport):
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        session_id = await _start_session(client, sample_rate=16000)
        chunk = float32_to_pcm16_bytes(np.zeros(480, dtype=np.float32))
        seqs = []
        for _ in range(5):
            resp = await client.post(f"/stream/{session_id}/chunk", content=chunk)
            assert resp.status_code == 200
            seqs.append(resp.json()["seq"])
        last = resp.json()
    assert seqs == [0, 1, 2, 3, 4]
    assert last["total_duration_s"] == pytest.approx(5 * 480 / 16000, abs=1e-6)


# ---------------------------------------------------------------------------
# ASR + command integration (ASR mocked, command interpreter real/unmodified)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_full_stream_lifecycle_with_mocked_asr_and_real_command_interpreter(transport, mock_asr):
    mock_asr._text = "turn on the lights"

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        session_id = await _start_session(client, sample_rate=16000)
        waveform = np.zeros(16000, dtype=np.float32)  # 1s of silence; content doesn't matter, ASR is mocked
        for payload in _pcm_chunks(waveform, sample_rate=16000, chunk_ms=30.0):
            resp = await client.post(
                f"/stream/{session_id}/chunk",
                params={"capture_ts": 1000.0, "send_ts": 1000.01},
                content=payload,
            )
            assert resp.status_code == 200

        stop_resp = await client.post(f"/stream/{session_id}/stop")

    assert stop_resp.status_code == 200
    data = stop_resp.json()
    assert data["asr"]["status"] == "OK"
    assert data["transcript"] == "turn on the lights"
    assert data["command"]["status"] == "OK"
    assert data["action"] == "turn_on"
    assert data["parameters"] == {"device": "lights"}
    assert data["missing_parameters"] == []

    timing = data["timing"]
    assert timing["n_chunks"] > 0
    assert timing["audio_duration_s"] == pytest.approx(1.0, abs=0.01)
    assert timing["asr_latency_ms"] >= 0.0
    assert timing["command_latency_ms"] >= 0.0
    assert timing["server_total_ms"] >= 0.0
    # Client-reported per-chunk timestamps must be threaded all the way through.
    assert timing["chunks"][0]["capture_ts"] == 1000.0
    assert timing["chunks"][0]["send_ts"] == 1000.01


@pytest.mark.asyncio
async def test_stream_with_missing_device_command(transport, mock_asr):
    mock_asr._text = "turn on"  # recognized verb, no device
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        session_id = await _start_session(client)
        stop_resp = await client.post(f"/stream/{session_id}/stop")
    data = stop_resp.json()
    assert data["action"] == "turn_on"
    assert data["missing_parameters"] == ["device"]


@pytest.mark.asyncio
async def test_stream_with_unrecognized_command(transport, mock_asr):
    mock_asr._text = "what is the weather today"
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        session_id = await _start_session(client)
        stop_resp = await client.post(f"/stream/{session_id}/stop")
    data = stop_resp.json()
    assert data["action"] == "unknown"
    assert data["missing_parameters"] == []


@pytest.mark.asyncio
async def test_stream_when_asr_not_implemented(transport, mock_asr):
    """If ASR reports NOT_IMPLEMENTED (e.g. model unavailable), the stream
    endpoint must surface that honestly rather than fabricating a transcript,
    and the command interpreter must still run cleanly on empty text."""
    mock_asr._status = "NOT_IMPLEMENTED"
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        session_id = await _start_session(client)
        stop_resp = await client.post(f"/stream/{session_id}/stop")
    data = stop_resp.json()
    assert data["asr"]["status"] == "NOT_IMPLEMENTED"
    assert data["transcript"] == ""
    assert data["command"]["status"] == "OK"
    assert data["action"] == "unknown"


@pytest.mark.asyncio
async def test_stop_with_zero_chunks_does_not_crash(transport, mock_asr):
    """Calling /stop immediately after /start (no audio at all) must be
    handled gracefully, not crash the server."""
    mock_asr._text = ""
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        session_id = await _start_session(client)
        stop_resp = await client.post(f"/stream/{session_id}/stop")
    assert stop_resp.status_code == 200
    data = stop_resp.json()
    assert data["timing"]["n_chunks"] == 0
    assert data["timing"]["audio_duration_s"] == 0.0
    assert data["action"] == "unknown"
