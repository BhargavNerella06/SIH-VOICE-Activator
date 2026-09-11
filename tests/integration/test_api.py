"""
tests/integration/test_api.py
==============================
Required test 5: FastAPI application starts successfully.
Tests /health and /system/status endpoints.
"""

import io
import struct
import wave

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient


# ---------------------------------------------------------------------------
# App fixture
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def app():
    from api.main import app as _app
    return _app


@pytest.fixture(scope="module")
def transport(app):
    return ASGITransport(app=app)


# ---------------------------------------------------------------------------
# Required test 5: FastAPI application starts successfully
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_app_starts_and_health(transport):
    """/health must return 200 with {"status": "ok"}."""
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


# ---------------------------------------------------------------------------
# /system/status
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_system_status_keys(transport):
    """/system/status must return all four module statuses."""
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/system/status")
    assert response.status_code == 200
    data = response.json()
    assert "voice_separation" in data
    assert "keyword_spotting" in data
    assert "asr" in data
    assert "command" in data


@pytest.mark.asyncio
async def test_system_status_command_is_always_ready(transport):
    """command is a stateless rule-based parser (Stage 3): no model to
    load, so it must always report OK."""
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/system/status")
    data = response.json()
    assert data["command"]["status"] == "OK"
    assert data["command"]["loaded"] is True


@pytest.mark.asyncio
async def test_system_status_asr_reports_real_state(transport):
    """
    ASR is a real Stage 3 local faster-whisper backend now: status must
    be OK if the model loaded (the common case once weights are cached
    locally), or an honest NOT_IMPLEMENTED (never a fake OK) if it
    didn't -- either way, no other status is acceptable.
    """
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/system/status")
    data = response.json()
    assert data["asr"]["status"] in ("OK", "NOT_IMPLEMENTED")


@pytest.mark.asyncio
async def test_system_status_keyword_spotting_reports_real_state(transport):
    """
    KWS is a real Stage 2.2 prototype now: status must be OK if a trained
    checkpoint exists locally, or an honest NOT_IMPLEMENTED (never a fake
    OK) if it doesn't -- either way, no other status is acceptable.
    """
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/system/status")
    data = response.json()
    assert data["keyword_spotting"]["status"] in ("OK", "NOT_IMPLEMENTED")


# ---------------------------------------------------------------------------
# /separate - minimal WAV upload (requires torchaudio)
# ---------------------------------------------------------------------------
def _make_wav_bytes(duration_s: float = 0.5, sr: int = 8000) -> bytes:
    """Generate a minimal silent WAV file in memory."""
    import numpy as np
    n = int(duration_s * sr)
    samples = (np.random.randn(n) * 0.1).astype(np.float32)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((samples * 32767).astype("<i2").tobytes())
    buf.seek(0)
    return buf.read()


@pytest.mark.asyncio
async def test_separate_endpoint_returns_200(transport):
    """POST /separate with a valid WAV must return 200 (needs torchaudio)."""
    pytest.importorskip("torchaudio", reason="torchaudio not installed")
    wav_bytes = _make_wav_bytes()
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/separate",
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
        )
    assert response.status_code == 200, response.text
    data = response.json()
    assert "num_sources" in data
    assert data["num_sources"] >= 1


def test_model_preloaded_at_app_startup():
    """
    The separation model must be loaded during FastAPI startup, before any
    request is handled -- not lazily on first use.
    """
    pytest.importorskip("torchaudio", reason="torchaudio not installed")
    from fastapi.testclient import TestClient
    from api.main import app as _app
    from voice_separation.engine import clear_engine_cache, peek_cached_engine

    clear_engine_cache()
    assert peek_cached_engine() is None, "engine must not be loaded before startup"

    with TestClient(_app):
        # The `with` block runs the ASGI lifespan startup event; no HTTP
        # request has been issued yet at this point.
        cached = peek_cached_engine()
        assert cached is not None, "engine must be cached by the startup event"
        assert cached.is_loaded

    clear_engine_cache()


def test_asr_model_preloaded_at_app_startup():
    """
    The ASR (faster-whisper) model must be warmed during FastAPI startup,
    same as voice separation -- not lazily on first /pipeline request.
    Skipped if faster-whisper isn't installed in this environment.
    """
    pytest.importorskip("faster_whisper", reason="faster-whisper not installed")
    from fastapi.testclient import TestClient
    from api.main import app as _app
    from asr.transcriber import clear_transcriber_cache, peek_cached_transcriber

    clear_transcriber_cache()
    assert peek_cached_transcriber() is None, "transcriber must not be loaded before startup"

    with TestClient(_app):
        cached = peek_cached_transcriber()
        assert cached is not None, "transcriber must be cached by the startup event"

    clear_transcriber_cache()


@pytest.mark.asyncio
async def test_separate_endpoint_8k_audio_metadata(transport):
    """8kHz upload: original and model sample rate must both report 8000."""
    pytest.importorskip("torchaudio", reason="torchaudio not installed")
    wav_bytes = _make_wav_bytes(duration_s=0.5, sr=8000)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/separate",
            files={"file": ("test8k.wav", wav_bytes, "audio/wav")},
        )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["original_sample_rate"] == 8000
    assert data["model_sample_rate"] == 8000
    assert data["sample_rate"] == 8000
    assert data["num_sources"] >= 1


@pytest.mark.asyncio
async def test_separate_endpoint_resamples_non_8k_audio(transport):
    """Uploading 16kHz audio must be resampled to the model's 8kHz rate."""
    pytest.importorskip("torchaudio", reason="torchaudio not installed")
    wav_bytes = _make_wav_bytes(duration_s=0.5, sr=16000)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/separate",
            files={"file": ("test16k.wav", wav_bytes, "audio/wav")},
        )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["original_sample_rate"] == 16000
    assert data["model_sample_rate"] == 8000
    assert data["sample_rate"] == 8000
    assert data["num_sources"] >= 1
    assert len(data["outputs"]) == data["num_sources"]


@pytest.mark.asyncio
async def test_separate_engine_not_reloaded_across_requests(transport, monkeypatch):
    """The separation engine must be cached, not reconstructed on every request."""
    pytest.importorskip("torchaudio", reason="torchaudio not installed")
    import voice_separation.engine as engine_mod

    engine_mod.clear_engine_cache()
    call_count = {"n": 0}
    original_init = engine_mod.SeparatorEngine.__init__

    def counting_init(self, *args, **kwargs):
        call_count["n"] += 1
        return original_init(self, *args, **kwargs)

    monkeypatch.setattr(engine_mod.SeparatorEngine, "__init__", counting_init)

    wav_bytes = _make_wav_bytes()
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        for _ in range(3):
            response = await client.post(
                "/separate",
                files={"file": ("test.wav", wav_bytes, "audio/wav")},
            )
            assert response.status_code == 200

    assert call_count["n"] == 1, (
        f"SeparatorEngine.__init__ ran {call_count['n']} times across 3 requests; "
        "expected exactly once due to caching"
    )
    engine_mod.clear_engine_cache()


@pytest.mark.asyncio
async def test_pipeline_endpoint_returns_200(transport):
    """
    POST /pipeline must return 200 with separation OK. KWS and ASR are
    real prototypes whose status legitimately depends on local model
    availability (a trained NOVA checkpoint / downloaded Whisper
    weights) -- OK or an honest NOT_IMPLEMENTED are both acceptable for
    those two, but neither may fabricate a result. command is a
    stateless rule-based parser with nothing to load, so it must
    always report OK.
    """
    pytest.importorskip("torchaudio", reason="torchaudio not installed")
    wav_bytes = _make_wav_bytes()
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/pipeline",
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
        )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["separation"]["status"] == "OK"
    assert data["keyword_spotting"]["status"] in ("DETECTED", "NO_DETECTION", "NOT_IMPLEMENTED")
    assert data["asr"]["status"] in ("OK", "NOT_IMPLEMENTED")
    assert data["command"]["status"] == "OK"
    assert "asr_text" in data["details"]
    assert "command_action" in data["details"]


# ---------------------------------------------------------------------------
# /voice-command - minimal audio -> ASR -> command pipeline (Stage 3)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_voice_command_endpoint_returns_200_on_silence(transport):
    """
    /voice-command must return 200 and a well-formed response even on
    silent input (no speech to transcribe). ASR legitimately depends on
    local model availability -- OK or an honest NOT_IMPLEMENTED are both
    acceptable, but command must always report OK (stateless parser).
    """
    wav_bytes = _make_wav_bytes(duration_s=1.0, sr=16000)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/voice-command",
            files={"file": ("silence.wav", wav_bytes, "audio/wav")},
        )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["asr"]["status"] in ("OK", "NOT_IMPLEMENTED")
    assert data["command"]["status"] == "OK"
    assert isinstance(data["transcript"], str)
    assert isinstance(data["action"], str)


def _synthesize_speech_wav(text: str) -> bytes:
    """Real speech via Windows SAPI TTS (dev-machine only helper)."""
    import os
    import tempfile

    import win32com.client as wc

    speaker = wc.Dispatch("SAPI.SpVoice")
    with tempfile.TemporaryDirectory() as tmp:
        raw_path = os.path.join(tmp, "speech.wav")
        stream = wc.Dispatch("SAPI.SpFileStream")
        stream.Open(raw_path, 3, False)  # 3 == SSFMCreateForWrite
        speaker.AudioOutputStream = stream
        speaker.Speak(text)
        stream.Close()
        with open(raw_path, "rb") as f:
            return f.read()


@pytest.mark.asyncio
async def test_voice_command_endpoint_transcribes_real_speech(transport):
    """
    End-to-end with real synthesized speech (not silence/noise): a clear
    "turn on the lights" utterance must be transcribed and matched to the
    turn_on/lights intent. Skipped where SAPI TTS (Windows-only) or
    faster-whisper aren't available.
    """
    pytest.importorskip("win32com.client", reason="Windows SAPI not available")
    pytest.importorskip("faster_whisper", reason="faster-whisper not installed")

    from asr.transcriber import get_transcriber
    if not get_transcriber().is_ready:
        pytest.skip(f"ASR model not available: {get_transcriber().load_error}")

    wav_bytes = _synthesize_speech_wav("turn on the lights")
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/voice-command",
            files={"file": ("speech.wav", wav_bytes, "audio/wav")},
        )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["asr"]["status"] == "OK"
    assert "turn on" in data["transcript"].lower()
    assert data["action"] == "turn_on"
    assert data["parameters"].get("device") == "lights"
