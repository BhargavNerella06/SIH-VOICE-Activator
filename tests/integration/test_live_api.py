"""
tests/integration/test_live_api.py
=====================================
Tests for api/routers/live.py -- the LIVE-mode event feed the frontend
(frontend/code.html) connects to.

Uses FastAPI's synchronous TestClient (no `with TestClient(app) as ...`
context manager), matching tests/integration/test_asr_stream.py's
pattern -- this deliberately does NOT trigger api.main's startup event,
so api.live_session's background mic thread never actually starts during
these tests. That thread's own behaviour (driving VoiceCommandOrchestrator
against a real microphone) isn't testable in CI anyway; these tests cover
the HTTP/WebSocket contract this router exposes to the frontend.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.device_state import reset_state


@pytest.fixture
def app():
    from api.main import app as _app
    return _app


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture(autouse=True)
def _reset_device_state():
    reset_state()
    yield
    reset_state()


def test_live_state_snapshot_shape(client):
    """GET /live/state must work even with the live pipeline thread never
    started (lifespan not triggered in this test), returning defaults
    rather than erroring."""
    response = client.get("/live/state")
    assert response.status_code == 200
    data = response.json()
    assert data["device_state"] == {
        "light": "OFF", "fan": "OFF", "tv": "OFF", "ac": "OFF", "door": "CLOSED",
    }
    assert data["pipeline_status"] == "STOPPED"
    assert data["kws_model"] == "TinyKWSNet"
    assert data["keyword"] == "NOVA"
    # A real checkpoint exists in this repo -- footprint must be a real
    # measured number, never fabricated/omitted.
    assert isinstance(data["model_footprint_kb"], (int, float))
    assert data["model_footprint_kb"] > 0


def test_live_state_reflects_device_changes(client):
    from api.device_state import execute_device
    execute_device("light", "turn_on")

    response = client.get("/live/state")
    assert response.json()["device_state"]["light"] == "ON"


def test_live_ws_sends_snapshot_on_connect(client):
    with client.websocket_connect("/live/ws") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "snapshot"
        assert msg["device_state"] == {
            "light": "OFF", "fan": "OFF", "tv": "OFF", "ac": "OFF", "door": "CLOSED",
        }
        assert msg["pipeline_status"] == "STOPPED"


def test_live_ws_disconnect_is_handled_cleanly(client):
    """Connecting then immediately closing must not raise on the server
    side (the endpoint's WebSocketDisconnect handling)."""
    with client.websocket_connect("/live/ws") as ws:
        ws.receive_json()  # snapshot
    # No exception on exit == clean disconnect handling.
