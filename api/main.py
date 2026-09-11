"""
api.main
========
FastAPI application entry point for the SIH Voice Activator.

Server : uvicorn api.main:app --reload --port 8000
Docs   : http://localhost:8000/docs
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.routers import separation, pipeline, status, voice_command, stream, asr_stream, live
from api.schemas import HealthResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


app = FastAPI(
    title="SIH Voice Activator — Milestone 1",
    description=(
        "Low-latency voice activation prototype for SIH 2026 PS-26172. "
        "Milestone 1: laptop demonstration with Conv-TasNet voice separation."
    ),
    version="0.1.0",
)

# Allow all origins for local development / demo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(separation.router)
app.include_router(pipeline.router)
app.include_router(status.router)
app.include_router(voice_command.router)
app.include_router(stream.router)
app.include_router(asr_stream.router)
app.include_router(live.router)


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health() -> HealthResponse:
    """Liveness check — always returns 200 OK if the server is running."""
    return HealthResponse(status="ok")


_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
_FRONTEND_PATH = _FRONTEND_DIR / "code.html"

# Serves frontend/assets/room-reference.jpg (the room-photo background the
# UI's device HUD cards are overlaid on) at /assets/... -- code.html loads
# it via a plain relative <img src="assets/room-reference.jpg">.
app.mount("/assets", StaticFiles(directory=str(_FRONTEND_DIR / "assets")), name="assets")


@app.get("/", include_in_schema=False)
async def frontend_index() -> FileResponse:
    """Serve the NOVA smart-room UI (frontend/code.html) so the whole demo
    is one process: `uvicorn api.main:app --port 8000`, then open
    http://localhost:8000/ full-screen -- see frontend/code.html's own
    WebSocket client for how it connects back to /live/ws same-origin."""
    return FileResponse(str(_FRONTEND_PATH))


@app.on_event("startup")
async def on_startup() -> None:
    logger.info("SIH Voice Activator API starting up (Milestone 1)")

    # Warm the voice-separation model cache now, so the first real request
    # doesn't pay the load cost. get_separator_engine() is cached, so every
    # router that calls it afterwards reuses this exact instance.
    from voice_separation.engine import get_separator_engine

    engine = get_separator_engine()
    if engine.is_loaded:
        logger.info(
            "Voice separation model preloaded at startup: source=%s, model_C=%s",
            engine.status.source, engine.status.model_C,
        )
    else:
        logger.error(
            "Voice separation model failed to preload at startup: %s",
            engine.status.error,
        )

    # Warm the ASR (faster-whisper) model cache the same way. If this fails
    # (e.g. no network on first use to fetch model weights), the app still
    # starts -- asr.transcriber.Transcriber degrades to NOT_IMPLEMENTED
    # rather than raising, matching keyword_spotting's graceful-degradation
    # pattern.
    from asr.transcriber import get_transcriber

    transcriber = get_transcriber()
    if transcriber.is_ready:
        logger.info(
            "ASR model preloaded at startup: model=%s, device=%s",
            transcriber.model_name, transcriber.device,
        )
    else:
        logger.warning(
            "ASR model failed to preload at startup: %s", transcriber.load_error,
        )

    # Start the LIVE-mode background pipeline (mic -> KWS -> ASR -> command
    # -> device state) and its WebSocket broadcast loop. Degrades to an
    # OFFLINE status pushed to clients (see api.live_session) rather than
    # failing startup if no microphone is present -- DEMO mode in
    # frontend/code.html does not depend on this at all.
    from api.live_session import get_live_session
    from api.routers.live import start_broadcast_loop

    get_live_session().start()
    start_broadcast_loop()
    logger.info("LIVE-mode pipeline thread and WebSocket broadcast loop started")


@app.on_event("shutdown")
async def on_shutdown() -> None:
    from api.live_session import get_live_session
    from api.routers.live import stop_broadcast_loop

    stop_broadcast_loop()
    get_live_session().stop()
    logger.info("LIVE-mode pipeline thread and WebSocket broadcast loop stopped")
