"""
api.routers.status
==================
GET /system/status  - Reports the load state of every pipeline module.
"""

from fastapi import APIRouter
from api.schemas import ModuleStatus, SystemStatusResponse

router = APIRouter(prefix="/system", tags=["system"])


@router.get("/status", response_model=SystemStatusResponse)
async def system_status():
    """
    Return the current load status of every pipeline module.

    voice_separation  : loaded via torchaudio bundle or checkpoint.
    keyword_spotting  : real Stage 2.2 prototype (NOVA); OK if a trained
                        checkpoint is present, NOT_IMPLEMENTED otherwise.
    asr               : real Stage 3 local faster-whisper transcriber
                        (dev-machine only); OK if the model loaded,
                        NOT_IMPLEMENTED if it failed to (e.g. no network
                        on first use to download model weights).
    command           : real Stage 3 rule-based intent parser; always OK
                        (stateless, no model to load).
    """
    # -- voice separation --
    from voice_separation.engine import get_separator_engine
    engine = get_separator_engine()   # cached; uses torchaudio bundle by default
    if engine.is_loaded:
        vs_status = ModuleStatus(
            loaded=True,
            status="OK",
            detail=f"source={engine.status.source}, model_C={engine.status.model_C}",
        )
    else:
        vs_status = ModuleStatus(
            loaded=False,
            status="ERROR",
            detail=engine.status.error,
        )

    # -- keyword spotting --
    from keyword_spotting.detector import get_keyword_detector
    detector = get_keyword_detector()   # cached; loads its checkpoint at most once
    if detector.is_ready:
        kws_status = ModuleStatus(
            loaded=True,
            status="OK",
            detail=f"keyword={detector.keywords[0]}, threshold={detector.threshold}",
        )
    else:
        kws_status = ModuleStatus(
            loaded=False,
            status="NOT_IMPLEMENTED",
            detail=detector.load_error,
        )

    # -- ASR --
    from asr.transcriber import get_transcriber
    transcriber = get_transcriber()   # cached; loads its model at most once
    if transcriber.is_ready:
        asr_status = ModuleStatus(
            loaded=True,
            status="OK",
            detail=f"model={transcriber.model_name}, device={transcriber.device}",
        )
    else:
        asr_status = ModuleStatus(
            loaded=False,
            status="NOT_IMPLEMENTED",
            detail=transcriber.load_error,
        )

    # -- command interpreter (stateless rule-based parser; always ready) --
    cmd_status = ModuleStatus(
        loaded=True,
        status="OK",
        detail="Rule-based intent parser (no model to load).",
    )

    return SystemStatusResponse(
        voice_separation=vs_status,
        keyword_spotting=kws_status,
        asr=asr_status,
        command=cmd_status,
    )
