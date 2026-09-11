"""
api.routers.pipeline
=====================
POST /pipeline  - End-to-end voice activation pipeline.

Current behaviour
-----------------
1. Separation  : runs Conv-TasNet on the uploaded audio.
2. KWS         : real Stage 2.2 "NOVA" prototype (reports NOT_IMPLEMENTED
                 if no trained checkpoint is present locally).
3. ASR         : real Stage 3 local transcription via faster-whisper
                 (dev-machine only; reports NOT_IMPLEMENTED if the
                 model failed to load, e.g. no network on first use).
4. Command     : real Stage 3 rule-based intent parser (see
                 command/interpreter.py). Always "OK"; action="unknown"
                 when no rule matches.

Each stage clearly reports its status; no fake results are generated.

Known limitation (ASR input source selection)
----------------------------------------------
Conv-TasNet's pretrained 2-speaker bundle always emits a fixed number of
output channels, regardless of how many real speakers are in the input.
For genuinely single-speaker dev-machine audio there is no principled
way to know in advance which channel will carry the (arbitrarily
ordered) speech, so ASR here is fed the separated channel with the
highest RMS energy -- a simple heuristic, not a guarantee. On
single-speaker input, separation can still distort or misattribute
content across channels (it is designed for overlapping-speaker
mixtures, not clean single-speaker audio), which can degrade
transcription quality independent of the ASR/command stages themselves.
See api/routers/voice_command.py for a minimal ASR -> command demo path
that bypasses separation entirely.
"""

from __future__ import annotations

import uuid
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from audio_processing.preprocess import AudioPreprocessor
from voice_separation.engine import get_separator_engine
from keyword_spotting.detector import get_keyword_detector
from asr.transcriber import get_transcriber
from command.interpreter import CommandInterpreter
from api.schemas import PipelineResponse, StatusInfo

router = APIRouter(prefix="/pipeline", tags=["pipeline"])

_OUT_DIR = Path("outputs")
_OUT_DIR.mkdir(exist_ok=True)


def _pick_dominant_source(sources: List[np.ndarray]) -> np.ndarray:
    """Pick the separated channel with the highest RMS energy to feed ASR.

    See the module docstring's "Known limitation" section: this is a
    documented heuristic for picking among Conv-TasNet's fixed-count
    output channels, not a guarantee that the chosen channel contains
    clean or correctly-separated speech.
    """
    if not sources:
        return np.zeros(1, dtype=np.float32)
    return max(sources, key=lambda s: float(np.sqrt(np.mean(np.square(s)))))


@router.post("", response_model=PipelineResponse)
async def pipeline(
    file: UploadFile = File(..., description="Audio WAV file"),
    model_path: Optional[str] = Form(
        None,
        description="Optional path to a Conv-TasNet checkpoint.",
    ),
):
    """
    Run the full voice activation pipeline: separation -> KWS -> ASR ->
    command. Separation, KWS, ASR, and command are all real (Stage 3);
    KWS and ASR each honestly report NOT_IMPLEMENTED if their model
    isn't available locally, rather than fabricating a result.
    """
    # -- save upload --
    suffix = Path(file.filename or "upload.wav").suffix or ".wav"
    tmp_path = _OUT_DIR / f"pipeline_upload_{uuid.uuid4().hex[:8]}{suffix}"
    contents = await file.read()
    tmp_path.write_bytes(contents)

    details: dict = {}

    # ------------------------------------------------------------------ #
    # Stage 1: Audio preprocessing                                        #
    # ------------------------------------------------------------------ #
    try:
        preprocessor = AudioPreprocessor(target_sr=8000)
        waveform, sr = preprocessor.load_file(str(tmp_path))
        details["input_duration_s"] = round(len(waveform) / sr, 3)
        details["sample_rate"] = sr
        sep_status = StatusInfo(status="OK")
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Preprocessing failed: {exc}")

    # ------------------------------------------------------------------ #
    # Stage 2: Voice separation (Conv-TasNet)                             #
    # ------------------------------------------------------------------ #
    engine = get_separator_engine(
        checkpoint_path=model_path if model_path else None,
        device="cpu",
    )
    if not engine.is_loaded:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=503,
            detail=f"Separation engine not loaded: {engine.status.error}",
        )
    try:
        sources = engine.separate(waveform, sr)
        details["num_sources"] = len(sources)
        details["engine_source"] = engine.status.source
        details["model_C"] = engine.status.model_C
        sep_status = StatusInfo(
            status="OK",
            message=f"Separated into {len(sources)} source(s) via {engine.status.source}.",
        )
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Separation error: {exc}")

    tmp_path.unlink(missing_ok=True)

    # ------------------------------------------------------------------ #
    # Stage 3: Keyword Spotting (real Stage 2.2 "NOVA" prototype)         #
    # ------------------------------------------------------------------ #
    kws_result = get_keyword_detector().detect(
        waveform=sources[0] if sources else np.zeros(1, dtype=np.float32),
        sample_rate=sr,
    )
    kws_status = StatusInfo(
        status=kws_result.status,
        message=kws_result.message,
    )

    # ------------------------------------------------------------------ #
    # Stage 4: ASR (real, local, dev-machine faster-whisper)              #
    # ------------------------------------------------------------------ #
    asr_result = get_transcriber().transcribe(
        waveform=_pick_dominant_source(sources),
        sample_rate=sr,
    )
    asr_status = StatusInfo(
        status=asr_result.status,
        message=asr_result.message,
    )
    details["asr_text"] = asr_result.text

    # ------------------------------------------------------------------ #
    # Stage 5: Command interpretation (real, rule-based)                  #
    # ------------------------------------------------------------------ #
    cmd_result = CommandInterpreter().interpret(text=asr_result.text)
    cmd_status = StatusInfo(
        status=cmd_result.status,
        message=cmd_result.message,
    )
    details["command_action"] = cmd_result.action
    details["command_parameters"] = cmd_result.parameters

    return PipelineResponse(
        separation=sep_status,
        keyword_spotting=kws_status,
        asr=asr_status,
        command=cmd_status,
        details=details,
    )
