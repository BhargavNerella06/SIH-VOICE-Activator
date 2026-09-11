"""
api.routers.voice_command
===========================
POST /voice-command  - Minimal audio -> ASR -> command-interpreter pipeline.

This is the direct Stage 3 deliverable: audio in, transcript + structured
action out, with no voice-separation or keyword-spotting stage in
between. It exists alongside /pipeline (which chains separation -> KWS ->
ASR -> command) because Conv-TasNet's pretrained 2-speaker separation
model is designed for overlapping-speaker mixtures, not clean
single-speaker dev-machine audio -- routing this endpoint's audio through
it first would not improve ASR quality and could distort it (see
api/routers/pipeline.py's "Known limitation" note). Use /pipeline to
exercise the full multi-stage chain; use this endpoint to exercise ASR +
command in isolation.

Both stages are real:
- ASR      : local faster-whisper (see asr/transcriber.py). Dev-machine
             only; NOT_IMPLEMENTED if the model failed to load.
- command  : rule-based intent parser (see command/interpreter.py).
             Always OK; action="unknown" when no rule matches.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from audio_processing.preprocess import AudioPreprocessor
from asr.transcriber import get_transcriber
from command.interpreter import CommandInterpreter
from api.schemas import StatusInfo, VoiceCommandResponse

router = APIRouter(prefix="/voice-command", tags=["voice-command"])

_OUT_DIR = Path("outputs")
_OUT_DIR.mkdir(exist_ok=True)

# faster-whisper's native sample rate; see asr/transcriber.py.
_ASR_SAMPLE_RATE = 16000


@router.post("", response_model=VoiceCommandResponse)
async def voice_command(
    file: UploadFile = File(..., description="Audio WAV file (any sample rate)"),
):
    """
    Transcribe the uploaded audio and interpret it as a structured
    command/action. No separation or keyword-spotting stage is applied.
    """
    suffix = Path(file.filename or "upload.wav").suffix or ".wav"
    tmp_path = _OUT_DIR / f"voice_command_{uuid.uuid4().hex[:8]}{suffix}"
    contents = await file.read()
    tmp_path.write_bytes(contents)

    try:
        preprocessor = AudioPreprocessor(target_sr=_ASR_SAMPLE_RATE)
        waveform, sr = preprocessor.load_file(str(tmp_path))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Preprocessing failed: {exc}")
    finally:
        tmp_path.unlink(missing_ok=True)

    asr_result = get_transcriber().transcribe(waveform, sample_rate=sr)
    cmd_result = CommandInterpreter().interpret(asr_result.text)

    return VoiceCommandResponse(
        asr=StatusInfo(status=asr_result.status, message=asr_result.message),
        transcript=asr_result.text,
        command=StatusInfo(status=cmd_result.status, message=cmd_result.message),
        action=cmd_result.action,
        parameters=cmd_result.parameters,
    )
