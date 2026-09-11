"""
api.routers.separation
======================
POST /separate  - Upload a WAV, run Conv-TasNet, return separated sources.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from audio_processing.utils import ensure_mono, normalize_audio, resample_if_needed, save_wav_int16
from api.schemas import SeparationResponse
from voice_separation.engine import get_separator_engine

router = APIRouter(prefix="/separate", tags=["separation"])

# Runtime output directory
_OUT_DIR = Path("outputs")
_OUT_DIR.mkdir(exist_ok=True)


@router.post("", response_model=SeparationResponse)
async def separate(
    file: UploadFile = File(..., description="Mono or stereo WAV file"),
    model_path: Optional[str] = Form(
        None,
        description="Path to a local Conv-TasNet checkpoint (.pth.tar). "
                    "Omit to use the torchaudio pretrained bundle.",
    ),
):
    """
    Separate speech sources from the uploaded audio file using Conv-TasNet.

    - If **model_path** is provided, the local checkpoint is used.
    - Otherwise the torchaudio CONVTASNET_BASE_LIBRI2MIX bundle is used
      (auto-downloaded on first call; requires torchaudio).

    Returns paths to the separated WAV files.
    """
    # -- save upload --
    suffix = Path(file.filename or "upload.wav").suffix or ".wav"
    tmp_path = _OUT_DIR / f"upload_{uuid.uuid4().hex[:8]}{suffix}"
    contents = await file.read()
    tmp_path.write_bytes(contents)

    # -- load & preprocess --
    try:
        y, sr = sf.read(str(tmp_path))
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Cannot read audio file: {exc}")

    # mono-convert
    original_sr = sr
    y = ensure_mono(y).astype(np.float32)

    # -- load engine (cached: loaded once per process, reused thereafter) --
    engine = get_separator_engine(
        checkpoint_path=model_path if model_path else None,
        device="cpu",
    )
    if not engine.is_loaded:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=503,
            detail=f"Separation engine failed to load: {engine.status.error}",
        )

    # -- resample to the model's expected sample rate, then normalise --
    model_sr = engine.sample_rate or original_sr
    if original_sr != model_sr:
        y = resample_if_needed(y, original_sr, model_sr)
    y = normalize_audio(y)

    # -- run inference --
    try:
        sources = engine.separate(y, model_sr)
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Separation failed: {exc}")

    # -- write outputs (at the model's sample rate, matching the samples) --
    out_paths = []
    for i, src in enumerate(sources):
        src_norm = normalize_audio(src.astype(np.float32))
        out_path = _OUT_DIR / f"source_{uuid.uuid4().hex[:8]}_{i+1}.wav"
        save_wav_int16(str(out_path), src_norm, model_sr)
        out_paths.append(str(out_path))

    tmp_path.unlink(missing_ok=True)

    return SeparationResponse(
        num_sources=len(sources),
        sample_rate=model_sr,
        original_sample_rate=original_sr,
        model_sample_rate=model_sr,
        source_shapes=[[len(s)] for s in sources],
        engine_source=engine.status.source,
        model_C=engine.status.model_C,
        outputs=out_paths,
    )


@router.get("/download")
async def download(path: str):
    """Download a previously separated WAV file by its path."""
    p = Path(path)
    if not p.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(p), media_type="audio/wav", filename=p.name)
