"""
api.schemas
===========
Pydantic request/response models for all API endpoints.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------

class StatusInfo(BaseModel):
    status: str
    message: str = ""


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = "ok"


# ---------------------------------------------------------------------------
# /system/status
# ---------------------------------------------------------------------------

class ModuleStatus(BaseModel):
    loaded: bool
    status: str
    detail: Optional[str] = None


class SystemStatusResponse(BaseModel):
    voice_separation: ModuleStatus
    keyword_spotting: ModuleStatus
    asr: ModuleStatus
    command: ModuleStatus


# ---------------------------------------------------------------------------
# /separate
# ---------------------------------------------------------------------------

class SeparationResponse(BaseModel):
    num_sources: int
    sample_rate: int            # rate of the returned/output audio (== model_sample_rate)
    original_sample_rate: int   # rate of the uploaded file, before resampling
    model_sample_rate: int      # rate the model expects and ran inference at
    source_shapes: List[List[int]]
    engine_source: str          # "checkpoint" | "torchaudio_bundle"
    model_C: Optional[int]
    outputs: List[str]          # file paths of written WAV files


# ---------------------------------------------------------------------------
# /pipeline
# ---------------------------------------------------------------------------

class PipelineResponse(BaseModel):
    separation: StatusInfo
    keyword_spotting: StatusInfo
    asr: StatusInfo
    command: StatusInfo
    details: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# /voice-command
# ---------------------------------------------------------------------------

class VoiceCommandResponse(BaseModel):
    asr: StatusInfo
    transcript: str
    command: StatusInfo
    action: str
    parameters: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# /stream/* (Stage 3 development-machine audio streaming)
# ---------------------------------------------------------------------------

class StreamStartResponse(BaseModel):
    session_id: str
    sample_rate: int
    created_ts: float


class StreamChunkResponse(BaseModel):
    seq: int
    n_samples: int
    receive_ts: float
    total_duration_s: float


class ChunkTimingInfo(BaseModel):
    seq: int
    n_samples: int
    receive_ts: float
    capture_ts: Optional[float] = None
    send_ts: Optional[float] = None


class StreamTiming(BaseModel):
    session_created_ts: float
    chunks: List[ChunkTimingInfo]
    audio_duration_s: float
    n_chunks: int
    stop_received_ts: float
    asr_start_ts: float
    transcript_ready_ts: float
    command_parsed_ts: float
    asr_latency_ms: float
    command_latency_ms: float
    server_total_ms: float


class StreamStopResponse(BaseModel):
    asr: StatusInfo
    transcript: str
    command: StatusInfo
    action: str
    parameters: Dict[str, Any] = {}
    missing_parameters: List[str] = []
    timing: StreamTiming
