"""
voice_pipeline - end-to-end laptop-demo orchestration:
microphone chunks -> wake detection -> AudioSessionManager -> ASR ->
CommandInterpreter. See voice_pipeline.orchestrator for the full
interface, mock-vs-real wake mode design, and honesty notes.
"""
from .orchestrator import (
    MockWakeTrigger,
    PipelineResult,
    PipelineTimings,
    VoiceCommandOrchestrator,
    WakeMode,
)
from .remote_asr_client import RemoteASRClient, RemoteASRResult, RemoteASRTimings

__all__ = [
    "VoiceCommandOrchestrator",
    "WakeMode",
    "MockWakeTrigger",
    "PipelineResult",
    "PipelineTimings",
    "RemoteASRClient",
    "RemoteASRResult",
    "RemoteASRTimings",
]
