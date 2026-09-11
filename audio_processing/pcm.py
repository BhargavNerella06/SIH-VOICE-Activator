"""
audio_processing.pcm
=====================
Small, dependency-free helpers for turning a waveform into fixed-duration
PCM16 chunks (for streaming) and back.

Design intent
-------------
This module knows nothing about HTTP, sessions, or ASR -- it is pure
audio-format plumbing shared by both sides of the streaming transport:
the dev-machine client simulator (scripts/stream_demo.py) and the
FastAPI streaming endpoint (api/routers/stream.py). An eventual ESP32
client would use the same on-wire format (raw little-endian int16 PCM)
without needing this module at all -- it's here so the Python side of
both ends agrees on one encoding.

PCM16 (16-bit signed integer PCM) is used on the wire, not float32,
because it is what real microphone/ADC hardware (including a future
ESP32 I2S mic) actually produces, and it's half the bytes of float32.
"""

from __future__ import annotations

from typing import List

import numpy as np

# Chosen to sit inside the 20-40ms low-latency streaming range requested
# for Stage 3. 30ms is a common speech-frame duration (matches most
# classic frame-based DSP/ASR front ends) and divides evenly into whole
# samples at all sample rates used in this repo (8000/16000 Hz).
DEFAULT_CHUNK_MS = 30.0


def chunk_samples(sample_rate: int, chunk_ms: float = DEFAULT_CHUNK_MS) -> int:
    """Number of samples in one chunk of `chunk_ms` milliseconds at `sample_rate`."""
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    if chunk_ms <= 0:
        raise ValueError(f"chunk_ms must be positive, got {chunk_ms}")
    n = int(round(sample_rate * (chunk_ms / 1000.0)))
    return max(1, n)


def frame_waveform(
    waveform: np.ndarray, sample_rate: int, chunk_ms: float = DEFAULT_CHUNK_MS
) -> List[np.ndarray]:
    """
    Split a mono float32 waveform into consecutive fixed-size chunks.

    The final chunk is shorter than the rest if `len(waveform)` isn't an
    exact multiple of the chunk size -- it is NOT zero-padded, so callers
    (and the server-side assembly buffer) must tolerate a short last
    chunk rather than assuming every chunk is the same length. An empty
    waveform yields an empty list.
    """
    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if waveform.size == 0:
        return []
    size = chunk_samples(sample_rate, chunk_ms)
    return [waveform[i:i + size] for i in range(0, len(waveform), size)]


def float32_to_pcm16_bytes(chunk: np.ndarray) -> bytes:
    """Encode a float32 array in [-1, 1] as little-endian signed 16-bit PCM bytes."""
    chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
    clipped = np.clip(chunk, -1.0, 1.0)
    ints = (clipped * 32767.0).astype("<i2")
    return ints.tobytes()


def pcm16_bytes_to_float32(data: bytes) -> np.ndarray:
    """
    Decode little-endian signed 16-bit PCM bytes to a float32 array in [-1, 1].

    Raises
    ------
    ValueError
        If `data` is not a whole number of 16-bit samples (an odd byte
        count) -- this is the "malformed chunk" case a real streaming
        transport must reject rather than silently misinterpret.
    """
    if len(data) % 2 != 0:
        raise ValueError(
            f"Malformed PCM16 chunk: {len(data)} bytes is not a whole number "
            "of 16-bit samples."
        )
    ints = np.frombuffer(data, dtype="<i2")
    return (ints.astype(np.float32) / 32767.0)
