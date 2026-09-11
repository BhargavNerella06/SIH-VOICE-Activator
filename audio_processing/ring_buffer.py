"""
audio_processing.ring_buffer
==============================
AudioRingBuffer - a fixed-capacity circular buffer of the most recent
audio samples.

Purpose (Stage 3 streaming architecture)
------------------------------------------
A wake-word detector always has some reaction latency: by the time it
fires, the first fraction of a second of the user's utterance has
already passed through the audio pipeline. A "pre-roll" buffer is the
standard fix -- a small ring buffer that continuously retains the last
N milliseconds of audio *regardless of whether a wake word has been
detected yet*, so that once a detection event exists, that trailing
window can be retrieved and prepended to what comes after, recovering
the audio that would otherwise be clipped.

This module implements only the generic circular-buffer mechanism. It
is deliberately independent of keyword_spotting/ -- it does not import
it, call it, or know what "NOVA" is. Per the Stage 3 task scope, wiring
this buffer to a real detection event is future work; today it is fed
continuously by the streaming session (api/streaming.py) purely to
demonstrate the architecture slot it will eventually fill.
"""

from __future__ import annotations

import threading

import numpy as np


class AudioRingBuffer:
    """
    Fixed-capacity circular buffer of float32 mono audio samples.

    Parameters
    ----------
    capacity_seconds : float
        How much trailing audio to retain, in seconds.
    sample_rate : int
        Sample rate of the audio being written, in Hz.

    Thread-safety: write()/read()/clear() are guarded by an internal
    lock, since a streaming session may be fed from one async request
    handler while another concurrently inspects it.
    """

    def __init__(self, capacity_seconds: float, sample_rate: int) -> None:
        if capacity_seconds <= 0:
            raise ValueError(f"capacity_seconds must be positive, got {capacity_seconds}")
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")

        self.capacity_seconds = capacity_seconds
        self.sample_rate = sample_rate
        self._capacity_samples = max(1, int(round(capacity_seconds * sample_rate)))
        self._buffer = np.zeros(self._capacity_samples, dtype=np.float32)
        self._write_pos = 0     # next write index, mod capacity
        self._filled = 0        # how many valid samples are currently stored (<= capacity)
        self._lock = threading.Lock()

    @property
    def capacity_samples(self) -> int:
        return self._capacity_samples

    @property
    def filled_samples(self) -> int:
        with self._lock:
            return self._filled

    @property
    def duration_seconds(self) -> float:
        """How much audio is currently retained (<= capacity_seconds)."""
        return self.filled_samples / float(self.sample_rate)

    @property
    def is_full(self) -> bool:
        with self._lock:
            return self._filled >= self._capacity_samples

    def write(self, chunk: np.ndarray) -> None:
        """
        Append a chunk of audio, overwriting the oldest samples once the
        buffer is at capacity. A chunk longer than the buffer's own
        capacity is accepted -- only its most recent `capacity_samples`
        samples end up retained, exactly as if it had been written
        sample-by-sample.
        """
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if chunk.size == 0:
            return

        with self._lock:
            cap = self._capacity_samples
            if chunk.size >= cap:
                # Only the tail end fits; it fully overwrites the buffer.
                self._buffer[:] = chunk[-cap:]
                self._write_pos = 0
                self._filled = cap
                return

            end = self._write_pos + chunk.size
            if end <= cap:
                self._buffer[self._write_pos:end] = chunk
            else:
                first_part = cap - self._write_pos
                self._buffer[self._write_pos:] = chunk[:first_part]
                self._buffer[:end - cap] = chunk[first_part:]
            self._write_pos = end % cap
            self._filled = min(cap, self._filled + chunk.size)

    def read(self) -> np.ndarray:
        """Return a copy of the currently retained audio, oldest sample first."""
        with self._lock:
            if self._filled < self._capacity_samples:
                # Not full yet: samples [0, filled) are in order already
                # (writes started at index 0 and haven't wrapped).
                return self._buffer[:self._filled].copy()
            # Full and possibly wrapped: oldest sample is at _write_pos.
            return np.concatenate(
                (self._buffer[self._write_pos:], self._buffer[:self._write_pos])
            ).copy()

    def clear(self) -> None:
        with self._lock:
            self._buffer[:] = 0.0
            self._write_pos = 0
            self._filled = 0
