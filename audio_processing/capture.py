"""
audio_processing.capture
=========================
MicCapture - real-time, streaming microphone capture using sounddevice.

Stage 2 / Milestone 2.1: chunked capture suitable for a downstream
keyword-spotting (KWS) stage.

Design constraints
-------------------
- No FastAPI / web-framework dependency of any kind, so this module can
  run standalone directly on an edge device, independent of the API
  service.
- Audio I/O happens on PortAudio's own background callback thread
  (managed internally by sounddevice); nothing here blocks a caller
  (e.g. an asyncio event loop) for the duration of a recording. The
  caller pulls chunks at its own pace via read_chunk().
- sounddevice is imported lazily inside methods, exactly as in
  Milestone 1, so importing this module never requires the audio
  backend to be installed.
"""

from __future__ import annotations

import logging
import queue
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Defaults chosen to suit a downstream KWS stage: 16 kHz mono is the
# standard input rate for common on-device keyword spotters (e.g.
# Porcupine, most MFCC/CNN keyword models), and 512 samples (~32 ms
# @ 16 kHz) matches Porcupine's native frame length while remaining a
# reasonable frame size for other frame-based KWS approaches.
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHUNK_SIZE = 512


class MicCaptureError(RuntimeError):
    """Raised when the microphone/device cannot be opened or fails during capture."""


class MicCapture:
    """
    Streaming microphone capture.

    Usage
    -----
        mic = MicCapture(sample_rate=16000, chunk_size=512)
        mic.start()
        try:
            chunk = mic.read_chunk()   # np.ndarray, shape (chunk_size,), float32
        finally:
            mic.stop()

    Or as a context manager::

        with MicCapture() as mic:
            chunk = mic.read_chunk()

    Parameters
    ----------
    sample_rate : int
        Capture sample rate in Hz. Default 16000 (standard KWS input rate).
    chunk_size : int
        Number of samples per chunk. Default 512 (~32 ms @ 16 kHz).
    channels : int
        Number of input channels to open. Captured audio is always
        downmixed to a single (mono) channel in read_chunk() output.
    device : int | str | None
        sounddevice device index or name. None uses the default input
        device.
    queue_maxsize : int
        Maximum number of pending chunks buffered before the oldest is
        dropped, bounding memory use if the consumer falls behind
        real time.
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        channels: int = 1,
        device: Optional[object] = None,
        queue_maxsize: int = 50,
    ) -> None:
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.device = device
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=queue_maxsize)
        self._stream = None
        self._running = False

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # Streaming interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open the input stream and begin capturing chunks in the background."""
        if self._running:
            return

        try:
            import sounddevice as sd  # type: ignore
        except ImportError as exc:
            raise MicCaptureError(
                "sounddevice is required for microphone capture. "
                "Install it with: pip install sounddevice"
            ) from exc

        def _callback(indata, frames, time_info, status) -> None:
            if status:
                logger.warning("MicCapture stream status: %s", status)
            mono = indata[:, 0] if getattr(indata, "ndim", 1) == 2 else indata
            chunk = np.asarray(mono, dtype=np.float32).copy()
            try:
                self._queue.put_nowait(chunk)
            except queue.Full:
                # Consumer is falling behind real time: drop the oldest
                # chunk rather than growing memory or blocking the audio
                # callback thread.
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._queue.put_nowait(chunk)
                except queue.Full:
                    pass

        try:
            stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                channels=self.channels,
                dtype="float32",
                device=self.device,
                callback=_callback,
            )
            stream.start()
        except Exception as exc:
            raise MicCaptureError(f"Failed to open microphone input stream: {exc}") from exc

        self._stream = stream
        self._running = True
        logger.info(
            "MicCapture started: sample_rate=%d chunk_size=%d channels=%d",
            self.sample_rate, self.chunk_size, self.channels,
        )

    def drain(self) -> None:
        """
        Discard any chunks buffered while nothing was consuming them (e.g.
        during an inter-clip pause between recordings).

        The stream keeps capturing in the background between read_chunk()
        calls, so without this, the next read_chunk() calls return stale
        audio from *before* the caller was ready rather than live audio
        from the moment capture actually starts.
        """
        with self._queue.mutex:
            self._queue.queue.clear()

    def read_chunk(self, timeout: Optional[float] = 5.0) -> np.ndarray:
        """
        Block until the next audio chunk is available and return it.

        Parameters
        ----------
        timeout : float or None
            Maximum seconds to wait for a chunk. None waits forever.

        Returns
        -------
        np.ndarray
            Mono float32 array of shape (chunk_size,).

        Raises
        ------
        MicCaptureError
            If capture has not been started (or was stopped), or no
            chunk arrives within `timeout` seconds.
        """
        if not self._running:
            raise MicCaptureError("MicCapture is not running; call start() first.")
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty as exc:
            raise MicCaptureError(
                f"No audio chunk received within {timeout}s "
                "(stream may have stalled or the device disconnected)."
            ) from exc

    def stop(self) -> None:
        """Stop and close the input stream. Safe to call multiple times."""
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as exc:
                logger.warning("Error while closing microphone stream: %s", exc)
            self._stream = None
        self._running = False
        with self._queue.mutex:
            self._queue.queue.clear()
        logger.info("MicCapture stopped")

    def __enter__(self) -> "MicCapture":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Backward-compatible one-shot recording (Milestone 1 behaviour)
    # ------------------------------------------------------------------

    def record(self, duration_seconds: float) -> np.ndarray:
        """
        Record for a fixed duration and return a float32 mono array.

        Kept for backward compatibility with Milestone 1 callers.
        Prefer start()/read_chunk()/stop() for streaming use cases such
        as keyword spotting.

        Raises
        ------
        MicCaptureError
            If sounddevice is not installed or the device fails.
        """
        try:
            import sounddevice as sd  # type: ignore
        except ImportError as exc:
            raise MicCaptureError(
                "sounddevice is required for microphone capture. "
                "Install it with: pip install sounddevice"
            ) from exc

        frames = int(duration_seconds * self.sample_rate)
        try:
            audio = sd.rec(
                frames,
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                device=self.device,
            )
            sd.wait()
        except Exception as exc:
            raise MicCaptureError(f"Failed to record from microphone: {exc}") from exc

        if audio.ndim == 2:
            audio = audio[:, 0]
        return audio.astype(np.float32)
