"""
asr.transcriber
===============
Transcriber - functional local ASR backend (Stage 3).

Status: FUNCTIONAL (development-machine only)
----------------------------------------------
Uses faster-whisper (CTranslate2 port of OpenAI Whisper) to run real,
local, offline speech-to-text. No cloud API is called at inference
time -- everything after the one-time model download runs on-device.

Scope / honesty notes (read before trusting anything from this module)
------------------------------------------------------------------------
- This targets a development machine (laptop-class CPU), not an
  edge/MCU device. The final product's ASR is intended to run on a
  more capable remote/server component, not on the same constrained
  hardware as keyword_spotting -- this module is a dev-machine stand-in
  for that remote service, not a claim that ASR runs on the edge.
- The model is generic pretrained Whisper (tiny.en by default). It has
  NOT been fine-tuned on NOVA or on this project's data in any way.
  No accuracy claim is made or implied for any specific vocabulary.
- The first call for a given model_name downloads model weights from
  Hugging Face Hub (a few tens of MB for "tiny.en") and caches them
  locally (~/.cache/huggingface). Without network access on first use,
  loading fails and this module reports status="NOT_IMPLEMENTED" with
  the underlying error -- it never fabricates a transcription.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from audio_processing.utils import ensure_mono, resample_if_needed

logger = logging.getLogger(__name__)

# faster-whisper / Whisper's native sample rate. Input at any other rate
# is resampled to this before transcription.
WHISPER_SAMPLE_RATE = 16000

DEFAULT_MODEL_NAME = "tiny.en"
DEFAULT_DEVICE = "cpu"
DEFAULT_COMPUTE_TYPE = "int8"


@dataclass
class TranscriptionResult:
    status: str          # "OK" | "NOT_IMPLEMENTED" | "ERROR"
    text: str = ""
    language: Optional[str] = None
    latency_ms: Optional[float] = None
    message: str = ""


class Transcriber:
    """
    Local speech-to-text via faster-whisper.

    Parameters
    ----------
    model_name : str
        A faster-whisper/Whisper model size, e.g. "tiny.en", "base.en".
        Defaults to "tiny.en" -- smallest/fastest, adequate for short
        command-style utterances on a dev machine.
    device : str
        "cpu" (default) or "cuda" if a GPU + CUDA build is available.
    compute_type : str
        CTranslate2 compute type, e.g. "int8" (default, fastest on CPU)
        or "float32".

    If the faster-whisper package is missing, or the model fails to
    load (e.g. no network on first use), the instance loads in a
    "not ready" state and every transcribe() call returns
    status="NOT_IMPLEMENTED" with the reason in `message` -- mirroring
    keyword_spotting.detector.KeywordDetector's graceful-degradation
    pattern. It never returns a fabricated transcript.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = DEFAULT_DEVICE,
        compute_type: str = DEFAULT_COMPUTE_TYPE,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self._model = None
        self.load_error: Optional[str] = None
        self._load_model()

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def _load_model(self) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            self.load_error = (
                f"faster-whisper is not installed ({exc}). "
                "Install it with `pip install faster-whisper`."
            )
            logger.warning(self.load_error)
            return
        try:
            self._model = WhisperModel(
                self.model_name, device=self.device, compute_type=self.compute_type
            )
            logger.info(
                "Loaded faster-whisper model '%s' (device=%s, compute_type=%s)",
                self.model_name, self.device, self.compute_type,
            )
        except Exception as exc:
            self.load_error = (
                f"Failed to load faster-whisper model '{self.model_name}': {exc}"
            )
            logger.error(self.load_error)

    def _not_ready_result(self) -> TranscriptionResult:
        return TranscriptionResult(
            status="NOT_IMPLEMENTED",
            message=self.load_error or "ASR model not loaded.",
        )

    def transcribe(
        self, waveform: np.ndarray, sample_rate: int
    ) -> TranscriptionResult:
        """
        Transcribe a complete mono waveform to text.

        Parameters
        ----------
        waveform : np.ndarray
            Audio samples, any float dtype/range; resampled to 16 kHz
            mono before transcription.
        sample_rate : int
            Sample rate of `waveform`.
        """
        if not self.is_ready:
            return self._not_ready_result()

        import time

        try:
            audio = ensure_mono(np.asarray(waveform, dtype=np.float32))
            if sample_rate != WHISPER_SAMPLE_RATE:
                audio = resample_if_needed(audio, sample_rate, WHISPER_SAMPLE_RATE)

            t0 = time.perf_counter()
            segments, info = self._model.transcribe(audio, beam_size=1)
            segments = list(segments)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            text = " ".join(seg.text.strip() for seg in segments).strip()
            return TranscriptionResult(
                status="OK",
                text=text,
                language=info.language,
                latency_ms=latency_ms,
                message=(
                    "Transcribed via faster-whisper "
                    f"({self.model_name}, dev-machine CPU inference)."
                ),
            )
        except Exception as exc:
            logger.error("Transcription failed: %s", exc)
            return TranscriptionResult(status="ERROR", message=f"Transcription failed: {exc}")


# ---------------------------------------------------------------------------
# Process-wide cache (mirrors voice_separation.engine.get_separator_engine)
# ---------------------------------------------------------------------------

_transcriber_cache: Dict[Tuple[str, str, str], "Transcriber"] = {}
_transcriber_cache_lock = threading.Lock()


def get_transcriber(
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = DEFAULT_DEVICE,
    compute_type: str = DEFAULT_COMPUTE_TYPE,
) -> "Transcriber":
    """Return a cached Transcriber, loading its model at most once per config."""
    key = (model_name, device, compute_type)
    transcriber = _transcriber_cache.get(key)
    if transcriber is not None:
        return transcriber
    with _transcriber_cache_lock:
        transcriber = _transcriber_cache.get(key)
        if transcriber is None:
            transcriber = Transcriber(
                model_name=model_name, device=device, compute_type=compute_type
            )
            _transcriber_cache[key] = transcriber
    return transcriber


def clear_transcriber_cache() -> None:
    """Drop all cached transcribers. Mainly useful for tests."""
    with _transcriber_cache_lock:
        _transcriber_cache.clear()


class TranscriptionError(RuntimeError):
    """Raised by transcribe() when ASR is unavailable or transcription fails.

    Callers that need to distinguish "not implemented" (no model loaded)
    from "error" (transcription itself failed) without exceptions should
    use get_transcriber().transcribe() directly, which returns a
    TranscriptionResult with a `status` field instead of raising.
    """


def transcribe(
    audio: np.ndarray,
    sample_rate: int,
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = DEFAULT_DEVICE,
    compute_type: str = DEFAULT_COMPUTE_TYPE,
) -> str:
    """
    Simple functional ASR interface: waveform + sample rate in, plain
    transcript text out.

    Uses the process-wide cached Transcriber (see get_transcriber()), so
    the model is loaded at most once regardless of how many times this
    is called.

    Raises
    ------
    TranscriptionError
        If the ASR model isn't available (e.g. faster-whisper not
        installed, or no network on first use to fetch model weights)
        or if transcription itself fails (e.g. malformed audio). The
        underlying reason is included in the exception message.
    """
    result = get_transcriber(
        model_name=model_name, device=device, compute_type=compute_type
    ).transcribe(audio, sample_rate=sample_rate)
    if result.status != "OK":
        raise TranscriptionError(
            result.message or f"ASR failed with status {result.status!r}."
        )
    return result.text


def peek_cached_transcriber(
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = DEFAULT_DEVICE,
    compute_type: str = DEFAULT_COMPUTE_TYPE,
) -> Optional["Transcriber"]:
    """Return the cached transcriber for this config, or None if not loaded yet."""
    return _transcriber_cache.get((model_name, device, compute_type))
