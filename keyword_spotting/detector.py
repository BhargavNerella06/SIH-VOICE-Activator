"""
keyword_spotting.detector
===========================
KeywordDetector - Stage 2.2 real KWS prototype for the wake word "NOVA".

Pipeline: audio -> a backend (TinyKWSNet by default -- see below;
openWakeWord remains supported) -> per-window wake-word probability ->
threshold + consecutive-frame debounce -> DETECTED / NO_DETECTION.

Independent of voice_separation: only generic, shared audio utilities
(mono/resample) are reused here -- Conv-TasNet is never imported or
invoked by this module.

If no trained checkpoint is available, the detector loads in a
"not ready" state and reports status="NOT_IMPLEMENTED" with
confidence=None rather than fabricating a result.

Backend selection (by `model_path` file extension)
------------------------------------------------------
- ".pt" / ".pth"  -> keyword_spotting.tinykws_backend.TinyKWSNetBackend
  (TinyKWSNet, the PRIMARY training path -- see
  keyword_spotting/training/train.py, default output
  models/kws_nova_cnn.pt). This is also DEFAULT_MODEL_PATH's format.
- ".onnx"         -> keyword_spotting.oww_backend.OpenWakeWordBackend
  (openWakeWord, kept fully working as optional detector
  infrastructure -- pass an explicit model_path=".../something.onnx"
  to use it; it is no longer the default).

Both backends expose the same `.score(window, sample_rate) -> float`
and `.label` shape, so everything below this point (_infer_window,
_apply_debounce, detect_chunk, detect) is backend-agnostic and
required no changes to support TinyKWSNet.

A backend that carries its own training-time feature configuration
(currently only TinyKWSNetBackend) has that configuration -- n_mels,
sample_rate, window_seconds -- adopted by this detector after loading,
overriding whatever the constructor's own sample_rate/window_seconds
arguments were. This is deliberate: inference must use the exact
feature configuration training used, never a possibly-mismatched
default (see _load_model).
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from audio_processing.utils import ensure_mono, resample_if_needed
from .features import DEFAULT_SAMPLE_RATE
from .oww_backend import OpenWakeWordBackend
from .tinykws_backend import TinyKWSNetBackend

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "kws_nova_cnn.pt"
DEFAULT_WINDOW_SECONDS = 1.0
DEFAULT_HOP_SECONDS = 0.2
DEFAULT_THRESHOLD = 0.85
DEFAULT_CONSECUTIVE_FRAMES = 3

SUPPORTED_KEYWORDS = ["NOVA"]

_ONNX_SUFFIXES = (".onnx",)
_TINYKWS_SUFFIXES = (".pt", ".pth")


@dataclass
class KeywordResult:
    status: str                        # "DETECTED" | "NO_DETECTION" | "NOT_IMPLEMENTED"
    keyword: Optional[str] = None
    confidence: Optional[float] = None
    latency_ms: Optional[float] = None
    message: str = ""

    def to_dict(self) -> Dict[str, object]:
        """Return the exact minimal JSON shape required by Stage 2.2."""
        d: Dict[str, object] = {"status": self.status}
        if self.status == "DETECTED":
            d["keyword"] = self.keyword
            d["confidence"] = self.confidence
            d["latency_ms"] = self.latency_ms
        return d


class KeywordDetector:
    """
    Real-time-capable keyword spotter for the wake word "NOVA".

    Parameters
    ----------
    keywords : list[str] | None
        Target wake word(s). This prototype only supports ["NOVA"];
        anything else raises ValueError.
    sample_rate : int
        Sample rate to use if the loaded backend doesn't declare its
        own (i.e. the openWakeWord backend). Ignored -- overridden by
        the checkpoint's own value -- for the TinyKWSNet backend; see
        the module docstring.
    window_seconds, hop_seconds : float
        Sliding-window analysis parameters: the model looks at the last
        `window_seconds` of audio, and a new inference runs every time
        `hop_seconds` worth of new audio has arrived. `window_seconds`
        is likewise overridden by a TinyKWSNet checkpoint's own value;
        `hop_seconds` is a detector-level inference-cadence choice, not
        a training parameter, and is never overridden by either backend.
    threshold : float
        Minimum per-window softmax probability for "NOVA" to count as a
        positive frame.
    consecutive_frames : int
        Number of consecutive positive frames required before declaring
        DETECTED. This is the anti-false-trigger debounce: a single
        high-probability frame is not enough on its own.
    model_path : str | Path | None
        Path to a trained model checkpoint. Backend is selected by file
        extension -- see the module docstring. Defaults to
        models/kws_nova_cnn.pt (TinyKWSNet, the primary training path).
    device : str
        torch device used to load/run the TinyKWSNet backend (e.g.
        "cpu"). Unused by the openWakeWord backend (inference always
        runs via onnxruntime's CPUExecutionProvider).
    """

    def __init__(
        self,
        keywords: Optional[List[str]] = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        window_seconds: float = DEFAULT_WINDOW_SECONDS,
        hop_seconds: float = DEFAULT_HOP_SECONDS,
        threshold: float = DEFAULT_THRESHOLD,
        consecutive_frames: int = DEFAULT_CONSECUTIVE_FRAMES,
        model_path: Optional[str] = None,
        device: str = "cpu",
    ) -> None:
        self.keywords = keywords or list(SUPPORTED_KEYWORDS)
        if list(self.keywords) != SUPPORTED_KEYWORDS:
            raise ValueError(
                f"This KWS prototype only supports {SUPPORTED_KEYWORDS}, got {self.keywords}"
            )
        self.sample_rate = sample_rate
        self.window_seconds = window_seconds
        self.hop_seconds = hop_seconds
        self.threshold = threshold
        self.consecutive_frames = consecutive_frames
        self.device = device
        self.n_mels: Optional[int] = None  # only meaningful for the TinyKWSNet backend

        self._model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self._model: Optional[object] = None
        self.load_error: Optional[str] = None
        # May override self.sample_rate / self.window_seconds / self.n_mels
        # from the checkpoint's own metadata -- see _load_model's docstring.
        self._load_model()

        self._window_samples = max(1, int(round(self.sample_rate * self.window_seconds)))
        self._hop_samples = max(1, int(round(self.sample_rate * self.hop_seconds)))

        # Streaming state
        self._buffer: Deque[float] = deque(maxlen=self._window_samples)
        self._samples_since_last_infer = 0
        self._consecutive_hits = 0

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def _load_model(self) -> None:
        if not self._model_path.exists():
            self.load_error = (
                f"No trained KWS model found at {self._model_path}. "
                "Train one with keyword_spotting/training/train.py (TinyKWSNet, "
                "the primary path -- produces a '.pt' checkpoint like "
                "models/kws_nova_cnn.pt) or provide an openWakeWord custom "
                "wake-word '.onnx' model via model_path (see "
                "keyword_spotting/oww_backend.py)."
            )
            logger.warning(self.load_error)
            return

        suffix = self._model_path.suffix.lower()
        try:
            if suffix in _TINYKWS_SUFFIXES:
                backend = TinyKWSNetBackend(self._model_path, device=self.device)
                self._model = backend
                # The checkpoint is the source of truth for the feature
                # configuration training actually used -- adopt it rather
                # than risk silently running inference with a mismatched
                # sample_rate/window_seconds/n_mels (see module docstring).
                self.sample_rate = backend.sample_rate
                self.window_seconds = backend.window_seconds
                self.n_mels = backend.n_mels
                logger.info(
                    "Loaded TinyKWSNet checkpoint from %s (n_mels=%d, "
                    "sample_rate=%d, window_seconds=%.2f)",
                    self._model_path, backend.n_mels, backend.sample_rate,
                    backend.window_seconds,
                )
            elif suffix in _ONNX_SUFFIXES:
                self._model = OpenWakeWordBackend(self._model_path)
                logger.info(
                    "Loaded openWakeWord model from %s (label=%s)",
                    self._model_path, self._model.label,
                )
            else:
                raise ValueError(
                    f"Unrecognized KWS model file extension {suffix!r} for "
                    f"{self._model_path} (expected one of "
                    f"{_TINYKWS_SUFFIXES + _ONNX_SUFFIXES})."
                )
        except Exception as exc:
            self.load_error = f"Failed to load KWS model {self._model_path}: {exc}"
            logger.error(self.load_error)

    # ------------------------------------------------------------------
    # Core inference on one fixed-length window
    # ------------------------------------------------------------------

    def _infer_window(self, window: np.ndarray) -> Tuple[float, float]:
        """Run the model on one window. Returns (nova_probability, latency_ms)."""
        t0 = time.perf_counter()
        nova_prob = self._model.score(window, sample_rate=self.sample_rate)  # type: ignore[union-attr]
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return nova_prob, latency_ms

    def _not_ready_result(self) -> KeywordResult:
        return KeywordResult(
            status="NOT_IMPLEMENTED",
            message=self.load_error or "KWS model not loaded.",
        )

    def _apply_debounce(self, prob: float, latency_ms: float) -> KeywordResult:
        if prob >= self.threshold:
            self._consecutive_hits += 1
        else:
            self._consecutive_hits = 0

        if self._consecutive_hits >= self.consecutive_frames:
            self._consecutive_hits = 0  # avoid immediately re-firing on the next frame
            return KeywordResult(
                status="DETECTED",
                keyword="NOVA",
                confidence=prob,
                latency_ms=latency_ms,
                message=f"Wake word 'NOVA' detected (confidence={prob:.3f}).",
            )
        return KeywordResult(
            status="NO_DETECTION",
            latency_ms=latency_ms,
            message="No wake word detected in this window.",
        )

    # ------------------------------------------------------------------
    # Streaming interface
    # ------------------------------------------------------------------

    def reset_stream(self) -> None:
        """Clear streaming state. Call when starting a new listening session."""
        self._buffer.clear()
        self._samples_since_last_infer = 0
        self._consecutive_hits = 0

    def detect_chunk(self, chunk: np.ndarray) -> Optional[KeywordResult]:
        """
        Feed one streaming audio chunk (e.g. from MicCapture.read_chunk()).

        Returns
        -------
        KeywordResult or None
            None if the model isn't ready to run an inference yet
            (buffer not full, or the next analysis hop hasn't been
            reached). Otherwise a KeywordResult with status DETECTED
            or NO_DETECTION. Returns immediately with status
            "NOT_IMPLEMENTED" if no trained model is loaded.
        """
        if not self.is_ready:
            return self._not_ready_result()

        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        self._buffer.extend(chunk.tolist())
        self._samples_since_last_infer += len(chunk)

        if len(self._buffer) < self._window_samples:
            return None
        if self._samples_since_last_infer < self._hop_samples:
            return None

        self._samples_since_last_infer = 0
        window = np.array(self._buffer, dtype=np.float32)
        prob, latency_ms = self._infer_window(window)
        return self._apply_debounce(prob, latency_ms)

    # ------------------------------------------------------------------
    # One-shot interface (whole waveform, e.g. /pipeline)
    # ------------------------------------------------------------------

    def detect(self, waveform: np.ndarray, sample_rate: int) -> KeywordResult:
        """
        Analyse a complete waveform by sliding the same window/hop used
        for streaming. Returns DETECTED if the debounce condition fires
        anywhere in the clip, else NO_DETECTION.
        """
        if not self.is_ready:
            return self._not_ready_result()

        waveform = ensure_mono(np.asarray(waveform, dtype=np.float32))
        if sample_rate != self.sample_rate:
            waveform = resample_if_needed(waveform, sample_rate, self.sample_rate)

        self.reset_stream()
        last_result: Optional[KeywordResult] = None
        step = self._hop_samples
        for start in range(0, len(waveform), step):
            chunk = waveform[start:start + step]
            if len(chunk) == 0:
                break
            result = self.detect_chunk(chunk)
            if result is not None:
                if result.status == "DETECTED":
                    return result
                last_result = result

        return last_result or KeywordResult(status="NO_DETECTION", message="No wake word detected.")


# ---------------------------------------------------------------------------
# Process-wide cache (mirrors voice_separation.engine.get_separator_engine)
# ---------------------------------------------------------------------------

_detector_cache: Dict[tuple, "KeywordDetector"] = {}
_detector_cache_lock = threading.Lock()


def get_keyword_detector(
    keywords: Optional[List[str]] = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    hop_seconds: float = DEFAULT_HOP_SECONDS,
    threshold: float = DEFAULT_THRESHOLD,
    consecutive_frames: int = DEFAULT_CONSECUTIVE_FRAMES,
    model_path: Optional[str] = None,
    device: str = "cpu",
) -> "KeywordDetector":
    """Return a cached KeywordDetector, loading its model at most once per config."""
    key = (
        tuple(keywords or SUPPORTED_KEYWORDS), sample_rate, window_seconds,
        hop_seconds, threshold, consecutive_frames, model_path, device,
    )
    detector = _detector_cache.get(key)
    if detector is not None:
        return detector
    with _detector_cache_lock:
        detector = _detector_cache.get(key)
        if detector is None:
            detector = KeywordDetector(
                keywords=keywords, sample_rate=sample_rate, window_seconds=window_seconds,
                hop_seconds=hop_seconds, threshold=threshold,
                consecutive_frames=consecutive_frames, model_path=model_path, device=device,
            )
            _detector_cache[key] = detector
    return detector


def clear_detector_cache() -> None:
    """Drop all cached detectors. Mainly useful for tests."""
    with _detector_cache_lock:
        _detector_cache.clear()
