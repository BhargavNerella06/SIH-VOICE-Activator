"""
voice_separation.engine
=======================
SeparatorEngine - thin inference adapter around the original Conv-TasNet.

Design constraints (Milestone 1)
---------------------------------
- Does NOT change model architecture, hyperparameters, or behaviour.
- Does NOT perform speaker-count auto-detection.
- The model has a fixed output head (C speakers as trained; default C=2).
- Always runs the full C-speaker separation pass and returns all C sources.
- No sys.path manipulation; uses package-relative imports.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .conv_tasnet import ConvTasNet
from .utils import remove_pad

logger = logging.getLogger(__name__)


@dataclass
class EngineStatus:
    """Describes the current load state of SeparatorEngine."""
    loaded: bool = False
    source: str = "none"          # "checkpoint" | "torchaudio_bundle" | "none"
    model_C: Optional[int] = None  # number of output speakers (from model)
    checkpoint_path: Optional[str] = None
    error: Optional[str] = None


class SeparatorEngine:
    """
    Wraps Conv-TasNet for single-call inference.

    Usage
    -----
    # From a trained checkpoint:
    engine = SeparatorEngine(checkpoint_path="models/final.pth.tar")

    # From torchaudio pretrained bundle (no checkpoint needed):
    engine = SeparatorEngine()

    # Run separation:
    sources = engine.separate(waveform_np, sample_rate)
    # sources: List[np.ndarray], one array per speaker, shape (T,)
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "cpu",
    ) -> None:
        self._device = device
        self._model: Optional[torch.nn.Module] = None
        self._status = EngineStatus()

        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)
        else:
            self._load_torchaudio_bundle()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_checkpoint(self, path: str) -> None:
        """Load a ConvTasNet checkpoint saved by solver.py."""
        p = Path(path)
        if not p.exists():
            self._status.error = f"Checkpoint not found: {path}"
            logger.error(self._status.error)
            return
        try:
            package = torch.load(str(p), map_location=self._device)
            model = ConvTasNet.load_model_from_package(package)
            model.to(self._device)
            model.eval()
            self._model = model
            self._status = EngineStatus(
                loaded=True,
                source="checkpoint",
                model_C=int(model.C),
                checkpoint_path=str(p),
            )
            logger.info(
                "Loaded Conv-TasNet checkpoint: %s (C=%d)", p, model.C
            )
        except Exception as exc:
            self._status.error = f"Failed to load checkpoint {path}: {exc}"
            logger.error(self._status.error)

    def _load_torchaudio_bundle(self) -> None:
        """Load the pretrained torchaudio CONVTASNET_BASE_LIBRI2MIX bundle."""
        try:
            import torchaudio  # noqa: F401
            from torchaudio.pipelines import CONVTASNET_BASE_LIBRI2MIX

            bundle = CONVTASNET_BASE_LIBRI2MIX
            model = bundle.get_model().to(self._device)
            model.eval()
            self._model = model
            self._bundle_sr: int = bundle.sample_rate
            self._status = EngineStatus(
                loaded=True,
                source="torchaudio_bundle",
                # torchaudio bundle is a 2-speaker model
                model_C=2,
            )
            logger.info(
                "Loaded torchaudio CONVTASNET_BASE_LIBRI2MIX (sr=%d)",
                self._bundle_sr,
            )
        except ModuleNotFoundError:
            self._status.error = (
                "torchaudio is not installed and no checkpoint_path was given. "
                "Install torchaudio or provide a checkpoint."
            )
            logger.error(self._status.error)
        except Exception as exc:
            self._status.error = f"Failed to load torchaudio bundle: {exc}"
            logger.error(self._status.error)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def status(self) -> EngineStatus:
        return self._status

    @property
    def is_loaded(self) -> bool:
        return self._status.loaded

    @property
    def sample_rate(self) -> Optional[int]:
        """
        Sample rate this engine expects its input waveform to be at.

        For the torchaudio bundle this comes from the bundle itself. For a
        local checkpoint the original training package does not store the
        sample rate, so we fall back to this repo's training default
        (8000 Hz, per src/train.py / egs/wsj0/run.sh upstream).
        """
        if not self.is_loaded:
            return None
        if self._status.source == "torchaudio_bundle":
            return self._bundle_sr
        return 8000

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def separate(
        self, waveform: np.ndarray, sample_rate: int
    ) -> List[np.ndarray]:
        """
        Run Conv-TasNet separation on a mono waveform.

        Parameters
        ----------
        waveform : np.ndarray, shape (T,), dtype float32
            Mono audio samples, already normalised.
        sample_rate : int
            Sample rate of the waveform.  The model is used as-is; if
            the sample rate does not match the training rate the caller
            is responsible for resampling before calling this method.

        Returns
        -------
        List[np.ndarray]
            One array per separated source, shape (T,).
            Number of sources == model.C (fixed at training time).

        Raises
        ------
        RuntimeError
            If the engine is not loaded.
        """
        if not self.is_loaded or self._model is None:
            raise RuntimeError(
                f"SeparatorEngine is not loaded. Status: {self._status.error}"
            )

        waveform = waveform.astype(np.float32)

        if self._status.source == "torchaudio_bundle":
            return self._run_torchaudio(waveform, sample_rate)
        else:
            return self._run_conv_tasnet(waveform, sample_rate)

    def _run_conv_tasnet(
        self, waveform: np.ndarray, sample_rate: int
    ) -> List[np.ndarray]:
        """Inference path for a locally-loaded ConvTasNet checkpoint."""
        x = torch.from_numpy(waveform).unsqueeze(0).to(self._device)  # [1, T]
        with torch.no_grad():
            # model.forward returns [M, C, T]; M=1 here
            est = self._model(x)  # type: ignore[operator]
        est_np: np.ndarray = est.squeeze(0).cpu().numpy()  # (C, T)
        return [est_np[i] for i in range(est_np.shape[0])]

    def _run_torchaudio(
        self, waveform: np.ndarray, sample_rate: int
    ) -> List[np.ndarray]:
        """Inference path for the torchaudio pretrained bundle."""
        # Bundle expects [1, 1, T]
        x = (
            torch.from_numpy(waveform)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self._device)
        )
        with torch.no_grad():
            est = self._model(x)  # type: ignore[operator]
        est_np: np.ndarray = est.squeeze(0).cpu().numpy()
        if est_np.ndim == 3:
            est_np = est_np[:, 0, :]
        return [est_np[i] for i in range(est_np.shape[0])]


# ---------------------------------------------------------------------------
# Process-wide cache
# ---------------------------------------------------------------------------
#
# The API layer previously constructed a brand-new SeparatorEngine (and thus
# reloaded/redownloaded the model) on every single request. That's wasteful
# and slow. SeparatorEngine instances are cached here, keyed by the
# (checkpoint_path, device) they were constructed with, so the underlying
# torch.nn.Module is loaded exactly once per process and reused afterwards.

_engine_cache: Dict[Tuple[Optional[str], str], "SeparatorEngine"] = {}
_engine_cache_lock = threading.Lock()


def get_separator_engine(
    checkpoint_path: Optional[str] = None,
    device: str = "cpu",
) -> "SeparatorEngine":
    """
    Return a cached SeparatorEngine for (checkpoint_path, device), creating
    and loading it on first use only.

    Safe to call concurrently from multiple requests: a lock guards the
    (rare) model-construction path, while the common cache-hit path avoids
    locking entirely.
    """
    key = (checkpoint_path, device)
    engine = _engine_cache.get(key)
    if engine is not None:
        return engine

    with _engine_cache_lock:
        # Re-check inside the lock in case another thread just built it.
        engine = _engine_cache.get(key)
        if engine is None:
            engine = SeparatorEngine(checkpoint_path=checkpoint_path, device=device)
            _engine_cache[key] = engine
    return engine


def clear_engine_cache() -> None:
    """Drop all cached engines. Mainly useful for tests."""
    with _engine_cache_lock:
        _engine_cache.clear()


def peek_cached_engine(
    checkpoint_path: Optional[str] = None,
    device: str = "cpu",
) -> Optional["SeparatorEngine"]:
    """
    Return the engine cached for (checkpoint_path, device), or None if it
    has not been loaded yet. Unlike get_separator_engine(), never triggers
    a load. Mainly useful for tests that assert *when* loading happened.
    """
    return _engine_cache.get((checkpoint_path, device))
