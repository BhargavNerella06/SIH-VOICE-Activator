"""
keyword_spotting.tinykws_backend
===================================
Thin wrapper around a trained TinyKWSNet checkpoint (produced by
keyword_spotting/training/train.py, default path models/kws_nova_cnn.pt).

Mirrors keyword_spotting.oww_backend.OpenWakeWordBackend's shape on
purpose: both backends expose `.label` and
`.score(window, sample_rate) -> float`, so keyword_spotting.detector.
KeywordDetector's inference/debounce logic (_infer_window,
_apply_debounce, detect_chunk, detect) needs no branching and no
changes at all to support this backend -- it is entirely
backend-agnostic already.

Feature-configuration honesty
------------------------------
The checkpoint is the single source of truth for n_mels, sample_rate,
and window_seconds: this backend reads them from the checkpoint and
exposes them as attributes so keyword_spotting.detector can adopt them
for its own windowing (rather than the caller's constructor defaults
possibly disagreeing with what the model was actually trained on --
see keyword_spotting.detector.KeywordDetector.__init__). Nothing here
hard-codes a feature-extraction value that could silently diverge from
training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from audio_processing.utils import ensure_mono, resample_if_needed
from .features import extract_features
from .model import TinyKWSNet

REQUIRED_CHECKPOINT_KEYS = ("state_dict", "n_mels", "sample_rate", "window_seconds", "keyword")

# Label convention fixed by keyword_spotting.model.TinyKWSNet's own
# docstring: class index 0 = unknown/negative, class index 1 = NOVA.
_NOVA_CLASS_INDEX = 1


class TinyKWSNetBackend:
    """
    Loads one trained TinyKWSNet checkpoint and scores fixed-length
    audio windows against it.

    Parameters
    ----------
    checkpoint_path : Path
        Path to a checkpoint saved by keyword_spotting.training.train
        (a torch.save'd dict with the keys in REQUIRED_CHECKPOINT_KEYS).
    device : str
        torch device to load and run the model on (e.g. "cpu").

    Raises
    ------
    ValueError
        If the checkpoint is missing required metadata, was trained for
        a different keyword, or its state_dict does not match
        TinyKWSNet's architecture for the checkpoint's own declared
        n_mels (a malformed/incompatible checkpoint).
    """

    def __init__(self, checkpoint_path: Path, device: str = "cpu") -> None:
        checkpoint_path = Path(checkpoint_path)
        package: Any = torch.load(str(checkpoint_path), map_location=device)

        if not isinstance(package, dict):
            raise ValueError(
                f"TinyKWSNet checkpoint {checkpoint_path} is not a dict "
                f"(got {type(package).__name__}); expected the format saved by "
                "keyword_spotting.training.train."
            )
        missing = [k for k in REQUIRED_CHECKPOINT_KEYS if k not in package]
        if missing:
            raise ValueError(
                f"TinyKWSNet checkpoint {checkpoint_path} is missing required "
                f"key(s) {missing}; expected the format saved by "
                "keyword_spotting.training.train."
            )

        keyword = package["keyword"]
        if keyword != "NOVA":
            raise ValueError(
                f"TinyKWSNet checkpoint {checkpoint_path} was trained for "
                f"keyword {keyword!r}, not 'NOVA'."
            )

        self.n_mels = int(package["n_mels"])
        self.sample_rate = int(package["sample_rate"])
        self.window_seconds = float(package["window_seconds"])
        self.keyword = keyword
        self.label = keyword  # parity with OpenWakeWordBackend.label

        model = TinyKWSNet(n_mels=self.n_mels, num_classes=2)
        try:
            model.load_state_dict(package["state_dict"])
        except Exception as exc:
            raise ValueError(
                f"TinyKWSNet checkpoint {checkpoint_path}'s state_dict is "
                f"incompatible with TinyKWSNet(n_mels={self.n_mels}): {exc}"
            ) from exc
        model.to(device)
        model.eval()
        self._model = model
        self._device = device

    def score(self, window: np.ndarray, sample_rate: int) -> float:
        """
        Score one independent, self-contained audio window and return
        the NOVA probability in [0, 1] (softmax over TinyKWSNet's two
        output classes).

        Parameters
        ----------
        window : np.ndarray
            Mono audio, nominal range [-1, 1].
        sample_rate : int
            Sample rate of `window`. Resampled to this backend's
            checkpoint-declared sample rate if different.
        """
        window = ensure_mono(np.asarray(window, dtype=np.float32).reshape(-1))
        if sample_rate != self.sample_rate:
            window = resample_if_needed(window, sample_rate, self.sample_rate)

        feat = extract_features(
            window,
            sample_rate=self.sample_rate,
            window_seconds=self.window_seconds,
            n_mels=self.n_mels,
        )
        x = torch.from_numpy(feat).unsqueeze(0).unsqueeze(0).to(self._device)  # [1, 1, n_mels, T]
        with torch.no_grad():
            logits = self._model(x)
            probs = torch.softmax(logits, dim=-1)
        return float(probs[0, _NOVA_CLASS_INDEX].item())
