"""
keyword_spotting.oww_backend
===============================
Thin wrapper around openWakeWord's Model class. Isolates every
openWakeWord-specific detail (model loading, frame framing, PCM
conversion, warm-up handling) behind one small class, so
keyword_spotting.detector stays backend-agnostic and its public
streaming interface (detect_chunk() -> KeywordResult) is unaffected by
which model backend is behind it.

Verified against the installed openwakeword==0.6.0 source
(site-packages/openwakeword/model.py, .../utils.py) before writing
this file -- nothing here is guessed:

- Model.predict() zeroes its output for the first WARMUP_CALLS calls
  made since the last reset() (openwakeword/model.py: "Zero
  predictions for first 5 frames during model initialization", keyed
  on `len(self.prediction_buffer[cls]) < 5`, i.e. a *call-count* guard,
  not a duration guard). Calling reset() and then a single predict()
  on a whole multi-frame window would therefore always return 0.0,
  even for a real detection. To score one independent window
  correctly, reset() once, then feed the window as a sequence of
  individual 80 ms (1280-sample) predict() calls so the warm-up is
  absorbed inside that same window, and take the max score across all
  calls (the warm-up calls only ever contribute 0.0, so they cannot
  hide a real positive score later in the window).
- reset() itself DOES exist (openwakeword/model.py Model.reset) --
  confirmed rather than assumed, per the implementation plan.
- The shared melspectrogram/embedding backbone (the frozen, generic
  part) is not bundled in the PyPI wheel; download_models() fetches it
  from GitHub release assets on first use and caches it under
  openwakeword's own resources/models directory. download_models()
  always fetches melspec+embedding+VAD regardless of its model_names
  filter, and only conditionally fetches the *named* pretrained
  wake-word models (alexa/hey_jarvis/hey_mycroft/...). We pass a
  filter that can never match one of those names, so only the shared
  backbone is downloaded -- we never load or repurpose another
  vendor's wake word.
- A custom wakeword model's predictions dict key is the ONNX file's
  stem (see Model.__init__: wakeword_model_names derived from
  os.path.splitext(os.path.basename(path))[0]), so the label is read
  back from the loaded Model rather than hard-coded here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np

from audio_processing.utils import resample_if_needed

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16000
FRAME_SAMPLES = 1280  # 80 ms @ 16 kHz -- openWakeWord's native frame size

# A model_names filter guaranteed not to match any of openWakeWord's
# bundled pretrained wake words (alexa, hey_jarvis, hey_mycroft,
# hey_rhasspy, timer, weather) -- see download_models() in
# openwakeword/utils.py: melspec/embedding/VAD are always fetched
# regardless of this list; only matching *named* wakeword models are.
_SHARED_FEATURES_ONLY_FILTER = ["__shared_features_only__"]


class OpenWakeWordBackend:
    """
    Loads one trained openWakeWord custom wake-word model (frozen
    shared embedding + a small trained classifier head) and scores
    fixed-length audio windows against it.
    """

    def __init__(self, model_path: Path) -> None:
        from openwakeword.utils import download_models
        from openwakeword.model import Model

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"openWakeWord model not found at {model_path}")

        # Fetches the shared melspectrogram/embedding backbone (and the
        # unrelated Silero VAD model, harmlessly) on first run only;
        # no-ops on every subsequent run since download_models() skips
        # files that already exist on disk.
        download_models(model_names=list(_SHARED_FEATURES_ONLY_FILTER))

        self._model = Model(
            wakeword_models=[str(model_path)],
            inference_framework="onnx",
        )
        labels: List[str] = list(self._model.models.keys())
        if len(labels) != 1:
            raise ValueError(
                f"Expected exactly one wake-word model loaded from {model_path}, "
                f"found {labels!r}"
            )
        self._label = labels[0]

    @property
    def label(self) -> str:
        return self._label

    def score(self, window: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> float:
        """
        Score one independent, self-contained audio window and return
        the wake-word probability in [0, 1].

        Parameters
        ----------
        window : np.ndarray
            Mono float32 audio, nominal range [-1, 1].
        sample_rate : int
            Sample rate of `window`. Resampled to 16 kHz if different.
        """
        window = np.asarray(window, dtype=np.float32).reshape(-1)
        if sample_rate != TARGET_SAMPLE_RATE:
            window = resample_if_needed(window, sample_rate, TARGET_SAMPLE_RATE)

        pcm16 = (np.clip(window, -1.0, 1.0) * 32767.0).astype(np.int16)
        n_frames = len(pcm16) // FRAME_SAMPLES
        if n_frames == 0:
            return 0.0

        self._model.reset()
        best = 0.0
        for i in range(n_frames):
            frame = pcm16[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES]
            preds = self._model.predict(frame)
            score = float(preds.get(self._label, 0.0))
            if score > best:
                best = score
        return best
