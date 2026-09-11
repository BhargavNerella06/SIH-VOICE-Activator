"""
keyword_spotting
=================
Stage 2.2: a real, working keyword-spotting prototype for the wake
word "NOVA".

KeywordDetector (this package's actual runtime path, used by
api/routers/pipeline.py, api/routers/stream.py, api/routers/asr_stream.py,
and voice_pipeline.orchestrator) supports two backends, selected
automatically by the `model_path` file extension (see
keyword_spotting/detector.py for the exact dispatch logic):

- TinyKWSNet (keyword_spotting/model.py, wrapped by
  keyword_spotting/tinykws_backend.py) -- a small, from-scratch CNN,
  loaded from a ".pt"/".pth" checkpoint produced by
  keyword_spotting/training/train.py. This is the PRIMARY/default
  backend: KeywordDetector.DEFAULT_MODEL_PATH points at
  models/kws_nova_cnn.pt. No trained checkpoint exists in this repo yet
  (the wake word remains untrained) -- KeywordDetector reports
  status="NOT_IMPLEMENTED" honestly until one does; no NOVA detection
  accuracy is claimed anywhere.
- openWakeWord (keyword_spotting/oww_backend.py, frozen shared
  embedding + a small trained classifier head loaded from an ONNX
  file) -- kept fully working as an OPTIONAL SECONDARY backend, selected
  by passing an explicit model_path ending in ".onnx". No longer the
  default.
"""
from .detector import KeywordDetector, KeywordResult, get_keyword_detector

__all__ = ["KeywordDetector", "KeywordResult", "get_keyword_detector"]
