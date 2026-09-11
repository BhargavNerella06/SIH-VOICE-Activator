"""
voice_separation
================
Wraps the original Conv-TasNet implementation (Kaituo XU, 2018) as a
reusable Python package.

Public API
----------
ConvTasNet         - The original model class (unchanged).
SeparatorEngine    - Thin inference wrapper used by the API layer.
get_separator_engine - Process-wide cached accessor for SeparatorEngine.
"""

from .conv_tasnet import ConvTasNet
from .engine import SeparatorEngine, get_separator_engine, peek_cached_engine

__all__ = ["ConvTasNet", "SeparatorEngine", "get_separator_engine", "peek_cached_engine"]
