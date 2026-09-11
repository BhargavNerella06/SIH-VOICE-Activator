"""
tests/unit/test_tinykws_backend.py
=====================================
Tests for keyword_spotting.tinykws_backend.TinyKWSNetBackend.

Uses a tiny, deterministic, LOCALLY-CONSTRUCTED TinyKWSNet checkpoint
(random-but-seeded weights) throughout -- never a real trained NOVA
model. No accuracy claim of any kind is made or tested here; these
tests only verify checkpoint loading, metadata propagation, and that
scoring runs and returns a valid probability.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from keyword_spotting.model import TinyKWSNet

SR = 1000
WINDOW_SECONDS = 0.5
N_MELS = 8


def _make_checkpoint(path: Path, **overrides) -> dict:
    torch.manual_seed(0)
    model = TinyKWSNet(n_mels=N_MELS, num_classes=2)
    package = {
        "state_dict": model.state_dict(),
        "n_mels": N_MELS,
        "sample_rate": SR,
        "window_seconds": WINDOW_SECONDS,
        "keyword": "NOVA",
    }
    package.update(overrides)
    torch.save(package, str(path))
    return package


# ---------------------------------------------------------------------------
# Checkpoint loading / metadata
# ---------------------------------------------------------------------------

def test_loads_valid_checkpoint_and_exposes_metadata(tmp_path):
    from keyword_spotting.tinykws_backend import TinyKWSNetBackend

    ckpt = tmp_path / "model.pt"
    _make_checkpoint(ckpt)

    backend = TinyKWSNetBackend(ckpt, device="cpu")
    assert backend.n_mels == N_MELS
    assert backend.sample_rate == SR
    assert backend.window_seconds == WINDOW_SECONDS
    assert backend.keyword == "NOVA"
    assert backend.label == "NOVA"


def test_missing_required_key_raises_value_error(tmp_path):
    from keyword_spotting.tinykws_backend import TinyKWSNetBackend

    ckpt = tmp_path / "model.pt"
    torch.save({"n_mels": 8, "sample_rate": 1000, "window_seconds": 0.5}, str(ckpt))  # no state_dict/keyword

    with pytest.raises(ValueError, match="missing required key"):
        TinyKWSNetBackend(ckpt)


def test_non_dict_checkpoint_raises_value_error(tmp_path):
    from keyword_spotting.tinykws_backend import TinyKWSNetBackend

    ckpt = tmp_path / "model.pt"
    torch.save(torch.zeros(4), str(ckpt))  # just a raw tensor, not our checkpoint format

    with pytest.raises(ValueError, match="not a dict"):
        TinyKWSNetBackend(ckpt)


def test_wrong_keyword_raises_value_error(tmp_path):
    from keyword_spotting.tinykws_backend import TinyKWSNetBackend

    ckpt = tmp_path / "model.pt"
    _make_checkpoint(ckpt, keyword="HELLO")

    with pytest.raises(ValueError, match="HELLO"):
        TinyKWSNetBackend(ckpt)


def test_incompatible_state_dict_raises_value_error(tmp_path):
    """A state_dict whose tensor shapes don't match TinyKWSNet's fixed
    architecture (e.g. trained with a different num_classes) is a
    malformed/incompatible checkpoint. Note: n_mels/window_seconds do
    NOT appear in any parameter shape (TinyKWSNet's AdaptiveAvgPool2d
    makes it spatial-size-agnostic by design -- see keyword_spotting/
    model.py), so only a genuine architecture mismatch like this one
    can be caught via load_state_dict()."""
    from keyword_spotting.tinykws_backend import TinyKWSNetBackend

    ckpt = tmp_path / "model.pt"
    torch.manual_seed(0)
    wrong_shaped_model = TinyKWSNet(n_mels=N_MELS, num_classes=3)  # backend always assumes 2
    torch.save(
        {
            "state_dict": wrong_shaped_model.state_dict(),
            "n_mels": N_MELS,
            "sample_rate": SR,
            "window_seconds": WINDOW_SECONDS,
            "keyword": "NOVA",
        },
        str(ckpt),
    )

    with pytest.raises(ValueError, match="incompatible"):
        TinyKWSNetBackend(ckpt)


def test_missing_checkpoint_file_raises(tmp_path):
    from keyword_spotting.tinykws_backend import TinyKWSNetBackend

    with pytest.raises(Exception):
        TinyKWSNetBackend(tmp_path / "does_not_exist.pt")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def test_score_returns_a_probability_in_range(tmp_path):
    from keyword_spotting.tinykws_backend import TinyKWSNetBackend

    ckpt = tmp_path / "model.pt"
    _make_checkpoint(ckpt)
    backend = TinyKWSNetBackend(ckpt)

    window = np.random.RandomState(0).randn(int(SR * WINDOW_SECONDS)).astype(np.float32) * 0.1
    score = backend.score(window, sample_rate=SR)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_score_resamples_when_input_rate_differs(tmp_path):
    from keyword_spotting.tinykws_backend import TinyKWSNetBackend

    ckpt = tmp_path / "model.pt"
    _make_checkpoint(ckpt)
    backend = TinyKWSNetBackend(ckpt)

    window = np.random.RandomState(0).randn(2 * SR).astype(np.float32) * 0.1  # 2x the checkpoint's rate
    score = backend.score(window, sample_rate=2 * SR)  # must not crash on resample path
    assert 0.0 <= score <= 1.0


def test_score_is_deterministic_for_the_same_input(tmp_path):
    from keyword_spotting.tinykws_backend import TinyKWSNetBackend

    ckpt = tmp_path / "model.pt"
    _make_checkpoint(ckpt)
    backend = TinyKWSNetBackend(ckpt)

    window = np.random.RandomState(1).randn(int(SR * WINDOW_SECONDS)).astype(np.float32) * 0.1
    s1 = backend.score(window, sample_rate=SR)
    s2 = backend.score(window, sample_rate=SR)
    assert s1 == s2
