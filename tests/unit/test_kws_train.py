"""
tests/unit/test_kws_train.py
===============================
End-to-end smoke test for keyword_spotting.training.train's CLI
entrypoint. Prior to this file, train.py had zero test coverage of its
own -- only its individual components (dataset, splitting, metrics,
model) were tested in isolation. This does not assert any accuracy
number (the tiny synthetic dataset here is not a real KWS training set
and no accuracy claim is made or implied) -- it only verifies that the
full dataset -> split -> train -> evaluate -> save pipeline runs
end-to-end without crashing and produces a checkpoint with the metadata
fields inference would need.
"""

from __future__ import annotations

import sys

import numpy as np
import soundfile as sf
import torch


def _write_wav(path, seed, seconds=1.0, sr=16000):
    path.parent.mkdir(parents=True, exist_ok=True)
    y = (np.random.RandomState(seed).randn(int(sr * seconds)) * 0.01).astype(np.float32)
    sf.write(str(path), y, sr, subtype="PCM_16")


def _make_tiny_dataset(base):
    # 4 distinct single-member groups per class -- enough for train.py's
    # default-shaped val/test fractions to each get at least one group,
    # without needing the "{tag}_grp{N}_var{M}" augmentation-group naming.
    for i in range(4):
        _write_wav(base / "positive" / f"pos_{i}.wav", seed=i)
        _write_wav(base / "negative" / f"neg_{i}.wav", seed=100 + i)


def test_train_main_runs_end_to_end_and_saves_a_loadable_checkpoint(tmp_path, monkeypatch, capsys):
    from keyword_spotting.training import train as train_module

    data_dir = tmp_path / "data"
    _make_tiny_dataset(data_dir)
    out_path = tmp_path / "model.pt"

    argv = [
        "train.py",
        "--data-dir", str(data_dir),
        "--out", str(out_path),
        "--epochs", "1",
        "--batch-size", "2",
        "--val-fraction", "0.25",
        "--test-fraction", "0.25",
        "--seed", "0",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    exit_code = train_module.main()

    assert exit_code == 0
    assert out_path.exists()

    package = torch.load(str(out_path), map_location="cpu")
    # Metadata a caller needs to reconstruct the model and its feature
    # pipeline for inference (requirement: "does training save enough
    # metadata for inference?").
    for key in ("state_dict", "n_mels", "sample_rate", "window_seconds", "keyword"):
        assert key in package, f"checkpoint is missing expected key: {key!r}"
    assert package["keyword"] == "NOVA"

    from keyword_spotting.model import TinyKWSNet
    model = TinyKWSNet(n_mels=package["n_mels"])
    model.load_state_dict(package["state_dict"])  # must not raise: architecture must match


def test_train_main_aborts_cleanly_when_a_class_is_missing(tmp_path, monkeypatch):
    """No negative examples at all -> a clean, early, non-crashing abort."""
    from keyword_spotting.training import train as train_module

    data_dir = tmp_path / "data"
    _write_wav(data_dir / "positive" / "pos_0.wav", seed=0)
    out_path = tmp_path / "model.pt"

    argv = [
        "train.py",
        "--data-dir", str(data_dir),
        "--out", str(out_path),
        "--epochs", "1",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    exit_code = train_module.main()

    assert exit_code == 1
    assert not out_path.exists()
