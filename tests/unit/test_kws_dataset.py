"""
tests/unit/test_kws_dataset.py
=================================
Tests for keyword_spotting.training.dataset: loading examples from
positive/negative directories, and that the returned paths integrate
correctly with group-aware splitting.
"""

import numpy as np
import soundfile as sf


def _write_wav(path, seconds=0.5, sr=16000):
    path.parent.mkdir(parents=True, exist_ok=True)
    y = (np.random.RandomState(0).randn(int(sr * seconds)) * 0.01).astype(np.float32)
    sf.write(str(path), y, sr, subtype="PCM_16")


def test_collect_examples_returns_waveforms_labels_and_paths(tmp_path):
    from keyword_spotting.training.dataset import collect_examples

    base = tmp_path / "data"
    _write_wav(base / "positive" / "pos_grp0000_var00.wav")
    _write_wav(base / "negative" / "neg_grp0000_var00.wav")

    waveforms, labels, paths = collect_examples([base], sample_rate=16000)

    assert len(waveforms) == len(labels) == len(paths) == 2
    assert set(labels) == {0, 1}
    assert all(p.suffix == ".wav" for p in paths)


def test_collect_examples_skips_missing_subdirs(tmp_path):
    from keyword_spotting.training.dataset import collect_examples

    base = tmp_path / "data"
    _write_wav(base / "positive" / "pos_grp0000_var00.wav")
    # no negative/ subdirectory at all

    waveforms, labels, paths = collect_examples([base], sample_rate=16000)
    assert len(waveforms) == 1
    assert labels == [1]


def test_collect_examples_paths_feed_group_derivation(tmp_path):
    """End-to-end: paths returned by collect_examples must be usable by
    keyword_spotting.training.splitting.derive_group_key."""
    from keyword_spotting.training.dataset import collect_examples
    from keyword_spotting.training.splitting import derive_group_key

    base = tmp_path / "data"
    for variant in range(3):
        _write_wav(base / "positive" / f"pos_grp0000_var{variant:02d}.wav")
    _write_wav(base / "positive" / "pos_grp0001_var00.wav")

    _, labels, paths = collect_examples([base], sample_rate=16000)
    groups = [derive_group_key(p) for p in paths]

    assert groups.count("pos_grp0000") == 3
    assert groups.count("pos_grp0001") == 1
    assert len(set(groups)) == 2


def test_kws_feature_dataset_shapes_match_collected_examples(tmp_path):
    from keyword_spotting.training.dataset import KWSFeatureDataset, collect_examples

    base = tmp_path / "data"
    _write_wav(base / "positive" / "pos_grp0000_var00.wav")
    _write_wav(base / "negative" / "neg_grp0000_var00.wav")

    waveforms, labels, _ = collect_examples([base], sample_rate=16000)
    dataset = KWSFeatureDataset(waveforms, labels, sample_rate=16000, window_seconds=1.0)

    assert len(dataset) == 2
    feat, label = dataset[0]
    assert feat.ndim == 3  # [1, n_mels, T]
    assert label.item() in (0, 1)
