"""
tests/unit/test_kws_splitting.py
===================================
Tests for keyword_spotting.training.splitting: group-aware
train/validation/test splitting that keeps augmented variants of the
same source utterance out of more than one split.
"""

from pathlib import Path

import pytest


def test_derive_group_key_recognizes_grouped_filename():
    from keyword_spotting.training.splitting import derive_group_key
    assert derive_group_key(Path("pos_grp0007_var02.wav")) == "pos_grp0007"
    assert derive_group_key(Path("neg_grp0000_var00.wav")) == "neg_grp0000"


def test_derive_group_key_groups_all_variants_of_one_base_utterance():
    from keyword_spotting.training.splitting import derive_group_key
    keys = {derive_group_key(Path(f"pos_grp0003_var{i:02d}.wav")) for i in range(5)}
    assert keys == {"pos_grp0003"}


def test_derive_group_key_fallback_treats_ungrouped_file_as_its_own_group():
    from keyword_spotting.training.splitting import derive_group_key
    # Real recordings and old-style/silence/noise synthetic clips have no
    # "grp"/"var" markers -- each is its own single-member group.
    assert derive_group_key(Path("real_ab12cd34_0007.wav")) == "real_ab12cd34_0007"
    assert derive_group_key(Path("neg_0361.wav")) == "neg_0361"


def _make_grouped_dataset(n_groups_per_class=10, variants_per_group=4):
    labels = []
    groups = []
    for label in (0, 1):
        tag = "pos" if label == 1 else "neg"
        for g in range(n_groups_per_class):
            for v in range(variants_per_group):
                labels.append(label)
                groups.append(f"{tag}_grp{g:04d}")
    return labels, groups


def test_group_split_indices_no_group_spans_multiple_splits():
    from keyword_spotting.training.splitting import group_split_indices

    labels, groups = _make_grouped_dataset()
    train_idx, val_idx, test_idx = group_split_indices(
        labels, groups, val_fraction=0.2, test_fraction=0.2, seed=0
    )

    train_groups = {groups[i] for i in train_idx}
    val_groups = {groups[i] for i in val_idx}
    test_groups = {groups[i] for i in test_idx}

    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)

    # every index accounted for exactly once
    all_idx = sorted(train_idx + val_idx + test_idx)
    assert all_idx == list(range(len(labels)))


def test_group_split_indices_reproducible_with_seed():
    from keyword_spotting.training.splitting import group_split_indices

    labels, groups = _make_grouped_dataset()
    split_a = group_split_indices(labels, groups, val_fraction=0.2, test_fraction=0.2, seed=42)
    split_b = group_split_indices(labels, groups, val_fraction=0.2, test_fraction=0.2, seed=42)
    assert split_a == split_b


def test_group_split_indices_different_seeds_can_differ():
    from keyword_spotting.training.splitting import group_split_indices

    labels, groups = _make_grouped_dataset(n_groups_per_class=20)
    split_a = group_split_indices(labels, groups, val_fraction=0.2, test_fraction=0.2, seed=1)
    split_b = group_split_indices(labels, groups, val_fraction=0.2, test_fraction=0.2, seed=2)
    assert split_a != split_b


def test_group_split_indices_approximately_respects_fractions():
    from keyword_spotting.training.splitting import group_split_indices

    labels, groups = _make_grouped_dataset(n_groups_per_class=20, variants_per_group=5)
    train_idx, val_idx, test_idx = group_split_indices(
        labels, groups, val_fraction=0.2, test_fraction=0.2, seed=0
    )
    total = len(labels)
    assert 0.15 < len(val_idx) / total < 0.25
    assert 0.15 < len(test_idx) / total < 0.25
    assert 0.5 < len(train_idx) / total < 0.7


def test_group_split_indices_raises_on_group_spanning_multiple_labels():
    from keyword_spotting.training.splitting import group_split_indices

    labels = [0, 1]
    groups = ["shared_grp0000", "shared_grp0000"]
    with pytest.raises(ValueError):
        group_split_indices(labels, groups, val_fraction=0.2, test_fraction=0.2, seed=0)


def test_group_split_indices_rejects_invalid_fractions():
    from keyword_spotting.training.splitting import group_split_indices

    labels, groups = _make_grouped_dataset()
    with pytest.raises(ValueError):
        group_split_indices(labels, groups, val_fraction=0.6, test_fraction=0.6, seed=0)
