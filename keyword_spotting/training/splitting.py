"""
keyword_spotting/training/splitting.py
=========================================
Group-aware train/validation/test splitting.

Splits are performed on *groups* of examples, not individual examples,
so that augmented variants of the same source utterance (e.g. the
pitch-shifted / time-stretched / noisy copies produced by
synth_data.py) always land together in exactly one split. Without
this, a model could be "validated" or "tested" on a near-duplicate of
something it trained on, silently inflating accuracy.

Splitting is stratified by label: each class's groups are shuffled and
divided independently, so train/val/test keep roughly the same class
balance as the full dataset.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

# Matches the "{tag}_grp{N}_var{M}.wav" filename convention written by
# synth_data.py for augmented variants of one base utterance.
_GROUP_RE = re.compile(r"^(?P<tag>[a-zA-Z]+)_grp(?P<group>\d+)_var\d+$")


def derive_group_key(path: Path) -> str:
    """
    Return an identifier shared by every augmented variant of the same
    source utterance, derived from the filename.

    Recognises the "{tag}_grp{N}_var{M}.wav" convention. Any other
    filename (real recordings, or synthetic silence/noise clips, none
    of which are augmented copies of one another) is treated as its
    own single-member group -- i.e. it can only ever land in one
    split anyway, so no grouping is needed for it.
    """
    stem = Path(path).stem
    m = _GROUP_RE.match(stem)
    if m:
        return f"{m.group('tag')}_grp{m.group('group')}"
    return stem


def group_split_indices(
    labels: Sequence[int],
    groups: Sequence[str],
    val_fraction: float,
    test_fraction: float,
    seed: int = 0,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Split example indices into (train, val, test) index lists such that:
      - no group appears in more than one split (see `derive_group_key`)
      - splitting is stratified per label, so class balance is
        preserved as closely as group sizes allow

    Parameters
    ----------
    labels, groups : parallel sequences, one entry per example.
    val_fraction, test_fraction : approximate fraction of *groups*
        (per class) to place in the validation / test splits.

    Raises
    ------
    ValueError
        If a group contains examples with more than one label (that
        would make "no duplicates across splits" ambiguous), or if
        the fractions are out of range.
    """
    if len(labels) != len(groups):
        raise ValueError("labels and groups must be the same length")
    if not (0.0 <= val_fraction < 1.0) or not (0.0 <= test_fraction < 1.0):
        raise ValueError("val_fraction and test_fraction must be in [0, 1)")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be < 1.0")

    group_label: Dict[str, int] = {}
    group_members: Dict[str, List[int]] = {}
    for idx, (label, group) in enumerate(zip(labels, groups)):
        if group in group_label and group_label[group] != label:
            raise ValueError(
                f"group {group!r} spans multiple labels "
                f"({group_label[group]} and {label}) -- cannot split safely"
            )
        group_label[group] = label
        group_members.setdefault(group, []).append(idx)

    rng = np.random.default_rng(seed)
    labels_present = sorted(set(group_label.values()))

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    for label in labels_present:
        label_groups = sorted(g for g, l in group_label.items() if l == label)
        order = rng.permutation(len(label_groups))
        label_groups = [label_groups[i] for i in order]

        n = len(label_groups)
        n_test = min(n, int(round(n * test_fraction)))
        n_val = min(n - n_test, int(round(n * val_fraction)))

        test_groups = label_groups[:n_test]
        val_groups = label_groups[n_test:n_test + n_val]
        train_groups = label_groups[n_test + n_val:]

        for g in test_groups:
            test_idx.extend(group_members[g])
        for g in val_groups:
            val_idx.extend(group_members[g])
        for g in train_groups:
            train_idx.extend(group_members[g])

    return train_idx, val_idx, test_idx
