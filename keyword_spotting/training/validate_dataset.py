"""
keyword_spotting/training/validate_dataset.py
=================================================
Validates the KWS dataset under data/kws/{synthetic,real}/{positive,negative}
before training. Checks:
  - corrupted / unreadable audio files
  - sample-rate consistency
  - per-class duration statistics
  - class balance (positive vs negative)
  - duplicate filenames across the dataset

This does NOT train anything and does NOT report model accuracy -- it
only checks that the data itself is sane.

Usage
-----
    python -m keyword_spotting.training.validate_dataset
    python -m keyword_spotting.training.validate_dataset --data-dir data/kws/synthetic --data-dir data/kws/real
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path
from typing import Dict, List

import soundfile as sf

DEFAULT_DATA_DIRS = ["data/kws/synthetic", "data/kws/real"]
EXPECTED_SAMPLE_RATE = 16000
IMBALANCE_WARN_RATIO = 5.0


def _scan(data_dirs: List[Path]) -> Dict[str, List[Path]]:
    files: Dict[str, List[Path]] = {"positive": [], "negative": []}
    for base in data_dirs:
        for label in ("positive", "negative"):
            folder = base / label
            if folder.exists():
                files[label].extend(sorted(folder.glob("*.wav")))
    return files


def validate(data_dirs: List[Path]) -> bool:
    ok = True
    print("=== KWS dataset validation ===")
    for base in data_dirs:
        print(f"Scanning: {base}  (exists={base.exists()})")
    print()

    files = _scan(data_dirs)
    n_pos, n_neg = len(files["positive"]), len(files["negative"])

    # -- corrupted files / duration / sample rate --
    durations: Dict[str, List[float]] = {"positive": [], "negative": []}
    sample_rates: collections.Counter = collections.Counter()
    corrupted: List[Path] = []
    zero_length: List[Path] = []

    for label, paths in files.items():
        for p in paths:
            try:
                info = sf.info(str(p))
            except Exception as exc:
                corrupted.append(p)
                print(f"  CORRUPTED: {p}  ({exc})")
                continue
            sample_rates[info.samplerate] += 1
            duration = info.frames / info.samplerate if info.samplerate else 0.0
            durations[label].append(duration)
            if duration <= 0.0:
                zero_length.append(p)

    print(f"Positive files : {n_pos}")
    print(f"Negative files : {n_neg}")
    if n_pos == 0 or n_neg == 0:
        print("FAIL: at least one class has zero examples.")
        ok = False

    print()
    if corrupted:
        print(f"FAIL: {len(corrupted)} corrupted/unreadable file(s) (listed above).")
        ok = False
    else:
        print("OK: no corrupted files found.")

    if zero_length:
        print(f"FAIL: {len(zero_length)} zero-length clip(s):")
        for p in zero_length:
            print(f"  {p}")
        ok = False

    # -- sample-rate consistency --
    print()
    print("Sample rates found:", dict(sample_rates))
    if not sample_rates:
        pass
    elif len(sample_rates) > 1:
        print(
            f"WARNING: mixed sample rates in raw files (expected uniform "
            f"{EXPECTED_SAMPLE_RATE} Hz). The training/inference pipeline "
            "resamples everything, so this is not fatal, but confirm it's "
            "intentional (e.g. raw real recordings before resampling)."
        )
    elif EXPECTED_SAMPLE_RATE not in sample_rates:
        print(f"WARNING: all files share one rate, but it is not the expected {EXPECTED_SAMPLE_RATE} Hz.")
    else:
        print(f"OK: all files at the expected {EXPECTED_SAMPLE_RATE} Hz.")

    # -- duration stats --
    print()
    for label in ("positive", "negative"):
        d = durations[label]
        if not d:
            print(f"{label}: no files")
            continue
        print(
            f"{label}: n={len(d)}  min={min(d):.2f}s  max={max(d):.2f}s  "
            f"mean={sum(d) / len(d):.2f}s  total={sum(d):.1f}s"
        )

    # -- class balance --
    print()
    if n_pos > 0 and n_neg > 0:
        ratio = n_neg / n_pos
        print(f"Class ratio (negative:positive) = {ratio:.2f} : 1")
        if ratio > IMBALANCE_WARN_RATIO or ratio < (1.0 / IMBALANCE_WARN_RATIO):
            print(f"WARNING: classes are heavily imbalanced (beyond {IMBALANCE_WARN_RATIO}:1).")
    else:
        print("Class ratio: N/A (one class is empty)")

    # -- duplicate filenames --
    print()
    basename_map: Dict[str, List[Path]] = collections.defaultdict(list)
    for label, paths in files.items():
        for p in paths:
            basename_map[p.name].append(p)
    duplicates = {name: paths for name, paths in basename_map.items() if len(paths) > 1}
    if duplicates:
        print(f"FAIL: {len(duplicates)} duplicate filename(s) found across the dataset:")
        for name, paths in duplicates.items():
            print(f"  {name}: {[str(p) for p in paths]}")
        ok = False
    else:
        print("OK: no duplicate filenames.")

    print()
    print("=== RESULT:", "PASS" if ok else "FAIL", "===")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", action="append", default=None,
        help="Directory containing positive/ and negative/ subfolders. Repeatable. "
             f"Default: {DEFAULT_DATA_DIRS}",
    )
    args = parser.parse_args()
    dirs = [Path(d) for d in (args.data_dir or DEFAULT_DATA_DIRS)]
    passed = validate(dirs)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
