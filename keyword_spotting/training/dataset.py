"""
keyword_spotting/training/dataset.py
=======================================
Builds a training dataset from WAV files under one or more
positive/negative directories, extracting fixed-length log-Mel
features via keyword_spotting.features -- the same function used at
inference time, so train and inference never disagree on features.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from audio_processing.utils import ensure_mono, normalize_audio, resample_if_needed
from keyword_spotting.features import (
    DEFAULT_SAMPLE_RATE,
    DEFAULT_WINDOW_SECONDS,
    extract_features,
)


def _load_wav(path: Path, target_sr: int) -> np.ndarray:
    y, sr = sf.read(str(path))
    y = ensure_mono(y).astype(np.float32)
    if sr != target_sr:
        y = resample_if_needed(y, sr, target_sr)
    return normalize_audio(y)


def collect_examples(
    data_dirs: List[Path],
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> Tuple[List[np.ndarray], List[int], List[Path]]:
    """
    Each directory in `data_dirs` must contain positive/ and/or negative/
    subdirectories of WAV files. Label convention: 1 = NOVA, 0 = negative.

    Returns waveforms and labels alongside the source path of each
    example, so callers can derive group keys (see
    keyword_spotting.training.splitting) for leakage-free splitting.
    """
    waveforms: List[np.ndarray] = []
    labels: List[int] = []
    paths: List[Path] = []
    for base in data_dirs:
        for label, subdir in ((1, "positive"), (0, "negative")):
            folder = base / subdir
            if not folder.exists():
                continue
            for wav_path in sorted(folder.glob("*.wav")):
                try:
                    y = _load_wav(wav_path, sample_rate)
                except Exception:
                    continue
                waveforms.append(y)
                labels.append(label)
                paths.append(wav_path)
    return waveforms, labels, paths


class KWSFeatureDataset(Dataset):
    """Precomputes log-Mel features for every example up front."""

    def __init__(
        self,
        waveforms: List[np.ndarray],
        labels: List[int],
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        window_seconds: float = DEFAULT_WINDOW_SECONDS,
    ) -> None:
        self.sample_rate = sample_rate
        self.window_seconds = window_seconds
        self.features = [
            extract_features(w, sample_rate=sample_rate, window_seconds=window_seconds)
            for w in waveforms
        ]
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        feat = torch.from_numpy(self.features[idx]).unsqueeze(0)  # [1, n_mels, T]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feat, label
