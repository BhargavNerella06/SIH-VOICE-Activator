"""
keyword_spotting/training/metrics.py
========================================
Evaluation metrics for a trained TinyKWSNet checkpoint: classification
accuracy, false-positive/false-negative rate, parameter count, model
file size, and per-window inference latency.

Kept separate from train.py so a saved checkpoint can be re-evaluated
later (e.g. on a fresh test set) without retraining.

Label convention (matches keyword_spotting.model): 0 = negative,
1 = NOVA / positive.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def model_file_size_bytes(path: Path) -> int:
    return Path(path).stat().st_size


def evaluate_classifier(model: nn.Module, loader: DataLoader) -> Dict[str, float]:
    """
    Run `model` over every batch in `loader` and compute accuracy,
    false-positive rate (negative examples predicted positive) and
    false-negative rate (positive examples predicted negative).
    """
    model.eval()
    correct = 0
    total = 0
    false_positives = 0
    false_negatives = 0
    n_actual_pos = 0
    n_actual_neg = 0

    with torch.no_grad():
        for feats, targets in loader:
            logits = model(feats)
            preds = logits.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

            pos_mask = targets == 1
            neg_mask = targets == 0
            n_actual_pos += int(pos_mask.sum().item())
            n_actual_neg += int(neg_mask.sum().item())
            false_positives += int(((preds == 1) & neg_mask).sum().item())
            false_negatives += int(((preds == 0) & pos_mask).sum().item())

    return {
        "accuracy": correct / total if total else 0.0,
        "false_positive_rate": false_positives / n_actual_neg if n_actual_neg else 0.0,
        "false_negative_rate": false_negatives / n_actual_pos if n_actual_pos else 0.0,
        "n_examples": total,
        "n_positive": n_actual_pos,
        "n_negative": n_actual_neg,
    }


def measure_inference_latency(
    model: nn.Module,
    n_mels: int,
    n_frames: int,
    n_runs: int = 50,
    n_warmup: int = 5,
) -> Dict[str, float]:
    """
    Measure single-window (batch size 1) CPU inference latency in
    milliseconds by repeatedly running the model forward pass on a
    fixed-shape dummy input. Reports mean/median/p95 over `n_runs`
    timed runs after `n_warmup` untimed warmup runs.
    """
    model.eval()
    dummy = torch.randn(1, 1, n_mels, n_frames)
    with torch.no_grad():
        for _ in range(n_warmup):
            model(dummy)

        timings_ms = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(dummy)
            timings_ms.append((time.perf_counter() - t0) * 1000.0)

    timings_ms.sort()
    n = len(timings_ms)
    return {
        "mean_ms": sum(timings_ms) / n,
        "median_ms": timings_ms[n // 2],
        "p95_ms": timings_ms[min(n - 1, int(round(0.95 * (n - 1))))],
        "n_runs": n_runs,
    }
