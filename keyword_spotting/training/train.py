"""
keyword_spotting/training/train.py
======================================
Trains TinyKWSNet on the locally collected NOVA / negative dataset.

LIMITATIONS: this trains on a small, locally generated/collected
dataset (a synthetic TTS bootstrap plus a modest number of real
recordings from one or two speakers). It is a first prototype, not a
production-accuracy model. Do not report or claim generalized accuracy
beyond what is measured here on a held-out split of this same small,
non-diverse dataset.

Splitting is group-aware (see keyword_spotting.training.splitting):
augmented variants of the same source utterance are kept together in
one of train/validation/test, so the reported val/test metrics are not
inflated by near-duplicate leakage.

Usage
-----
    python -m keyword_spotting.training.train \
        --data-dir data/kws/synthetic --data-dir data/kws/real \
        --out models/kws_nova_cnn.pt --epochs 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from keyword_spotting.features import DEFAULT_N_MELS, DEFAULT_SAMPLE_RATE, DEFAULT_WINDOW_SECONDS
from keyword_spotting.model import TinyKWSNet
from keyword_spotting.training.dataset import KWSFeatureDataset, collect_examples
from keyword_spotting.training.metrics import (
    count_parameters,
    evaluate_classifier,
    measure_inference_latency,
    model_file_size_bytes,
)
from keyword_spotting.training.splitting import derive_group_key, group_split_indices


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", action="append", required=True,
                         help="Directory containing positive/ and negative/ subfolders. Repeatable.")
    parser.add_argument("--out", type=str, default="models/kws_nova_cnn.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.15,
                         help="Approximate fraction of groups per class held out for validation.")
    parser.add_argument("--test-fraction", type=float, default=0.15,
                         help="Approximate fraction of groups per class held out for testing.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    data_dirs = [Path(d) for d in args.data_dir]

    waveforms, labels, paths = collect_examples(data_dirs, sample_rate=DEFAULT_SAMPLE_RATE)
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    print(f"Loaded {len(labels)} examples: {n_pos} positive (NOVA), {n_neg} negative")
    if n_pos == 0 or n_neg == 0:
        print("ERROR: need at least one example of each class. Aborting.")
        return 1

    groups = [derive_group_key(p) for p in paths]
    n_groups = len(set(groups))
    print(f"Grouped into {n_groups} source-utterance group(s) for leakage-free splitting")

    dataset = KWSFeatureDataset(
        waveforms, labels, sample_rate=DEFAULT_SAMPLE_RATE, window_seconds=DEFAULT_WINDOW_SECONDS
    )

    train_idx, val_idx, test_idx = group_split_indices(
        labels, groups, val_fraction=args.val_fraction, test_fraction=args.test_fraction, seed=args.seed
    )
    print(f"Split: {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test examples")
    if not val_idx or not test_idx:
        print("ERROR: validation or test split is empty -- add more data or lower the fractions. Aborting.")
        return 1

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    model = TinyKWSNet(n_mels=DEFAULT_N_MELS)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_state = None

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for feats, targets in train_loader:
            optimizer.zero_grad()
            logits = model(feats)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * feats.size(0)
        train_loss = total_loss / max(1, len(train_set))

        val_metrics = evaluate_classifier(model, val_loader)
        val_acc = val_metrics["accuracy"]

        print(f"Epoch {epoch + 1}/{args.epochs} | train_loss={train_loss:.4f} | val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate_classifier(model, test_loader)
    param_count = count_parameters(model)
    n_frames = dataset.features[0].shape[-1]
    latency = measure_inference_latency(model, n_mels=DEFAULT_N_MELS, n_frames=n_frames)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "n_mels": DEFAULT_N_MELS,
            "sample_rate": DEFAULT_SAMPLE_RATE,
            "window_seconds": DEFAULT_WINDOW_SECONDS,
            "keyword": "NOVA",
            "best_val_acc": best_val_acc,
            "test_accuracy": test_metrics["accuracy"],
            "test_false_positive_rate": test_metrics["false_positive_rate"],
            "test_false_negative_rate": test_metrics["false_negative_rate"],
            "n_train": len(train_set),
            "n_val": len(val_set),
            "n_test": len(test_set),
            "param_count": param_count,
        },
        str(out_path),
    )
    model_size_bytes = model_file_size_bytes(out_path)

    print()
    print("=== Final report ===")
    print(f"Saved model to {out_path}")
    print(f"Best val_acc          : {best_val_acc:.3f}  (n_val={len(val_set)})")
    print(f"Test accuracy         : {test_metrics['accuracy']:.3f}  (n_test={len(test_set)})")
    print(f"Test false-positive rt: {test_metrics['false_positive_rate']:.3f}  "
          f"(n_negative={test_metrics['n_negative']})")
    print(f"Test false-negative rt: {test_metrics['false_negative_rate']:.3f}  "
          f"(n_positive={test_metrics['n_positive']})")
    print(f"Parameter count       : {param_count}")
    print(f"Model file size       : {model_size_bytes} bytes ({model_size_bytes / 1024:.1f} KB)")
    print(f"Inference latency     : mean={latency['mean_ms']:.3f}ms  "
          f"median={latency['median_ms']:.3f}ms  p95={latency['p95_ms']:.3f}ms "
          f"(batch=1, CPU, n_runs={latency['n_runs']})")
    print()
    print(
        "NOTE: trained on a small, locally generated/collected dataset "
        "(synthetic TTS + limited real recordings). Treat accuracy as "
        "prototype-level only, not production-grade."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
