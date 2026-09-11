"""
keyword_spotting/training/calibrate_oww_threshold.py
=========================================================
Picks a threshold/consecutive_frames operating point for the trained
openWakeWord "NOVA" model using a SMALL real-microphone validation
set -- not a training set. This deliberately does not train anything;
it only measures the already-trained model against real recordings and
suggests a threshold.

Per the Stage 2 plan, the wake-word model itself is trained externally
(WSL2/Colab, Piper-based synthetic data + openWakeWord's pre-built
negative/background datasets -- see keyword_spotting/oww_backend.py's
module docstring and the implementation plan). This script only
consumes the resulting models/nova.onnx.

LIMITATIONS: a 15-20 clip validation set is enough to sanity-check that
the model responds to a real voice and pick a reasonable threshold. It
is NOT a statistically meaningful accuracy/false-accept-rate
evaluation. Treat the accuracy number this script prints as a
small-sample sanity check only, exactly like keyword_spotting/training/train.py
does for its own (different, from-scratch) model.

Usage
-----
    # Record a small real-mic validation set first (reuses the
    # existing recording tool as-is, at a small --count):
    python -m keyword_spotting.training.record_data \
        --label positive --count 15 --out-dir data/kws/calibration
    python -m keyword_spotting.training.record_data \
        --label negative --count 15 --out-dir data/kws/calibration

    # Then calibrate against the trained model:
    python -m keyword_spotting.training.calibrate_oww_threshold \
        --data-dir data/kws/calibration --model models/nova.onnx
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf

from audio_processing.utils import ensure_mono, normalize_audio, resample_if_needed
from keyword_spotting.oww_backend import TARGET_SAMPLE_RATE, OpenWakeWordBackend

CANDIDATE_THRESHOLDS = [round(t, 2) for t in np.arange(0.30, 0.96, 0.05)]
DEFAULT_RECOMMENDED_CONSECUTIVE_FRAMES = 2


def _load_wav(path: Path, target_sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    y, sr = sf.read(str(path))
    y = ensure_mono(y).astype(np.float32)
    if sr != target_sr:
        y = resample_if_needed(y, sr, target_sr)
    return normalize_audio(y)


def _collect_scores(backend: OpenWakeWordBackend, folder: Path) -> List[float]:
    scores = []
    for wav_path in sorted(folder.glob("*.wav")):
        y = _load_wav(wav_path)
        scores.append(backend.score(y, sample_rate=TARGET_SAMPLE_RATE))
    return scores


def _best_operating_point(
    pos_scores: List[float], neg_scores: List[float]
) -> Tuple[float, int, float, float, float]:
    """Sweep thresholds; return (threshold, consecutive_frames, accuracy, false_accept_rate, false_reject_rate).

    consecutive_frames is reported for use by KeywordDetector's debounce,
    but this single-clip-per-score evaluation can only directly measure
    the per-clip threshold; consecutive_frames=1 is used for scoring
    here and higher values are reported as the conservative default
    recommendation (they only reduce false accepts further at some
    cost to reaction time, per detector.py's existing debounce design).
    """
    best = (0.5, DEFAULT_RECOMMENDED_CONSECUTIVE_FRAMES, -1.0, 1.0, 1.0)
    for t in CANDIDATE_THRESHOLDS:
        tp = sum(1 for s in pos_scores if s >= t)
        fn = len(pos_scores) - tp
        fp = sum(1 for s in neg_scores if s >= t)
        tn = len(neg_scores) - fp
        total = tp + fn + fp + tn
        accuracy = (tp + tn) / total if total else 0.0
        far = fp / len(neg_scores) if neg_scores else 0.0
        frr = fn / len(pos_scores) if pos_scores else 0.0
        if accuracy > best[2]:
            best = (t, DEFAULT_RECOMMENDED_CONSECUTIVE_FRAMES, accuracy, far, frr)
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=str, default="data/kws/calibration",
                         help="Directory with positive/ and negative/ WAV subfolders (small real-mic set).")
    parser.add_argument("--model", type=str, default="models/nova.onnx")
    parser.add_argument("--out", type=str, default="models/nova_calibration.json")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: no trained openWakeWord model found at {model_path}.")
        print(
            "This script only calibrates an already-trained model; it does not "
            "train one. See keyword_spotting/oww_backend.py's module docstring "
            "and the Stage 2 implementation plan for the external (WSL2/Colab) "
            "training steps that produce this file."
        )
        return 1

    data_dir = Path(args.data_dir)
    pos_dir, neg_dir = data_dir / "positive", data_dir / "negative"
    if not pos_dir.exists() or not neg_dir.exists():
        print(f"ERROR: expected {pos_dir} and {neg_dir} to exist.")
        print(
            "Record a small real-mic validation set first, e.g.:\n"
            f"  python -m keyword_spotting.training.record_data --label positive --count 15 --out-dir {data_dir}\n"
            f"  python -m keyword_spotting.training.record_data --label negative --count 15 --out-dir {data_dir}"
        )
        return 1

    backend = OpenWakeWordBackend(model_path)
    pos_scores = _collect_scores(backend, pos_dir)
    neg_scores = _collect_scores(backend, neg_dir)

    if not pos_scores or not neg_scores:
        print(f"ERROR: need at least one WAV in both {pos_dir} and {neg_dir}.")
        return 1

    threshold, consecutive_frames, accuracy, far, frr = _best_operating_point(pos_scores, neg_scores)

    result = {
        "model": str(model_path),
        "n_positive_clips": len(pos_scores),
        "n_negative_clips": len(neg_scores),
        "recommended_threshold": threshold,
        "recommended_consecutive_frames": consecutive_frames,
        "measured_accuracy": round(accuracy, 3),
        "measured_false_accept_rate": round(far, 3),
        "measured_false_reject_rate": round(frr, 3),
        "positive_scores": [round(s, 3) for s in pos_scores],
        "negative_scores": [round(s, 3) for s in neg_scores],
        "note": (
            f"Measured on only {len(pos_scores)} positive / {len(neg_scores)} negative "
            "real-mic clips. This is a small-sample sanity check and threshold "
            "suggestion, NOT a statistically meaningful accuracy or "
            "false-accept-rate evaluation. Do not report this number as "
            "production accuracy."
        ),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Positive scores: {result['positive_scores']}")
    print(f"Negative scores: {result['negative_scores']}")
    print(f"Recommended threshold={threshold}, consecutive_frames={consecutive_frames}")
    print(f"Measured on this small set: accuracy={accuracy:.3f}, false_accept_rate={far:.3f}, false_reject_rate={frr:.3f}")
    print(f"Wrote {out_path}")
    print()
    print(result["note"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
