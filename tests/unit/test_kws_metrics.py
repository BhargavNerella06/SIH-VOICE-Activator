"""
tests/unit/test_kws_metrics.py
=================================
Tests for keyword_spotting.training.metrics: accuracy/FPR/FNR
computation, parameter counting, model file size, and inference
latency measurement.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class _AlwaysPredict(nn.Module):
    """Stub model that ignores its input and always predicts one fixed class."""

    def __init__(self, predicted_class: int):
        super().__init__()
        self.predicted_class = predicted_class
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch = x.size(0)
        logits = torch.full((batch, 2), -10.0)
        logits[:, self.predicted_class] = 10.0
        return logits


def _loader(labels):
    feats = torch.zeros(len(labels), 1, 4, 4)
    targets = torch.tensor(labels, dtype=torch.long)
    return DataLoader(TensorDataset(feats, targets), batch_size=4)


def test_count_parameters_matches_model_num_parameters():
    from keyword_spotting.model import TinyKWSNet
    from keyword_spotting.training.metrics import count_parameters

    model = TinyKWSNet(n_mels=40)
    assert count_parameters(model) == model.num_parameters()
    assert count_parameters(model) > 0


def test_model_file_size_bytes_reads_actual_file_size(tmp_path):
    from keyword_spotting.training.metrics import model_file_size_bytes

    path = tmp_path / "dummy.pt"
    path.write_bytes(b"0" * 1234)
    assert model_file_size_bytes(path) == 1234


def test_evaluate_classifier_perfect_predictions():
    from keyword_spotting.training.metrics import evaluate_classifier

    class _Oracle(nn.Module):
        def __init__(self, labels):
            super().__init__()
            self._labels = labels
            self._call = 0
            self.dummy = nn.Parameter(torch.zeros(1))

        def forward(self, x):
            batch = x.size(0)
            chunk = self._labels[self._call:self._call + batch]
            self._call += batch
            logits = torch.full((batch, 2), -10.0)
            for i, label in enumerate(chunk):
                logits[i, label] = 10.0
            return logits

    labels = [1, 1, 0, 0, 0]
    model = _Oracle(labels)
    result = evaluate_classifier(model, _loader(labels))

    assert result["accuracy"] == 1.0
    assert result["false_positive_rate"] == 0.0
    assert result["false_negative_rate"] == 0.0
    assert result["n_positive"] == 2
    assert result["n_negative"] == 3


def test_evaluate_classifier_computes_fpr_and_fnr():
    from keyword_spotting.training.metrics import evaluate_classifier

    # Model always predicts positive (class 1).
    labels = [1, 1, 0, 0, 0, 0]  # 2 positive, 4 negative
    model = _AlwaysPredict(predicted_class=1)
    result = evaluate_classifier(model, _loader(labels))

    assert result["accuracy"] == 2 / 6
    # All 4 negatives were predicted positive -> FPR = 1.0
    assert result["false_positive_rate"] == 1.0
    # Both positives were predicted correctly -> FNR = 0.0
    assert result["false_negative_rate"] == 0.0


def test_evaluate_classifier_always_negative_gives_full_fnr():
    from keyword_spotting.training.metrics import evaluate_classifier

    labels = [1, 1, 1, 0, 0]  # 3 positive, 2 negative
    model = _AlwaysPredict(predicted_class=0)
    result = evaluate_classifier(model, _loader(labels))

    assert result["false_positive_rate"] == 0.0
    assert result["false_negative_rate"] == 1.0


def test_measure_inference_latency_returns_positive_timings():
    from keyword_spotting.model import TinyKWSNet
    from keyword_spotting.training.metrics import measure_inference_latency

    model = TinyKWSNet(n_mels=40)
    result = measure_inference_latency(model, n_mels=40, n_frames=101, n_runs=5, n_warmup=1)

    assert result["mean_ms"] > 0
    assert result["median_ms"] > 0
    assert result["p95_ms"] >= result["median_ms"] >= 0
    assert result["n_runs"] == 5
