"""
tests/unit/test_benchmark_kws_calculations.py
==================================================
Focused validation of the calculations behind docs/STAGE_2_EDGE_BENCHMARK.md's
INT8-size estimate and buffer-size math (scripts/benchmark_kws.py). These are
plain functions, not part of the installed package, so they're imported here
via sys.path manipulation -- same convention the script itself uses.

This does NOT test model accuracy or training -- only that the benchmark
script's own numerical estimates are actually correct, since they underpin
size/RAM-budget claims in the edge-readiness report.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
import benchmark_kws as bk  # noqa: E402

from keyword_spotting.model import TinyKWSNet  # noqa: E402


# ---------------------------------------------------------------------------
# fold_batchnorm_into_conv: the folded conv must reproduce conv+BN's output
# ---------------------------------------------------------------------------

def test_fold_batchnorm_into_conv_preserves_output_numerically():
    torch.manual_seed(0)
    conv = nn.Conv2d(3, 5, kernel_size=3, padding=1, bias=True)
    bn = nn.BatchNorm2d(5)
    bn.eval()
    # Give BN non-trivial (non-identity) learned statistics/affine params,
    # otherwise this test would pass trivially.
    with torch.no_grad():
        bn.weight.copy_(torch.rand(5) + 0.5)
        bn.bias.copy_(torch.randn(5))
        bn.running_mean.copy_(torch.randn(5))
        bn.running_var.copy_(torch.rand(5) + 0.5)

    x = torch.randn(2, 3, 8, 8)
    with torch.no_grad():
        expected = bn(conv(x))

    folded_w, folded_b = bk.fold_batchnorm_into_conv(conv, bn)
    with torch.no_grad():
        actual = torch.nn.functional.conv2d(x, folded_w, folded_b, padding=1)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)


def test_fold_batchnorm_into_conv_preserves_shapes():
    conv = nn.Conv2d(16, 32, kernel_size=3, padding=1)
    bn = nn.BatchNorm2d(32)
    w, b = bk.fold_batchnorm_into_conv(conv, bn)
    assert w.shape == conv.weight.shape
    assert b.shape == (32,)


# ---------------------------------------------------------------------------
# estimated_int8_deployment_size_bytes: totals must match a hand count
# ---------------------------------------------------------------------------

def test_int8_estimate_matches_hand_computed_totals_for_a_small_model():
    torch.manual_seed(0)
    model = TinyKWSNet(n_mels=8, num_classes=2)
    est = bk.estimated_int8_deployment_size_bytes(model)

    # Weight element counts: conv1 (1*16*3*3) + conv2 (16*32*3*3) +
    # linear1 (32*32) + linear2 (32*2). BatchNorm params are folded away
    # entirely, contributing zero extra weight elements.
    expected_weight_elems = (1 * 16 * 3 * 3) + (16 * 32 * 3 * 3) + (32 * 32) + (32 * 2)
    # Bias element counts: one bias per folded conv/linear output channel.
    expected_bias_elems = 16 + 32 + 32 + 2

    assert est["total_weight_elems"] == expected_weight_elems
    assert est["total_bias_elems"] == expected_bias_elems
    assert est["weight_bytes_int8"] == expected_weight_elems * 1
    assert est["bias_bytes_int32"] == expected_bias_elems * 4
    assert est["estimated_total_bytes"] == est["weight_bytes_int8"] + est["bias_bytes_int32"]


def test_int8_estimate_smaller_than_fp32_param_bytes():
    """INT8 (weights) + int32 (biases) must always be smaller than the
    equivalent FP32 parameter footprint for this architecture -- if this
    ever stopped being true, the "compression" claim in the report would
    be false."""
    torch.manual_seed(0)
    model = TinyKWSNet(n_mels=40, num_classes=2)
    rows = bk.per_layer_param_report(model)
    fp32_bytes = sum(r["fp32_bytes"] for r in rows)
    est = bk.estimated_int8_deployment_size_bytes(model)
    assert est["estimated_total_bytes"] < fp32_bytes


# ---------------------------------------------------------------------------
# feature_pipeline_memory_report: buffer byte-size math
# ---------------------------------------------------------------------------

def test_feature_pipeline_memory_report_buffer_sizes():
    report = bk.feature_pipeline_memory_report(sample_rate=16000, window_seconds=1.0, n_mels=40)

    assert report["window_samples"] == 16000
    assert report["audio_buffer_float32_bytes"] == 16000 * 4
    assert report["audio_buffer_int16_bytes"] == 16000 * 2

    n_frames = report["n_frames_measured"]
    assert n_frames > 0
    assert report["mel_feature_buffer_float32_bytes"] == 40 * n_frames * 4
    assert report["mel_feature_buffer_int8_estimated_bytes"] == 40 * n_frames * 1


def test_feature_pipeline_memory_report_scales_with_window_seconds():
    short = bk.feature_pipeline_memory_report(sample_rate=8000, window_seconds=0.5, n_mels=20)
    long = bk.feature_pipeline_memory_report(sample_rate=8000, window_seconds=1.0, n_mels=20)
    assert long["window_samples"] == 2 * short["window_samples"]
    assert long["audio_buffer_float32_bytes"] == 2 * short["audio_buffer_float32_bytes"]
