#!/usr/bin/env python
"""
scripts/benchmark_kws.py
===========================
Stage 2 edge-readiness benchmark for keyword_spotting.model.TinyKWSNet
and the log-Mel feature pipeline it consumes (keyword_spotting.features).

This script does NOT record audio and does NOT train anything. It only
measures the *current, untrained* TinyKWSNet architecture (random
initial weights) so its size/latency/memory profile can be assessed
before any NOVA-specific training happens. Accuracy is out of scope
here entirely -- this is a footprint/latency benchmark only.

Every number this script prints is either:
  - MEASURED  : actually observed on this machine, this run.
  - ESTIMATED : computed analytically from a documented formula/
                assumption (e.g. "if quantized to INT8 with BatchNorm
                folded, weights would occupy N bytes"). Never presented
                as if it were measured.

Usage
-----
    python scripts/benchmark_kws.py
    python scripts/benchmark_kws.py --json outputs/benchmark_kws.json
    python scripts/benchmark_kws.py --iters 200 --duration 5
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn

from keyword_spotting.features import (
    DEFAULT_HOP_MS,
    DEFAULT_N_MELS,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_WINDOW_MS,
    DEFAULT_WINDOW_SECONDS,
    extract_features,
)
from keyword_spotting.model import TinyKWSNet

try:
    import psutil
    _HAVE_PSUTIL = True
except ImportError:
    _HAVE_PSUTIL = False


BUDGET_RAM_BYTES = 256 * 1024  # SIH PS requirement: edge runtime RAM < 256 KB


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _kb(n_bytes: float) -> str:
    return f"{n_bytes / 1024.0:.2f} KB"


def _fmt_pct(x: float) -> str:
    return f"{x:.1f}%"


def _section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
# 1. Model parameter / size accounting
# ---------------------------------------------------------------------------

def per_layer_param_report(model: nn.Module) -> List[Dict[str, object]]:
    """MEASURED: parameter count and FP32 byte size per named parameter tensor."""
    rows = []
    for name, p in model.named_parameters():
        n = p.numel()
        rows.append({
            "name": name,
            "shape": list(p.shape),
            "n_params": n,
            "fp32_bytes": n * 4,
        })
    return rows


def measured_state_dict_size_bytes(model: nn.Module) -> int:
    """
    MEASURED: exact byte size of torch.save(model.state_dict()) held in
    memory (no disk I/O). This is the real FP32 checkpoint size,
    including BatchNorm buffers (running_mean/running_var/
    num_batches_tracked) that pure parameter counting would miss.
    """
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes


def fold_batchnorm_into_conv(
    conv: nn.Conv2d, bn: nn.BatchNorm2d
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Standard conv+BN fusion used by every INT8/TFLite export toolchain
    (this is exactly what the TFLite converter and most ONNX->TFLite
    tools do automatically before quantizing). Returns (folded_weight,
    folded_bias) with the SAME shape as the conv's own weight/bias --
    BatchNorm's gamma/beta/running stats disappear entirely from the
    deployed graph, they are algebraically absorbed into the conv.
    """
    with torch.no_grad():
        eps = bn.eps
        gamma = bn.weight
        beta = bn.bias
        mean = bn.running_mean
        var = bn.running_var
        std = torch.sqrt(var + eps)
        scale = gamma / std

        w = conv.weight * scale.reshape(-1, 1, 1, 1)
        b_conv = conv.bias if conv.bias is not None else torch.zeros_like(beta)
        b = beta + (b_conv - mean) * scale
    return w, b


def estimated_int8_deployment_size_bytes(model: TinyKWSNet) -> Dict[str, object]:
    """
    ESTIMATED: size of TinyKWSNet if converted to a standard INT8
    deployment graph (BatchNorm folded into the preceding Conv2d, all
    weights quantized to int8, all biases kept at int32 -- this is the
    scheme TFLite's full-integer post-training quantization uses, per
    https://www.tensorflow.org/lite/performance/post_training_quantization).

    This folds BN analytically (no export toolchain required to
    produce this estimate) but does NOT run a real quantizer, so the
    result is an estimate, not a measurement. See
    docs/STAGE_2_EDGE_BENCHMARK.md for the toolchain-measured TFLite
    number where available.
    """
    features = model.features
    conv1, bn1 = features[0], features[1]
    conv2, bn2 = features[4], features[5]
    assert isinstance(conv1, nn.Conv2d) and isinstance(bn1, nn.BatchNorm2d)
    assert isinstance(conv2, nn.Conv2d) and isinstance(bn2, nn.BatchNorm2d)

    w1, b1 = fold_batchnorm_into_conv(conv1, bn1)
    w2, b2 = fold_batchnorm_into_conv(conv2, bn2)

    linear_layers = [m for m in model.classifier if isinstance(m, nn.Linear)]

    rows = []
    total_weight_elems = 0
    total_bias_elems = 0
    for name, w, b in [("features.0+1 (conv1+bn1 folded)", w1, b1),
                        ("features.4+5 (conv2+bn2 folded)", w2, b2)]:
        rows.append({"name": name, "weight_elems": w.numel(), "bias_elems": b.numel()})
        total_weight_elems += w.numel()
        total_bias_elems += b.numel()
    for i, lin in enumerate(linear_layers):
        rows.append({
            "name": f"classifier linear[{i}] ({tuple(lin.weight.shape)})",
            "weight_elems": lin.weight.numel(),
            "bias_elems": lin.bias.numel() if lin.bias is not None else 0,
        })
        total_weight_elems += lin.weight.numel()
        total_bias_elems += lin.bias.numel() if lin.bias is not None else 0

    weight_bytes_int8 = total_weight_elems * 1
    bias_bytes_int32 = total_bias_elems * 4
    total_bytes = weight_bytes_int8 + bias_bytes_int32

    return {
        "per_layer": rows,
        "total_weight_elems": total_weight_elems,
        "total_bias_elems": total_bias_elems,
        "weight_bytes_int8": weight_bytes_int8,
        "bias_bytes_int32": bias_bytes_int32,
        "estimated_total_bytes": total_bytes,
    }


# ---------------------------------------------------------------------------
# 2. Feature pipeline (audio buffer / Mel feature) memory
# ---------------------------------------------------------------------------

def feature_pipeline_memory_report(
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    n_mels: int = DEFAULT_N_MELS,
) -> Dict[str, object]:
    """
    MEASURED (shape) + computed byte accounting for the audio/feature
    buffers the current pipeline actually allocates per inference
    window. Two audio-buffer variants are reported because the mic
    capture path (audio_processing/capture.py) and this benchmark both
    use float32, but a real firmware ADC path would typically capture
    raw int16 PCM -- both are worth knowing.
    """
    window_samples = int(round(sample_rate * window_seconds))
    y = np.zeros(window_samples, dtype=np.float32)
    feat = extract_features(y, sample_rate=sample_rate, window_seconds=window_seconds, n_mels=n_mels)
    n_frames = feat.shape[1]

    audio_buf_f32_bytes = window_samples * 4
    audio_buf_i16_bytes = window_samples * 2
    feature_buf_f32_bytes = feat.size * 4
    feature_buf_i8_bytes = feat.size * 1  # if quantized post log-mel

    return {
        "sample_rate": sample_rate,
        "window_seconds": window_seconds,
        "window_samples": window_samples,
        "n_mels": n_mels,
        "n_frames_measured": n_frames,
        "audio_buffer_float32_bytes": audio_buf_f32_bytes,
        "audio_buffer_int16_bytes": audio_buf_i16_bytes,
        "mel_feature_buffer_float32_bytes": feature_buf_f32_bytes,
        "mel_feature_buffer_int8_estimated_bytes": feature_buf_i8_bytes,
    }


# ---------------------------------------------------------------------------
# 3. Intermediate activation tensor memory (forward-hook based)
# ---------------------------------------------------------------------------

def intermediate_tensor_report(model: nn.Module, input_shape: Tuple[int, ...]) -> Dict[str, object]:
    """
    MEASURED: actual output tensor shape/dtype/byte-size of every leaf
    submodule during one real forward pass, captured via forward hooks.

    Two summary numbers are reported:
      - naive_sum_bytes: sum of every intermediate tensor's size. This
        is a pessimistic upper bound -- PyTorch eager mode does not
        reuse buffers, so this overstates what a real arena-allocator
        runtime (e.g. TFLite Micro) would need.
      - peak_single_tensor_bytes: the single largest intermediate
        tensor. A well-implemented arena allocator's true peak lies
        somewhere between this and naive_sum_bytes; the exact figure
        requires running the actual target runtime's allocator (e.g.
        TFLM's AllocationPlanner), which this script does not do.
    """
    rows: List[Dict[str, object]] = []
    hooks = []

    def make_hook(name: str):
        def hook(module, inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            if not torch.is_tensor(t):
                return
            n_bytes = t.numel() * t.element_size()
            rows.append({
                "layer": name,
                "type": type(module).__name__,
                "shape": list(t.shape),
                "dtype": str(t.dtype),
                "bytes": n_bytes,
            })
        return hook

    for name, module in model.named_modules():
        if name == "" or len(list(module.children())) > 0:
            continue  # only leaf modules
        hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        x = torch.randn(*input_shape)
        model(x)

    for h in hooks:
        h.remove()

    naive_sum = sum(r["bytes"] for r in rows)
    peak_single = max((r["bytes"] for r in rows), default=0)

    return {
        "input_shape": list(input_shape),
        "per_layer": rows,
        "naive_sum_bytes": naive_sum,
        "peak_single_tensor_bytes": peak_single,
    }


# ---------------------------------------------------------------------------
# 4. Inference latency
# ---------------------------------------------------------------------------

def latency_benchmark(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    n_warmup: int = 10,
    n_iters: int = 100,
    num_threads: Optional[int] = None,
) -> Dict[str, object]:
    """MEASURED: wall-clock per-call latency on this machine's CPU."""
    prev_threads = torch.get_num_threads()
    if num_threads is not None:
        torch.set_num_threads(num_threads)

    model.eval()
    x = torch.randn(*input_shape)
    with torch.no_grad():
        for _ in range(n_warmup):
            model(x)

        samples_ms = []
        with torch.no_grad():
            for _ in range(n_iters):
                t0 = time.perf_counter()
                model(x)
                samples_ms.append((time.perf_counter() - t0) * 1000.0)

    if num_threads is not None:
        torch.set_num_threads(prev_threads)

    samples_ms.sort()
    n = len(samples_ms)
    return {
        "num_threads": num_threads if num_threads is not None else prev_threads,
        "n_iters": n_iters,
        "mean_ms": statistics.mean(samples_ms),
        "median_ms": statistics.median(samples_ms),
        "min_ms": samples_ms[0],
        "max_ms": samples_ms[-1],
        "p95_ms": samples_ms[int(0.95 * (n - 1))],
        "p99_ms": samples_ms[int(0.99 * (n - 1))],
    }


# ---------------------------------------------------------------------------
# 5. Process memory (RSS) and CPU utilization during continuous inference
# ---------------------------------------------------------------------------

def process_memory_and_cpu_report(
    build_model_fn,
    input_shape: Tuple[int, ...],
    duration_s: float = 3.0,
) -> Dict[str, object]:
    """
    MEASURED, but explicitly a HOST-PROCESS measurement: this is the
    RSS/CPU% of the whole Python + PyTorch process running this
    benchmark, not the RAM/CPU a compiled C/C++ firmware runtime would
    use for the same model. PyTorch's own runtime overhead (operator
    dispatch, autograd bookkeeping even under no_grad, allocator
    caching) dwarfs the actual model+buffer footprint by orders of
    magnitude. Reported for completeness and to make that gap visible,
    not as an edge-hardware estimate. See docs/STAGE_2_EDGE_BENCHMARK.md.
    """
    if not _HAVE_PSUTIL:
        return {"error": "psutil not installed; process memory/CPU not measured"}

    # Pin to a single thread: a low-power edge MCU/SoC target has one
    # inference core, not this dev machine's full core count, so
    # torch's default intra-op thread pool would otherwise make the
    # CPU% reading meaningless for edge extrapolation.
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)

    proc = psutil.Process(os.getpid())

    rss_before_model = proc.memory_info().rss
    model = build_model_fn()
    rss_after_model = proc.memory_info().rss

    model.eval()
    x = torch.randn(*input_shape)
    with torch.no_grad():
        model(x)
    rss_after_first_inference = proc.memory_info().rss

    peak_rss = rss_after_first_inference
    stop_flag = threading.Event()
    lock = threading.Lock()
    iters_done = [0]

    def infer_loop():
        with torch.no_grad():
            while not stop_flag.is_set():
                model(x)
                with lock:
                    iters_done[0] += 1

    t = threading.Thread(target=infer_loop, daemon=True)
    proc.cpu_percent(interval=None)  # prime the internal counter
    t.start()

    cpu_samples = []
    t_end = time.time() + duration_s
    while time.time() < t_end:
        time.sleep(0.2)
        rss_now = proc.memory_info().rss
        peak_rss = max(peak_rss, rss_now)
        cpu_samples.append(proc.cpu_percent(interval=None))

    stop_flag.set()
    t.join(timeout=2.0)
    torch.set_num_threads(prev_threads)

    system_cpu_count = psutil.cpu_count(logical=True)
    return {
        "rss_before_model_bytes": rss_before_model,
        "rss_after_model_bytes": rss_after_model,
        "rss_after_first_inference_bytes": rss_after_first_inference,
        "peak_rss_during_continuous_inference_bytes": peak_rss,
        "model_only_rss_delta_bytes": rss_after_model - rss_before_model,
        "duration_s": duration_s,
        "iters_completed": iters_done[0],
        "throughput_inferences_per_s": iters_done[0] / duration_s,
        "cpu_percent_samples": cpu_samples,
        "cpu_percent_mean": statistics.mean(cpu_samples) if cpu_samples else None,
        "cpu_percent_max": max(cpu_samples) if cpu_samples else None,
        "logical_cpu_count": system_cpu_count,
        "torch_num_threads_during_measurement": 1,
        "note": (
            "torch.set_num_threads(1) was forced for this measurement so CPU% "
            "reflects one inference core saturated (edge-relevant), not this dev "
            "machine's full {}-core intra-op thread pool. psutil's cpu_percent is "
            "per-process; 100 percent == one full logical core saturated.".format(
                system_cpu_count,
            )
        ),
    }


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------

def build_report(n_iters: int, duration_s: float) -> Dict[str, object]:
    torch.manual_seed(0)
    model = TinyKWSNet(n_mels=DEFAULT_N_MELS, num_classes=2)
    model.eval()

    fp_report = feature_pipeline_memory_report()
    n_frames = fp_report["n_frames_measured"]
    input_shape = (1, 1, DEFAULT_N_MELS, n_frames)

    param_rows = per_layer_param_report(model)
    total_params = sum(r["n_params"] for r in param_rows)
    fp32_param_bytes = sum(r["fp32_bytes"] for r in param_rows)
    state_dict_bytes = measured_state_dict_size_bytes(model)
    int8_estimate = estimated_int8_deployment_size_bytes(model)

    intermediate = intermediate_tensor_report(model, input_shape)

    latency_default = latency_benchmark(model, input_shape, n_iters=n_iters)
    latency_single_thread = latency_benchmark(model, input_shape, n_iters=n_iters, num_threads=1)

    proc_report = process_memory_and_cpu_report(
        lambda: TinyKWSNet(n_mels=DEFAULT_N_MELS, num_classes=2), input_shape, duration_s=duration_s
    )

    edge_budget = {
        "budget_bytes": BUDGET_RAM_BYTES,
        "model_int8_estimated_bytes": int8_estimate["estimated_total_bytes"],
        "audio_buffer_int16_bytes": fp_report["audio_buffer_int16_bytes"],
        "mel_feature_buffer_int8_estimated_bytes": fp_report["mel_feature_buffer_int8_estimated_bytes"],
        "peak_activation_int8_estimated_bytes": intermediate["peak_single_tensor_bytes"] // 4,  # fp32->int8 /4 estimate
        "naive_sum_activation_int8_estimated_bytes": intermediate["naive_sum_bytes"] // 4,
    }
    edge_budget["sum_conservative_bytes"] = (
        edge_budget["model_int8_estimated_bytes"]
        + edge_budget["audio_buffer_int16_bytes"]
        + edge_budget["mel_feature_buffer_int8_estimated_bytes"]
        + edge_budget["peak_activation_int8_estimated_bytes"]
    )
    edge_budget["fits_conservative_estimate"] = edge_budget["sum_conservative_bytes"] < BUDGET_RAM_BYTES

    return {
        "model": {
            "n_mels": DEFAULT_N_MELS,
            "num_classes": 2,
            "total_params": total_params,
            "fp32_param_bytes_measured": fp32_param_bytes,
            "state_dict_bytes_measured": state_dict_bytes,
            "per_layer_params": param_rows,
            "int8_deployment_estimate": int8_estimate,
        },
        "feature_pipeline": fp_report,
        "intermediate_tensors": intermediate,
        "latency": {
            "default_threads": latency_default,
            "single_thread": latency_single_thread,
        },
        "process_memory_and_cpu": proc_report,
        "edge_ram_budget_estimate": edge_budget,
    }


def print_report(report: Dict[str, object]) -> None:
    m = report["model"]
    fp = report["feature_pipeline"]
    it = report["intermediate_tensors"]
    lat = report["latency"]
    proc = report["process_memory_and_cpu"]
    budget = report["edge_ram_budget_estimate"]

    _section("1. TinyKWSNet parameter / model size (MEASURED)")
    print(f"  Total trainable parameters : {m['total_params']:,}")
    print(f"  FP32 parameter bytes       : {_kb(m['fp32_param_bytes_measured'])} "
          f"({m['fp32_param_bytes_measured']} bytes)")
    print(f"  torch.save(state_dict) size: {_kb(m['state_dict_bytes_measured'])} "
          f"({m['state_dict_bytes_measured']} bytes) -- includes BN running stats")
    for r in m["per_layer_params"]:
        print(f"    {r['name']:<30} shape={str(r['shape']):<18} "
              f"n={r['n_params']:>6}  fp32={_kb(r['fp32_bytes'])}")

    _section("2. INT8 deployment size (ESTIMATED -- BatchNorm folded, weights int8, biases int32)")
    est = m["int8_deployment_estimate"]
    for r in est["per_layer"]:
        print(f"    {r['name']:<40} weight_elems={r['weight_elems']:>6}  bias_elems={r['bias_elems']:>4}")
    print(f"  Estimated INT8 total size  : {_kb(est['estimated_total_bytes'])} "
          f"({est['estimated_total_bytes']} bytes)")
    print(f"  Compression vs FP32 (measured/estimated ratio): "
          f"{m['fp32_param_bytes_measured'] / est['estimated_total_bytes']:.2f}x")

    _section("3. Audio / feature buffer memory (MEASURED shapes)")
    print(f"  Sample rate                : {fp['sample_rate']} Hz")
    print(f"  Analysis window            : {fp['window_seconds']} s ({fp['window_samples']} samples)")
    print(f"  Mel bins x frames          : {fp['n_mels']} x {fp['n_frames_measured']}")
    print(f"  Audio buffer (float32)     : {_kb(fp['audio_buffer_float32_bytes'])}")
    print(f"  Audio buffer (int16 PCM)   : {_kb(fp['audio_buffer_int16_bytes'])}")
    print(f"  Mel feature buffer (fp32)  : {_kb(fp['mel_feature_buffer_float32_bytes'])}")
    print(f"  Mel feature buffer (int8, estimated): {_kb(fp['mel_feature_buffer_int8_estimated_bytes'])}")

    _section("4. Intermediate activation tensors (MEASURED via forward hooks, FP32 eager mode)")
    for r in it["per_layer"]:
        print(f"    {r['layer']:<20} {r['type']:<14} shape={str(r['shape']):<20} {_kb(r['bytes'])}")
    print(f"  Naive sum of all intermediate tensors: {_kb(it['naive_sum_bytes'])} "
          f"(pessimistic upper bound; PyTorch eager does not reuse buffers)")
    print(f"  Peak single tensor                   : {_kb(it['peak_single_tensor_bytes'])} "
          f"(a real arena allocator's true peak lies between these two figures)")

    _section("5. Inference latency (MEASURED, this dev machine, CPU)")
    for label, d in (("default thread count", lat["default_threads"]), ("single thread", lat["single_thread"])):
        print(f"  [{label}, torch.threads={d['num_threads']}, n={d['n_iters']}]")
        print(f"    mean={d['mean_ms']:.3f} ms  median={d['median_ms']:.3f} ms  "
              f"p95={d['p95_ms']:.3f} ms  p99={d['p99_ms']:.3f} ms  "
              f"min={d['min_ms']:.3f} ms  max={d['max_ms']:.3f} ms")

    _section("6. Host-process RSS / CPU during continuous inference (MEASURED, NOT edge-representative)")
    if "error" in proc:
        print(f"  {proc['error']}")
    else:
        print(f"  RSS before model construction        : {_kb(proc['rss_before_model_bytes'])}")
        print(f"  RSS after model construction         : {_kb(proc['rss_after_model_bytes'])}")
        print(f"  Model construction RSS delta          : {_kb(proc['model_only_rss_delta_bytes'])}")
        print(f"  RSS after first inference             : {_kb(proc['rss_after_first_inference_bytes'])}")
        print(f"  Peak RSS during {proc['duration_s']}s continuous inference: {_kb(proc['peak_rss_during_continuous_inference_bytes'])}")
        print(f"  Inferences completed / throughput     : {proc['iters_completed']} "
              f"({proc['throughput_inferences_per_s']:.1f}/s)")
        print(f"  CPU% mean / max (this process)        : "
              f"{_fmt_pct(proc['cpu_percent_mean'])} / {_fmt_pct(proc['cpu_percent_max'])} "
              f"(of {proc['logical_cpu_count']} logical cores; 100%=1 core)")
        print(f"  {proc['note']}")
        print("  NOTE: this includes the full CPython + PyTorch process (interpreter,")
        print("  operator dispatch, allocator). It is NOT a proxy for compiled-firmware")
        print("  RAM/CPU on target edge hardware -- see docs/STAGE_2_EDGE_BENCHMARK.md.")

    _section("7. Prototype edge RAM budget (ESTIMATED, conservative worst case)")
    print(f"  Requirement                          : < {_kb(budget['budget_bytes'])}")
    print(f"  Model weights (INT8, estimated)      : {_kb(budget['model_int8_estimated_bytes'])}")
    print(f"  Audio buffer (int16 PCM)             : {_kb(budget['audio_buffer_int16_bytes'])}")
    print(f"  Mel feature buffer (int8, estimated) : {_kb(budget['mel_feature_buffer_int8_estimated_bytes'])}")
    print(f"  Peak activation tensor (int8, estimated, /4 from fp32): {_kb(budget['peak_activation_int8_estimated_bytes'])}")
    print(f"  Conservative sum                     : {_kb(budget['sum_conservative_bytes'])}")
    print(f"  Fits under {_kb(budget['budget_bytes'])}?              : {budget['fits_conservative_estimate']}")
    print("  This is a hand-computed, worst-case-additive estimate, NOT a measurement")
    print("  from a real embedded allocator/interpreter arena. See")
    print("  docs/STAGE_2_EDGE_BENCHMARK.md section 7 for full methodology and caveats.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=100, help="Latency benchmark iterations")
    parser.add_argument("--duration", type=float, default=3.0, help="Continuous-inference sampling window (s)")
    parser.add_argument("--json", type=str, default=None, help="Optional path to write the full report as JSON")
    args = parser.parse_args()

    report = build_report(n_iters=args.iters, duration_s=args.duration)
    print_report(report)

    if args.json:
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nFull report written to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
