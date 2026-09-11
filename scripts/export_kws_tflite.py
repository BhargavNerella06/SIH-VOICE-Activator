#!/usr/bin/env python
"""
scripts/export_kws_tflite.py
================================
Stage 2 edge-readiness: export keyword_spotting.model.TinyKWSNet to
ONNX -> TensorFlow SavedModel -> TFLite (float32 and INT8), and report
real measured file sizes.

This does NOT train anything. By default it exports the model with its
current (randomly initialized, or loaded-from-checkpoint if
--checkpoint is given) weights, purely to verify the export toolchain
and measure real artifact sizes. If no NOVA-trained checkpoint exists
yet, the produced .tflite files are NOT usable for real detection --
they only prove the architecture converts and quantizes cleanly.

Requirements
------------
This script needs `onnx`, `tensorflow`, and `onnx2tf` (plus onnx2tf's
own dependencies: tf_keras, ai-edge-litert, onnxscript, psutil). These
are NOT part of requirements.txt/requirements-dev.txt on purpose --
tensorflow pulls in its own numpy/protobuf pins that can fight with
the torch/librosa stack this project's main environment already uses.
Verified (this Stage) in an ISOLATED venv:

    python -m venv .venv-tflite-export
    .venv-tflite-export/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cpu
    .venv-tflite-export/Scripts/pip install onnx tensorflow onnx2tf onnx_graphsurgeon sng4onnx onnxsim tf_keras ai-edge-litert onnxscript psutil
    .venv-tflite-export/Scripts/python scripts/export_kws_tflite.py

On Windows, create that venv at a SHORT path (e.g. C:\\tmp\\...), not
under a deeply-nested temp/session directory -- onnx's and pkg_resources'
own test-data files hit Windows' ~260-char MAX_PATH limit otherwise
(observed directly during Stage 2: pip install failed with
"WinError 206: filename or extension is too long" until the venv was
moved to C:\\tmp\\tflite_test).

Usage
-----
    python scripts/export_kws_tflite.py --out-dir outputs/kws_export
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from keyword_spotting.features import DEFAULT_N_MELS
from keyword_spotting.model import TinyKWSNet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=str, default="outputs/kws_export")
    parser.add_argument("--checkpoint", type=str, default=None,
                         help="Optional path to a trained TinyKWSNet state_dict (.pt). "
                              "Without this, the exported model has random weights and "
                              "is only useful for toolchain/shape/quantization verification.")
    parser.add_argument("--n-mels", type=int, default=DEFAULT_N_MELS)
    parser.add_argument("--n-frames", type=int, default=101,
                         help="Fixed time-axis length for export. TFLite / TFLite Micro "
                              "require static shapes, unlike TinyKWSNet's training-time "
                              "AdaptiveAvgPool2d flexibility -- see docs/STAGE_2_EDGE_BENCHMARK.md.")
    args = parser.parse_args()

    try:
        import onnx
        import onnx2tf
        import tensorflow as tf
    except ImportError as e:
        print(f"Missing export dependency: {e}")
        print("This script requires an isolated venv with onnx/tensorflow/onnx2tf.")
        print("See the module docstring (python -c \"import scripts.export_kws_tflite\") "
              "or docs/STAGE_2_EDGE_BENCHMARK.md section 5 for exact install commands.")
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    model = TinyKWSNet(n_mels=args.n_mels, num_classes=2)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state)
        print(f"Loaded trained weights from {args.checkpoint}")
    else:
        print("No --checkpoint given: exporting with random weights "
              "(toolchain/shape verification only, not a usable detector).")
    model.eval()

    dummy = torch.randn(1, 1, args.n_mels, args.n_frames)

    onnx_path = out_dir / "tiny_kws_net.onnx"
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["log_mel"], output_names=["logits"],
        opset_version=18, dynamic_axes=None,
    )
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    ops_used = sorted({n.op_type for n in onnx_model.graph.node})
    print(f"ONNX export OK: {onnx_path} ({onnx_path.stat().st_size} bytes); ops used: {ops_used}")

    tf_dir = out_dir / "tf_saved_model"
    onnx2tf.convert(
        input_onnx_file_path=str(onnx_path),
        output_folder_path=str(tf_dir),
        copy_onnx_input_output_names_to_tflite=True,
        non_verbose=True,
    )
    fp32_candidates = sorted(tf_dir.glob("*float32.tflite"))
    if not fp32_candidates:
        print("onnx2tf did not produce a float32 .tflite file; aborting INT8 step.")
        return 1
    fp32_path = fp32_candidates[0]
    print(f"Float32 TFLite OK: {fp32_path} ({fp32_path.stat().st_size} bytes)")

    def representative_dataset():
        rng = np.random.RandomState(0)
        for _ in range(20):
            yield [rng.randn(1, args.n_mels, args.n_frames, 1).astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_int8 = converter.convert()
    int8_path = out_dir / "tiny_kws_net_int8.tflite"
    int8_path.write_bytes(tflite_int8)
    print(f"INT8 TFLite OK: {int8_path} ({int8_path.stat().st_size} bytes)")

    print("\nDone. NOTE: these models have random (or checkpoint-loaded) weights only -- "
          "they verify the export pipeline, not detection accuracy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
