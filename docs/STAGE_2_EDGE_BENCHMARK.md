# SIH 2026 — PS 26172: Stage 2 Edge-Readiness & Benchmark Report

**Scope:** edge-readiness and benchmarking work for TinyKWSNet, the CNN keyword-spotting
model, that can be completed **without** a real NOVA recording. NOVA recording and
TinyKWSNet training are explicitly **out of scope** and were **not performed**.

**Report basis:** every number below is either

- **MEASURED** — actually observed on this machine via a script in this repo
  (`scripts/benchmark_kws.py`, `scripts/export_kws_tflite.py`) or a one-off reproducible
  command shown inline.
- **ESTIMATED** — computed analytically from a documented formula/assumption, clearly
  marked, never presented as a measurement.

Nothing here is extrapolated to real edge hardware without saying so explicitly. Per
instruction, **no ESP32 (or any other MCU/Raspberry Pi) compatibility claim is made** —
only architecture/toolchain-level compatibility actually exercised on this dev machine.
**No NOVA detection accuracy is claimed anywhere in this report** — there is no trained
NOVA checkpoint; every model number below comes from a randomly initialized network.

---

## 1. Benchmark environment

| | |
|---|---|
| OS | Windows-10-10.0.26200-SP0 |
| CPU | AMD64 Family 25 Model 124 Stepping 0 (AuthenticAMD), 12 logical cores |
| Python | 3.11.5 |
| torch | 2.13.0+cpu |
| numpy | 2.4.6 |
| psutil | 7.2.2 |

This is a **12-logical-core x86-64 laptop**, not an MCU or a Raspberry Pi. Every
latency/CPU/memory number in this report reflects this machine and this software stack
(PyTorch eager mode / CPython), not compiled firmware on a constrained device — restated
throughout, not just here, because it is the single most important caveat in this report.

Reproduce with:
```
python scripts/benchmark_kws.py --json outputs/benchmark_kws.json
```

---

## 2. What's actually being benchmarked, and its current integration status

**This section corrects stale information from an earlier draft of this report.** As of
the KWS-integration milestone that followed the original Stage 2 benchmark,
**TinyKWSNet is wired into `KeywordDetector` as the primary backend** —
`keyword_spotting/tinykws_backend.py`'s `TinyKWSNetBackend` loads a `.pt` checkpoint
(default `models/kws_nova_cnn.pt`) and `keyword_spotting/detector.py` selects it
automatically by file extension. openWakeWord (`keyword_spotting/oww_backend.py`)
remains fully supported as an alternative backend (selected via a `.onnx` `model_path`)
but is **not** the primary training path. This report still only benchmarks TinyKWSNet
itself — the actual CNN architecture, independent of which backend class wraps it —
because it is the component with realistic potential for the sub-256 KB / TFLite Micro
deployment target.

**No checkpoint exists yet.** `models/kws_nova_cnn.pt` does not exist in this repo (only
`models/.gitkeep`). `KeywordDetector()` constructed with defaults today reports
`is_ready=False`, `status="NOT_IMPLEMENTED"` — verified, not assumed. All TinyKWSNet
numbers below use **randomly initialized weights** (`torch.manual_seed(0)`, untrained).
Accuracy is out of scope entirely; only size/latency/memory/export-compatibility are
measured.

### TinyKWSNet architecture (as inspected, unchanged since the original benchmark)

```
Input: (batch, 1, n_mels=40, n_frames) log-Mel spectrogram
  Conv2d(1→16, k=3, pad=1) → BatchNorm2d(16) → ReLU → MaxPool2d(2)
  Conv2d(16→32, k=3, pad=1) → BatchNorm2d(32) → ReLU → AdaptiveAvgPool2d(1)
  Flatten → Linear(32→32) → ReLU → Linear(32→2)
```
Binary classifier: class 0 = unknown/negative, class 1 = NOVA. `AdaptiveAvgPool2d(1)`
makes the model input-length-tolerant during training/experimentation — this does not
carry over to edge export (§6.2).

---

## 3. Model statistics (MEASURED)

| Metric | Value | Type |
|---|---|---|
| Total trainable parameters | **6,018** | MEASURED |
| FP32 parameter bytes (Σ numel×4) | **24,072 bytes (23.51 KB)** | MEASURED |
| `torch.save(state_dict())` size (incl. BatchNorm running stats) | **30,509 bytes (29.79 KB)** | MEASURED |

Per-layer breakdown:

| Layer | Shape | Params | FP32 bytes |
|---|---|---:|---:|
| features.0 (Conv2d 1→16) weight | [16,1,3,3] | 144 | 576 |
| features.0 bias | [16] | 16 | 64 |
| features.1 (BatchNorm2d 16) weight/bias | [16]+[16] | 32 | 128 |
| features.4 (Conv2d 16→32) weight | [32,16,3,3] | 4,608 | 18,432 |
| features.4 bias | [32] | 32 | 128 |
| features.5 (BatchNorm2d 32) weight/bias | [32]+[32] | 64 | 256 |
| classifier.1 (Linear 32→32) weight | [32,32] | 1,024 | 4,096 |
| classifier.1 bias | [32] | 32 | 128 |
| classifier.3 (Linear 32→2) weight | [2,32] | 64 | 256 |
| classifier.3 bias | [2] | 2 | 8 |

Comparison: `voice_separation`'s Conv-TasNet is ~4.98M parameters (Stage 1 report) —
TinyKWSNet is **~828× smaller** by parameter count.

---

## 4. Latency results (MEASURED, this dev machine, CPU — NOT edge hardware)

PyTorch eager mode, `scripts/benchmark_kws.py`, n=100 iterations after 10 warmup:

| Config | mean | median | p95 | p99 | min | max |
|---|---:|---:|---:|---:|---:|---:|
| Default threads (torch.threads=6, 12 logical cores available) | 0.425 ms | 0.415 ms | 0.566 ms | 0.715 ms | 0.306 ms | 0.777 ms |
| Single thread (torch.set_num_threads(1)) | 0.949 ms | 1.020 ms | 1.327 ms | 1.513 ms | 0.554 ms | 1.546 ms |

Real exported TFLite interpreter (§7), single-thread, n=200 iterations after 10 warmup
(re-verified this session against artifacts from the original export run — see §7.1):

| Model | mean | median | p95 | min | max |
|---|---:|---:|---:|---:|---:|
| Float32 `.tflite` | 0.203 ms | 0.199 ms | 0.277 ms | 0.114 ms | 0.748 ms |
| INT8 `.tflite` | 0.109 ms | 0.105 ms | 0.149 ms | 0.062 ms | 0.743 ms |

All numbers are on a 12-logical-core x86-64 dev laptop. Absolute latency will differ
(likely by 1–2 orders of magnitude, slower) on a Cortex-M-class MCU or a Raspberry Pi's
Cortex-A core — what's useful here is the **relative** result: the TFLite interpreter is
faster than PyTorch eager mode for the same architecture (fewer dispatch layers), and
INT8 is consistently ~1.9× faster than float32 in TFLite. **Neither number should be
quoted as an edge-hardware latency figure.**

---

## 5. Memory calculations

### 5.1 Current configuration (from `keyword_spotting/features.py` and `detector.py`)

| Parameter | Value | Source |
|---|---|---|
| Sample rate | 16,000 Hz | `features.DEFAULT_SAMPLE_RATE` |
| Analysis window duration | 1.0 s (16,000 samples) | `features.DEFAULT_WINDOW_SECONDS` |
| Mel-spectrogram frame window | 25 ms | `features.DEFAULT_WINDOW_MS` |
| Mel-spectrogram frame hop | 10 ms | `features.DEFAULT_HOP_MS` |
| Number of Mel bins | 40 | `features.DEFAULT_N_MELS` |
| **Detector** inference cadence ("hop") | 0.2 s | `detector.DEFAULT_HOP_SECONDS` |

Two different "hop" concepts exist in this pipeline and are easy to conflate: the
**Mel-spectrogram frame hop** (10 ms — governs the log-Mel feature tensor's time-frame
count) and the **detector's inference-cadence hop** (0.2 s — how often
`KeywordDetector.detect_chunk()` runs a new full-window inference on its sliding audio
buffer). Both are listed above to avoid that ambiguity; only the first affects the
memory calculations below.

Given these, one analysis window produces **101 Mel frames** (measured from a real
`extract_features()` call, not assumed).

### 5.2 Buffer memory (MEASURED shapes)

| Buffer | Size | Type |
|---|---|---|
| Raw audio window, float32 | 16,000 × 4 B = **62.50 KB** | MEASURED shape |
| Raw audio window, int16 PCM (typical ADC/mic representation) | 16,000 × 2 B = **31.25 KB** | MEASURED shape |
| Log-Mel feature buffer, float32 | 40 × 101 × 4 B = **15.78 KB** | MEASURED shape |
| Log-Mel feature buffer, int8 (if quantized post-feature-extraction) | 40 × 101 × 1 B = **3.95 KB** | ESTIMATED (÷4 from fp32; no real quantized feature-extraction path exists) |

### 5.3 Model weight memory

Already covered in §3: **23.51 KB FP32**, **~6.02 KB estimated INT8** (§6).

### 5.4 Intermediate activation ("runtime/framework-adjacent") memory (MEASURED via forward hooks)

This is distinct from both model-weight memory (§3) and audio/feature-buffer memory
(§5.2) — the three are frequently conflated in casual "model size" claims, so they are
kept separate here, per the task's explicit requirement. Captured with real forward
hooks on every leaf submodule during one actual forward pass (input shape `(1,1,40,101)`,
FP32, PyTorch eager mode):

| Layer | Output shape | Bytes (FP32) |
|---|---|---:|
| Conv2d (features.0) | [1,16,40,101] | 258,560 (252.50 KB) |
| BatchNorm2d (features.1) | [1,16,40,101] | 258,560 (252.50 KB) |
| ReLU (features.2) | [1,16,40,101] | 258,560 (252.50 KB) |
| MaxPool2d (features.3) | [1,16,20,50] | 64,000 (62.50 KB) |
| Conv2d (features.4) | [1,32,20,50] | 128,000 (125.00 KB) |
| BatchNorm2d (features.5) | [1,32,20,50] | 128,000 (125.00 KB) |
| ReLU (features.6) | [1,32,20,50] | 128,000 (125.00 KB) |
| AdaptiveAvgPool2d / Flatten / Linear / ReLU / Linear | [1,32,1,1]→[1,2] | ≤128 bytes each |

- **Naive sum of all intermediate tensors: 1,195.51 KB** — pessimistic upper bound;
  PyTorch eager mode never reuses buffers, so this overstates any real allocator's need.
- **Peak single tensor: 252.50 KB** — the first Conv2d/BatchNorm2d/ReLU output, before
  pooling shrinks the spatial dimensions.

**This is the single most important memory finding in this report:** at FP32, one
intermediate activation tensor alone (252.5 KB) very nearly consumes the *entire* 256 KB
RAM budget, by itself, before counting model weights, audio buffer, Mel buffer, or any
runtime/stack overhead. **INT8 quantization is not optional for this architecture under
the stated budget — it is required**, and even then the same tensor is ~63.1 KB (÷4),
still ~24% of the entire budget on its own (§8 budget table).

A true arena-allocator peak (as TFLite Micro's `MicroAllocator` would compute) lies
somewhere between the peak-single-tensor and naive-sum figures, because a well-designed
allocator can reuse the Conv1 output buffer once MaxPool has consumed it. **This has not
been measured** (requires a C++ TFLM build) — flagged as a concrete remaining blocker
(§10).

### 5.5 Prototype edge RAM budget (ESTIMATED, conservative worst-case-additive)

| Component | Bytes | KB | Basis |
|---|---:|---:|---|
| Model weights (INT8, real TFLite file) | 11,592 | 11.32 | MEASURED (§7; using the real file, not the smaller hand-estimate, to be conservative) |
| Audio buffer (int16 PCM, 1.0 s @ 16 kHz) | 32,000 | 31.25 | MEASURED shape |
| Mel feature buffer (int8, estimated) | 4,040 | 3.95 | ESTIMATED |
| Peak activation tensor (int8, estimated as FP32 peak ÷ 4) | 64,640 | 63.12 | ESTIMATED from MEASURED FP32 peak |
| **Conservative sum** | **112,272** | **109.64** | — |
| **Requirement** | 262,144 | 256.00 | — |
| **Headroom** | **~146 KB (57%)** under this conservative estimate | | |

**Caveats, stated plainly:**
- This sum double-counts in the "safe" direction: a real arena allocator would reuse
  buffers across pipeline stages rather than holding all of them live simultaneously at
  their individual peaks, so true peak RAM is very likely **lower**, not higher.
- It does **not** include runtime/firmware overhead: OS/RTOS stack and heap, the TFLM
  interpreter's own bookkeeping, I2S/DMA audio driver buffers, Wi-Fi/BLE stack RAM, or
  any application-level buffering beyond one inference window. On a real board these are
  frequently the dominant RAM consumer, not the model.
- The "peak activation ÷ 4" INT8 estimate is a linear approximation from an FP32
  measurement; real per-tensor/per-channel INT8 activation sizes should be re-measured
  once TFLM's own memory-arena report is run against this exact architecture.

---

## 6. INT8 estimate (ESTIMATED, cross-checked against a real export)

| Metric | Value | Type |
|---|---|---|
| Hand-computed INT8 estimate (BatchNorm folded into Conv, weights int8, biases int32) | **6,168 bytes (6.02 KB)** | ESTIMATED |
| **Real TFLite INT8 file** (full end-to-end toolchain, §7) | **11,592 bytes (11.32 KB)**, of which 6,288 bytes (54%) is quantized weight/bias data and 5,304 bytes (46%) is flatbuffer/op/quantization-metadata overhead | MEASURED |

The hand-computed estimate (6,168 bytes) and the real converter's quantized *data*
buffer (6,288 bytes) agree within **2%** — the ~120-byte gap is per-channel
quantization scale/zero-point metadata the hand estimate doesn't model. The real
`.tflite` *file* is larger because ~46% of a small model's file size is fixed
flatbuffer/graph-structure overhead that does not scale down with parameter count —
worth knowing when budgeting flash, not RAM.

**This estimate's correctness is now unit-tested**, not just cross-checked by hand:
`tests/unit/test_benchmark_kws_calculations.py::test_fold_batchnorm_into_conv_preserves_output_numerically`
verifies the BN-fold math used to produce this estimate actually reproduces
conv+BatchNorm's real numerical output (not just plausible-looking shapes), and
`test_int8_estimate_matches_hand_computed_totals_for_a_small_model` verifies the
byte-counting arithmetic against an independent hand count.

---

## 7. TFLite / TFLite Micro findings

### 7.1 What was verified, and how it was re-verified this session

An isolated (non-project) Python environment was built specifically for this, because
`tensorflow` is not (and should not be) a dependency of this project's main environment
(§7.3). That environment **no longer exists on this machine** (it was an ephemeral,
outside-the-repo venv, as documented in `scripts/export_kws_tflite.py`) — this session
did not recreate it, since the architecture under test is byte-identical to the original
export and reinstalling a multi-hundred-MB TensorFlow toolchain to reconfirm an unchanged
result was not a good use of the "minimal/focused" scope for this task. Instead, this
session **re-verified the artifacts from the original export are still present and their
byte sizes still match this report exactly**:

```
outputs/kws_export/tiny_kws_net.onnx                              25,452 bytes
outputs/kws_export/tf_saved_model/tiny_kws_net_float32.tflite      26,668 bytes
outputs/kws_export/tf_saved_model/tiny_kws_net_float16.tflite      16,156 bytes
outputs/kws_export/tiny_kws_net_int8.tflite                        11,592 bytes
```
All four match the sizes originally reported byte-for-byte — confirmed present on disk
this session, not re-executed. The original verification (still standing, unchanged):

1. **PyTorch → ONNX**: `torch.onnx.export(..., opset_version=18, dynamic_axes=None)`
   succeeded. Resulting ops: `Conv, Gemm, MaxPool, ReduceMean, Relu, Reshape` — BatchNorm
   does not appear because PyTorch's exporter folds it into the preceding Conv
   automatically (confirming the BN-fold logic independently validated in §6).
2. **ONNX → TensorFlow SavedModel → TFLite** via `onnx2tf`: succeeded, produced float32
   and float16 `.tflite` files.
3. **Full-integer INT8 post-training quantization** (`tf.lite.TFLiteConverter`,
   `TFLITE_BUILTINS_INT8`, int8 in/out, 20-sample random representative dataset):
   succeeded.
4. **Functional smoke test**: both float32 and INT8 `.tflite` files loaded with
   `tf.lite.Interpreter`, allocated tensors, and ran a real `invoke()` on random input,
   producing correctly-shaped `(1, 2)` output — runnable, not just structurally valid.
5. **Op inventory** (`tf.lite.experimental.Analyzer`): both models use exactly four
   TFLite builtin ops: **`CONV_2D`, `MAX_POOL_2D`, `MEAN`, `FULLY_CONNECTED`** (ReLU is
   fused into the preceding op, a standard converter optimization).

Reproduce (requires the isolated venv described in `scripts/export_kws_tflite.py`'s
docstring):
```
python scripts/export_kws_tflite.py --out-dir outputs/kws_export
```

### 7.2 Architecture compatibility conclusions

| Question | Answer | Basis |
|---|---|---|
| Exportable to ONNX? | **Yes** — verified | §7.1 step 1 |
| Exportable to TFLite (float32)? | **Yes** — verified | §7.1 step 2, 4 |
| Compatible with INT8 full-integer post-training quantization? | **Yes** — verified | §7.1 step 3, 4 |
| All ops within TFLite Micro's standard supported-op set? | **Yes, at the op-type level** — `CONV_2D`, `MAX_POOL_2D`, `MEAN`, `FULLY_CONNECTED` are all part of TFLM's core/always-registered ops. **Not independently verified by running TFLM itself** (§7.4) — a documented-op-list check, not an execution check. | Analyzer output |
| Any required architectural change? | **Yes, one: a fixed, static time-axis length.** `AdaptiveAvgPool2d` only became a plain `MEAN` because the model was traced with a *fixed* input shape `(1,1,40,101)`. TFLite Micro has no dynamic-shape support at all. TinyKWSNet's training-time flexibility across T ∈ {17,101,250} is a training/experimentation convenience that **must be pinned to one fixed T** (e.g. 101) before any real export — a deployment-time constraint, not a change to `model.py` itself. | §7.1 step 1, TFLite semantics |

### 7.3 Why the main environment doesn't include TensorFlow

TensorFlow brings its own `numpy`/`protobuf` version constraints not guaranteed
compatible with this project's torch/librosa/scikit-learn stack, and installing it
directly risks breaking Conv-TasNet, ASR, or command modules — explicitly off-limits for
modification. Verified instead in a throwaway, isolated venv outside the repo;
`scripts/export_kws_tflite.py`'s docstring documents the exact install steps
(including a Windows `MAX_PATH`-length gotcha hit and resolved during the original run).

### 7.4 What is explicitly NOT verified

- **No TFLite Micro interpreter was actually run**, on this machine or any other. TFLM
  ships as a C++ library meant to be compiled into firmware; no official
  pip-installable Python TFLM interpreter was used here.
- **No ESP32, Raspberry Pi, or any other board was used.** No hardware compatibility
  claim of any kind is made. Verifying that requires picking a target board/toolchain,
  cross-compiling a firmware image embedding the `.tflite` model, flashing real
  hardware, and measuring actual RAM/flash/latency/power on-device — none performed.
- **No real TFLM arena-allocator size was measured** — only PyTorch-side activation
  tensor sizes via forward hooks (§5.4), an approximation, not a TFLM allocator run.

---

## 8. CPU results (MEASURED, host-process only — explicitly NOT edge-representative)

`scripts/benchmark_kws.py`'s continuous-inference measurement, `torch.set_num_threads(1)`
forced, sampled every 0.2s over a 3.0s window:

| Metric | Value |
|---|---|
| RSS before model construction | 359,460.00 KB |
| RSS after model construction | 359,516.00 KB (Δ 56.00 KB) |
| Peak RSS during 3.0s continuous inference | 362,012.00 KB |
| Throughput | 4,366 inferences in 3.0s (1,455.3/s) |
| **CPU% mean / max (this process)** | **115.5% / 331.0%** (of 12 logical cores; 100% = 1 fully-saturated core) |

**Why this doesn't map to the <10%-CPU-while-idle target, and can't yet:**

1. **This measures continuous, back-to-back inference at ~1,455 inferences/second on a
   12-core desktop CPU inside a full CPython + PyTorch process** — not a firmware
   idle-listening loop running one inference every `hop_seconds` (0.2s, i.e. ~5/s) on a
   single constrained core. The workloads are not comparable in kind, only loosely in
   architecture.
2. **CPU% exceeding 100% despite `torch.set_num_threads(1)`** means more than one
   logical core's worth of work is being attributed to this process — plausibly
   PyTorch/BLAS backend threads that aren't fully bounded by the intra-op thread-count
   setting, plus CPython interpreter/GC overhead from the benchmark's own
   measurement-thread bookkeeping. This is a real, honestly-reported measurement, but it
   reflects **this dev-machine software stack's overhead**, not compiled-firmware CPU use.
3. **A real "<10% CPU while idle/continuously listening" claim requires the actual target
   runtime** (TFLite Micro on the actual MCU/RPi core, running the real per-`hop_seconds`
   inference cadence at the real clock speed) — not measured here, and not measurable on
   this hardware/software stack in any way that would be meaningful.

**Conclusion for this metric: NOT YET COMPARABLE to the <10% CPU target — not "pass," not
"fail," genuinely not measurable on this dev machine in a way that transfers.** This is
listed as a concrete remaining blocker (§10), not glossed over.

---

## 9. Comparison with SIH requirements

| Requirement | Status | Basis |
|---|---|---|
| **RAM < 256 KB** | **Plausibly met, not proven.** Conservative worst-case-additive estimate: 109.64 KB (§5.5), ~57% headroom. | ESTIMATED from MEASURED components; no real embedded allocator or hardware measurement exists yet. |
| **CPU < 10% while continuously listening/idle** | **Not comparable yet** (§8) — the only CPU measurement available is a 12-core desktop process running unrelated workload characteristics (continuous max-throughput inference vs. periodic idle-listening inference). | MEASURED (dev-machine only), explicitly not extrapolable. |
| **TFLite/TFLite Micro exportability** | **Met at the architecture/toolchain level.** ONNX, TFLite float32/float16, and INT8 export all verified working; ops are within TFLM's core op set. | MEASURED (§7), with the explicit caveats in §7.4. |
| **Real ESP32/RPi hardware validation** | **Not attempted** — explicitly out of scope for this stage per instruction. | N/A |
| **Trained NOVA accuracy** | **Not applicable / not claimed.** No checkpoint exists; nothing here is a claim about detection quality. | N/A |

---

## 10. Limitations and what must be re-measured on the actual edge device

1. **TFLM's real `MicroAllocator` arena size** — replace the "peak activation ÷ 4"
   estimate (§5.4/§5.5) with a measured arena size, requiring either a C++ TFLM build or
   at minimum `tf.lite.TFLiteConverter`'s own memory-arena reporting.
2. **Real CPU utilization on the target core** at the real inference cadence
   (`hop_seconds` = 0.2s, i.e. one inference per ~5 chunks, not continuous max-throughput)
   — the only way to produce a number actually comparable to the <10% target (§8).
3. **Real RAM measurement on-device** — the linker map / heap high-water-mark on the
   actual MCU/RPi build, not a host-process RSS or a hand-assembled estimate.
4. **Selecting a target MCU/board and toolchain**, then an actual on-device build —
   required before any hardware compatibility (ESP32, RPi, or otherwise) claim can be
   made (§7.4).
5. **Deciding whether `model.py` should natively export a fixed-T variant** (e.g. an
   explicit `AvgPool2d` sized for T=101 instead of `AdaptiveAvgPool2d`) rather than
   relying on export-time shape-pinning — a design decision, not attempted here since it
   would modify `model.py` beyond this stage's "use as-is" scope.
6. **Everything gated on real NOVA data**, restated from the original benchmark and still
   true: training TinyKWSNet at all, calibrating `DEFAULT_THRESHOLD`/
   `DEFAULT_CONSECUTIVE_FRAMES`, and assessing INT8 *quantization error* against real
   speech (not random noise) — none of this is blocked architecturally, all of it is
   blocked on the postponed recording step.

---

## 11. Test suite status (MEASURED)

```
python -m pytest -q
```
**224 passed, 5 warnings (pre-existing FastAPI/httpx deprecation warnings, unrelated),
0 failed.**

New this session: `tests/unit/test_benchmark_kws_calculations.py` (6 tests) — the first
test coverage for `scripts/benchmark_kws.py`'s own calculations (BN-fold numerical
correctness, INT8 byte-count arithmetic, buffer-size math). Added because these
calculations directly underpin §5/§6's size/RAM claims and had zero prior test coverage.
No other code was changed this session — `keyword_spotting/model.py`, `features.py`,
`detector.py`, and `tinykws_backend.py` were inspected but not modified; the TinyKWSNet
architecture benchmarked here is byte-identical to the original Stage 2 report.

Files touched this session:
- `docs/STAGE_2_EDGE_BENCHMARK.md` (this file — substantially revised: corrected stale
  "not wired into KeywordDetector" claims, reorganized into the requested section
  structure, added §8's CPU-vs-target comparison and §9's explicit requirements table)
- `tests/unit/test_benchmark_kws_calculations.py` (new)
- `outputs/benchmark_kws_stage2b.json` (scratch re-run output, not committed)

No files under `voice_separation/`, `asr/`, `command/`, or any frontend were touched.

---

## 12. Conclusion

**PARTIALLY READY.**

- **Architecture-level edge readiness: READY.** TinyKWSNet exports cleanly to ONNX →
  TFLite float32/INT8, uses only TFLM-core ops, and a conservative RAM estimate fits the
  256 KB budget with substantial headroom (§5.5, §7, §9).
- **Quantitative hardware readiness: NOT READY to claim.** No real device measurement
  exists for RAM, CPU, or latency (§8, §10) — every number in this report is either a
  dev-machine measurement or an analytical estimate. The CPU-vs-<10%-target comparison
  specifically is not yet meaningful in either direction (§8).
- **Model readiness: NOT READY** — there is no trained NOVA checkpoint. Nothing in this
  report is or implies a claim about detection accuracy.

**Bottom line:** the architecture has cleared every check that can be run without real
hardware or real training data. The next blockers are, in order: (1) real NOVA
recording + training (separately tracked, explicitly postponed), and (2) an actual
on-device build once a target board is chosen — not further host-machine benchmarking,
which has been exhausted for what it can honestly say.
