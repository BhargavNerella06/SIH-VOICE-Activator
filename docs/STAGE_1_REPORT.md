# SIH 2026 — PS 26172: Stage-1 Progress Report

**Project:** Low Latency and Efficient Voice Activator for Edge Devices
**Repository:** SIH-VOICE-ACTIVATOR
**Report basis:** Direct inspection of the current working tree, test runs, and runtime probes performed during this session. No numbers below are invented.

---

## 1. Project Objective vs. Current Implementation

PS 26172 asks for a **low-latency, resource-efficient voice activation pipeline for edge devices**: capture audio → isolate the target speaker from overlapping/noisy speech → detect a wake word → transcribe the command → act on it, all with edge-appropriate footprint and latency.

**Current state:** the repository implements the **first stage of that pipeline (voice separation) as a working FastAPI service**, with the remaining four stages (KWS, ASR, command interpretation, action) present only as **explicit architectural placeholders** that return `NOT_IMPLEMENTED`. No component has been evaluated on, or ported to, actual edge hardware. The project addresses the "voice separation front end" of PS 26172 concretely; it does not yet address "low-latency edge deployment" as a measured, demonstrated property.

---

## 2. Architecture: audio → preprocessing → separation → KWS → ASR → command → action

| Stage | Status | Detail |
|---|---|---|
| Audio input (file upload) | **IMPLEMENTED AND WORKING** | `/separate`, `/pipeline` accept WAV uploads via `UploadFile` |
| Audio input (live mic) | **PLACEHOLDER / NOT INTEGRATED** | `audio_processing/capture.py` (`MicCapture`) exists but is not wired into any API route |
| Preprocessing (mono, resample, normalize) | **IMPLEMENTED AND WORKING** | `audio_processing/preprocess.py`, `audio_processing/utils.py` |
| Voice separation (Conv-TasNet) | **IMPLEMENTED AND WORKING** | `voice_separation/engine.py` via torchaudio's pretrained `CONVTASNET_BASE_LIBRI2MIX` |
| Keyword spotting (KWS) | **PLACEHOLDER / NOT IMPLEMENTED** | `keyword_spotting/detector.py`, always returns `NOT_IMPLEMENTED` |
| ASR | **PLACEHOLDER / NOT IMPLEMENTED** | `asr/transcriber.py`, always returns `NOT_IMPLEMENTED`, empty text |
| Command interpretation | **PLACEHOLDER / NOT IMPLEMENTED** | `command/interpreter.py`, always returns `NOT_IMPLEMENTED`, action `"none"` |
| Action / device control | **NOT AVAILABLE** | No code anywhere triggers an external action |
| Speaker-count auto-detection | **IMPLEMENTED but NOT INTEGRATED / NOT VALIDATED** | `audio_processing/speaker_detection.py` — MFCC+KMeans speaker-count estimator exists, is standalone, **not called by any router**, has no dedicated tests |

The `/pipeline` endpoint chains preprocessing → separation → KWS-stub → ASR-stub → command-stub in one call, so the *wiring* of the full pipeline exists end-to-end even though 3 of 5 functional stages are stubs by design (this is stated explicitly in the code's own docstrings, not hidden).

---

## 3. Repository Structure

| Directory | Purpose | Status |
|---|---|---|
| `api/` | FastAPI app, routers, Pydantic schemas | Working |
| `voice_separation/` | Conv-TasNet model (ported from original RTP repo), `SeparatorEngine` adapter, plus original training/eval/solver scripts carried over unused | Model+engine working; training scripts present but unused/untested in this context |
| `audio_processing/` | Preprocessing, resample/normalize utils, VAD stub, mic capture, speaker-count estimator | Mixed: utils working, VAD/capture/speaker-detection are stubs or unintegrated |
| `keyword_spotting/` | KWS placeholder | Stub only |
| `asr/` | ASR placeholder | Stub only |
| `command/` | Command interpreter placeholder | Stub only |
| `models/` | Intended location for a local checkpoint | **Empty** (only `.gitkeep`) |
| `tests/` | Unit + integration tests | 33 tests, all passing (see §11) |
| `outputs/` | Runtime scratch dir for separated WAV files | Populated with leftover files from manual/test runs — not source-controlled artifacts |
| `pyproject.toml` / `requirements.txt` / `requirements-dev.txt` | Packaging & dependency declarations | Present and consistent with each other |
| `sih_voice_activator.egg-info/` | Build artifact from `pip install -e .` | Present, not source |

**Notable:** the root `README.md` is still the **original RTP repository's README** (title "VOICE-SEPARATION-FROM-MIXED-AUDIO-SIGNAL", describing WSJ0 training workflow) — it has not been rewritten to describe the SIH project. This repository is a restructured fork of the original RTP repo (confirmed via `git status` showing `src/*.py` renamed into `voice_separation/*.py`).

---

## 4. FastAPI Backend

- **Startup**: `api/main.py` builds `app = FastAPI(...)`, registers CORS (`allow_origins=["*"]`, all methods/headers — fine for local demo, not a production posture), includes 3 routers, and on the `startup` event **eagerly preloads the separation model** (see §5) so the first real request is not the one paying model-load cost.
- **Routers**:
  - `api/routers/separation.py` → `POST /separate`, `GET /separate/download`
  - `api/routers/pipeline.py` → `POST /pipeline`
  - `api/routers/status.py` → `GET /system/status`
  - Plus `GET /health` defined directly in `main.py`.
- **Request/response flow**: multipart file upload → temp file written to `outputs/` → `soundfile` read → preprocess → engine inference → WAV(s) written to `outputs/` → JSON response with file paths (not raw audio bytes) → client fetches via `/separate/download?path=...`.
- **Configuration**: no config file / env-based settings system; sample rate, device (`"cpu"`), and paths are hardcoded in route handlers.
- **Packaging**: `pyproject.toml` correctly declares `api`, `voice_separation`, `audio_processing`, `keyword_spotting`, `asr`, `command` as installable packages; `pip install -e ".[dev]"` works (verified — package importable, tests run against installed package structure).

**IMPLEMENTED AND WORKING**: startup, routing, request parsing, CORS, packaging.
**IMPLEMENTED but NOT FULLY VALIDATED**: `/separate/download` (no test exercises it); error paths for malformed uploads (only a couple of failure branches are exercised by tests).

---

## 5. Voice Separation — Exact Detail

- **Model actually used by default:** torchaudio's pretrained `CONVTASNET_BASE_LIBRI2MIX` bundle (`torchaudio.pipelines.CONVTASNET_BASE_LIBRI2MIX`), **not** a project-trained checkpoint.
- **Source:** downloaded/cached by torchaudio; local cache file confirmed at `~/.cache/torch/hub/torchaudio/models/conv_tasnet_base_libri2mix.pt`, **19,131 KB (20,049,107 bytes)** on disk.
- **Architecture:** Conv-TasNet (same family as the ported `voice_separation/conv_tasnet.py`), pretrained on LibriMix by the torchaudio maintainers — **not** the original RTP repo's own weights (which don't exist, per the earlier confirmed finding).
- **Params:** measured directly this session — **4,984,881 parameters** (~5M) for the loaded bundle model.
- **Sample rate:** 8,000 Hz (confirmed via `bundle.sample_rate` and `engine.sample_rate`).
- **Speaker count:** fixed at **2** (`model_C = 2`); no dynamic speaker-count selection.
- **Loading mechanism:** `SeparatorEngine._load_torchaudio_bundle()` (default path when no `checkpoint_path` given) vs. `SeparatorEngine._load_checkpoint()` (loads a `.pth.tar` package dict via `ConvTasNet.load_model_from_package` — dead code path today since `models/` is empty).
- **Model caching:** **YES, IMPLEMENTED** as of this stage — `voice_separation/engine.get_separator_engine()` is a process-wide, thread-safety-locked singleton keyed by `(checkpoint_path, device)`. The FastAPI startup event calls it eagerly, so the model loads once at process start and is reused by `/separate`, `/pipeline`, and `/system/status` for the life of the process. Verified by a dedicated test (`test_model_preloaded_at_app_startup`) and by direct log inspection (the "Loaded torchaudio..." log line appears exactly once across multiple requests).
- **Inference flow (`/separate`):** upload → mono-convert (`ensure_mono`) → cached-engine lookup → resample to 8 kHz if the upload's rate differs (`resample_if_needed`) → peak-normalize → `engine.separate()` → per-source WAV written at 8 kHz → JSON response including `original_sample_rate` and `model_sample_rate`.
- **Latency (measured this session, CPU, dev machine, 0.5 s test clips — NOT a formal or edge-hardware benchmark):**
  - Cold model construction: ~0.19–0.21 s
  - Cached model "load": effectively 0 s (dict lookup)
  - End-to-end `/separate` call: ~0.28 s (first, cold) → ~0.10–0.12 s (subsequent, cached)
  - These numbers reflect a development laptop/CI environment, not target edge hardware, and are not a substitute for a proper latency benchmark suite.

---

## 6. Original RTP Integration

- **Reused as-is:** `ConvTasNet` architecture (`voice_separation/conv_tasnet.py`), `pit_criterion.py`, `utils.py` (`overlap_and_add`, `remove_pad`), `data.py`, `solver.py`, `train.py`, `preprocess.py`, `evaluate.py` — all carried over from `src/*.py` in the original RTP repo, unchanged in logic (confirmed via `git status` renames).
- **Changed:** import paths made package-relative (`from .conv_tasnet import ...`), file renames (`separate.py` → `separate_offline.py`), and the new `SeparatorEngine` adapter (`engine.py`) which did **not** exist in the original repo — it's new integration glue, not a modification of the model.
- **Original RTP trained weights:** confirmed in the prior inspection — **none exist**, anywhere in that repo's working tree or git history. Training/eval/separate scripts are architecture-only.
- **What remains available for future work:** the full training pipeline (`train.py`, `solver.py`, `data.py`, `egs/wsj0/run.sh` recipe) is intact and could train a project-specific checkpoint on WSJ0-mix or LibriMix data if/when that data and compute become available — this has explicitly **not** been attempted in this stage.

---

## 7. Audio Processing

| Function | Status |
|---|---|
| Mono conversion (`ensure_mono`) | **IMPLEMENTED AND WORKING**, unit-tested |
| Resampling (`resample_if_needed`, librosa-based) | **IMPLEMENTED AND WORKING**, unit-tested, and now used in both `/separate` and `/pipeline` |
| Normalization (`normalize_audio`, peak-normalize) | **IMPLEMENTED AND WORKING**, unit-tested |
| `AudioPreprocessor` (used by `/pipeline`) | **IMPLEMENTED AND WORKING**, unit-tested |
| VAD (`VoiceActivityDetector`) | **PLACEHOLDER** — energy-threshold stub only; docstring explicitly says a real WebRTC/Silero VAD is planned; not integrated into any route |
| Mic capture (`MicCapture`) | **IMPLEMENTED but NOT INTEGRATED / NOT VALIDATED** — blocking `sounddevice` recorder exists, no route or test exercises it |
| Speaker-count detection (`speaker_detection.py`) | **IMPLEMENTED but NOT INTEGRATED / NOT VALIDATED** — MFCC+KMeans+silhouette pipeline exists, explicitly documented as not called by `/pipeline`, has no tests |

**Current limitations:** no streaming/chunked audio support (whole-file only); no VAD gating before separation; `/separate` and `/pipeline` each preprocess independently rather than sharing one code path (minor duplication, not a correctness bug); speaker-count is always fixed at 2 regardless of actual audio content.

---

## 8. Keyword Spotting (KWS)

- **Status: PLACEHOLDER / NOT IMPLEMENTED.**
- **File/class:** `keyword_spotting/detector.py`, class `KeywordDetector`, method `detect()`.
- **Real model:** none. `detect()` unconditionally returns `KeywordResult(status="NOT_IMPLEMENTED", keyword=None, confidence=None)` — it does not even inspect the waveform.
- **Remaining work:** everything — feature extraction, model selection (the docstring names Porcupine or a custom MFCC/CNN approach as future options), training or licensing a model, integration into `/pipeline`, latency/edge-footprint validation.

---

## 9. ASR

- **Status: PLACEHOLDER / NOT IMPLEMENTED.**
- **File/class:** `asr/transcriber.py`, class `Transcriber`, method `transcribe()`.
- **Real model:** none. Always returns `TranscriptionResult(status="NOT_IMPLEMENTED", text="")`.
- Docstring names Whisper (local) or Vosk (offline) as planned backends — neither is wired up or even imported.

---

## 10. Command / Action System

- **Status: PLACEHOLDER / NOT IMPLEMENTED.**
- `command/interpreter.py`, class `CommandInterpreter`, method `interpret()` — always returns `action="none"`, `status="NOT_IMPLEMENTED"`.
- No action/execution layer (no device control, no OS hooks, no output side-effects) exists anywhere in the repo.

---

## 11. Testing

**Test files (5, all under `tests/`):**

| File | Tests | Focus |
|---|---|---|
| `tests/integration/test_api.py` | 11 | App startup, `/health`, `/system/status`, `/separate` (8 kHz + 16 kHz + caching), `/pipeline` |
| `tests/unit/test_audio_processing.py` | 8 | mono, normalize, resample, preprocessor, VAD stub |
| `tests/unit/test_conv_tasnet.py` | 6 | Import, instantiation, forward-shape, determinism, `SeparatorEngine` numpy I/O, status fields |
| `tests/unit/test_engine_caching.py` | 3 | Cache identity, load-count-once, cached-engine defaults |
| `tests/unit/test_stubs.py` | 7 | KWS/ASR/Command all correctly report `NOT_IMPLEMENTED`, no fake outputs |
| `tests/unit/test_conv1d_learn.py` | **0** | Leftover exploratory script from the original author (Conv1d shape scratch-work); has no `test_*` functions and contributes nothing to the suite — dead weight, not a real test |

**Latest full run:** `33 passed, 5 warnings in ~6.4–6.9s` (warnings are deprecation notices for `@app.on_event`, not failures).

**What is actually validated:** import/instantiate/forward-pass correctness of Conv-TasNet at small hyperparameters; `SeparatorEngine` numpy I/O contract; that the API starts and responds; that separation runs end-to-end at both 8 kHz and 16 kHz input and returns valid metadata; that the model is loaded once (at startup) and not reloaded per request; that KWS/ASR/command stubs never fabricate results.

**Important untested paths:** `/separate/download`; malformed/corrupt audio uploads beyond the one exercised failure case; the checkpoint-loading branch of `SeparatorEngine` (no checkpoint exists to test against); `MicCapture`; `speaker_detection.py`; concurrent/parallel request behavior under the FastAPI event loop; any behavior on non-CPU devices.

---

## 12. Current Runtime

- **Launch:** `uvicorn api.main:app --reload --port 8000` (per `api/main.py` module docstring).
- **Port:** 8000 (default, not configurable via env/config in current code).
- **Swagger/OpenAPI:** available at `/docs` (Swagger UI) and `/redoc` — FastAPI defaults, not disabled anywhere in `main.py`.
- **API status:** all declared endpoints respond correctly in tests; no known broken route.

---

## 13. Performance

Only figures actually measured in this session are reported; everything else is explicitly marked unmeasured.

| Metric | Value | Conditions |
|---|---|---|
| Model parameters | 4,984,881 | torchaudio `CONVTASNET_BASE_LIBRI2MIX` |
| Model file size on disk | 20,049,107 bytes (~19.1 MB) | cached `.pt` file |
| Cold model load time | ~0.19–0.21 s | CPU, dev machine |
| Cached model "load" | ~0 s | dict lookup |
| `/separate` latency, first request | ~0.28 s | 0.5 s WAV, CPU, dev machine |
| `/separate` latency, subsequent requests | ~0.10–0.12 s | same conditions |
| RAM usage | **Not yet measured** | |
| CPU utilization | **Not yet measured** | |
| Power consumption | **Not yet measured** | |
| Edge-device (e.g., Raspberry Pi/Jetson/MCU) latency | **Not yet measured** — no edge hardware testing has occurred | |
| Separation quality (SI-SNRi/SDR) on this project's own data | **Not yet measured** — no evaluation run against project-specific mixtures | |

---

## 14. Current Limitations and Technical Risks

- **No edge validation whatsoever** — everything has run on a development machine only; the PS's core "low-latency, efficient, edge" claim is unproven.
- **Separation quality is unvalidated** for whatever real-world/target audio conditions the jury or use case cares about — the bundle is a generic LibriMix-trained model, not tuned to project data.
- **No real KWS/ASR/command** — 3 of 5 functional pipeline stages are stubs; the "voice activator" cannot yet activate on anything.
- **No streaming audio path** — whole-file upload only; a low-latency activator ultimately needs streaming/chunked processing, which doesn't exist yet.
- **Fixed 2-speaker assumption** baked into the default model; the standalone speaker-count estimator is unintegrated.
- **CORS wide open (`*`)** — fine for a local demo, a risk if ever exposed beyond localhost.
- **Stale root README** — documentation risk for anyone (including a jury member browsing the repo) forming a first impression.
- **No config/secrets management, no logging aggregation, no auth** — expected at this stage but worth naming as gaps before any real deployment discussion.

---

## 15. SIH Requirement Coverage

| Requirement | Current status | Evidence | Gap |
|---|---|---|---|
| Voice separation from mixed/overlapping speech | Implemented (generic pretrained model) | `voice_separation/engine.py`, passing tests | No project-specific training/tuning; quality unmeasured |
| Low latency | Partially demonstrated on dev hardware only | ~0.1s/req measured (§13) | No edge-hardware measurement; no streaming path |
| Efficient / edge-suitable | Not yet demonstrated | Model is ~5M params, ~19MB, CPU-only path exists | No RAM/CPU/power profiling; no quantization; no edge deployment |
| Wake-word / keyword activation | Not implemented | `keyword_spotting/detector.py` stub | Entire KWS stage remains to be built |
| Speech-to-text / command understanding | Not implemented | `asr/transcriber.py`, `command/interpreter.py` stubs | Entire ASR + NLU/command stage remains to be built |
| Action execution | Not available | No code | Entire action layer remains to be designed and built |
| Robust automated testing | Implemented for what exists | 33/33 tests passing | Stub modules only tested for "honesty" (no fake output), not real functionality since none exists yet |
| API/service layer | Implemented and working | FastAPI app, 3 routers, Swagger docs | No auth, no config system, no production hardening |

---

## 16. Current Stage Assessment

**Best description: Functional Prototype (of the voice-separation sub-system only); Concept/Architecture-stage for the overall voice-activator pipeline.**

Reasoning: one real, working, tested, API-exposed capability exists (Conv-TasNet-based separation, with proper model caching and sample-rate handling) — that's beyond "concept." But the product the PS actually asks for (an end-to-end voice *activator* that spots a wake word, understands a command, and acts) has 3 of 5 stages entirely unimplemented, with zero edge-hardware validation anywhere. It is not yet a "Functional Prototype" of the *whole system*, and nowhere near "Deployment Ready."

---

## 17. Recommended Next Stage — Top 5 Priorities

1. **Implement a real KWS module** (e.g., a small CNN/MFCC classifier or Porcupine integration) and wire it into `/pipeline`, replacing the stub.
2. **Integrate a real offline ASR** (Whisper-tiny/base or Vosk) sized for edge feasibility, replacing the ASR stub.
3. **Build a minimal rule-based command interpreter** for a small fixed command set, replacing the command stub, to complete one true end-to-end path.
4. **Establish real performance baselines**: measure RAM, CPU%, and latency per stage on the actual target edge hardware (or at least a representative low-power device), not just the dev laptop.
5. **Evaluate separation quality on representative data** (SI-SNRi/SDR on audio resembling the target use case) rather than relying on generic LibriMix pretrained performance.

---

## 18. Demo Readiness

**Can be demonstrated to a jury today, on a laptop:**
- Starting the FastAPI service and showing Swagger UI (`/docs`).
- Uploading a mixed-speaker WAV file (any sample rate) to `/separate` and getting back two separated-speaker WAV files, with correct resampling metadata shown in the JSON response.
- Showing `/system/status` reporting model load state honestly.
- Running `/pipeline` end-to-end and showing that separation succeeds while KWS/ASR/command **honestly and explicitly** report `NOT_IMPLEMENTED` — useful to demonstrate architectural completeness/roadmap, not to claim a finished product.
- Running the automated test suite live (33/33 passing) as evidence of engineering rigor.

**Cannot be demonstrated:**
- Any live "say a wake word and it responds" interaction — no KWS exists.
- Any speech-to-text or command execution.
- Any live microphone demo (capture code exists but isn't wired into the API).
- Any edge-device (non-laptop) demo — nothing has been tested off this class of hardware.
- Any latency/efficiency claim beyond the ad hoc dev-machine numbers in §13.

---

## 19. Presentation-Ready Summary

- **Problem:** Voice-controlled edge devices need to isolate a target speaker from noisy/overlapping speech, recognize a wake word, and understand a spoken command — all with low latency and limited compute, without cloud dependence.
- **Proposed solution:** A modular five-stage pipeline (separation → KWS → ASR → command → action) built around Conv-TasNet for source separation, exposed through a lightweight FastAPI service suitable for eventual on-device deployment.
- **What we have built:** A working FastAPI backend with a cached, pretrained Conv-TasNet separation engine that correctly handles arbitrary input sample rates, returns clean per-speaker audio, and is backed by a 33-test automated suite (all passing). The full pipeline scaffolding (routes, schemas, stub interfaces for KWS/ASR/command) is in place and honestly reports what is and isn't implemented.
- **Current results:** Reliable 2-speaker separation via a pretrained model (no project-specific training yet); sub-150ms cached inference on a development machine for short clips; zero false claims — every unimplemented stage is explicitly marked as such in both code and API responses.
- **Current limitations:** No wake-word detection, no speech-to-text, no command execution, and no edge-hardware testing yet; separation model is generic, not tuned to project-specific audio.
- **Next milestone:** Implement real KWS and a minimal ASR + command interpreter to complete one true end-to-end voice-activation path, then begin edge-hardware performance validation.

---

## 20. Stage-1 Completion Checklist

- ✅ FastAPI backend scaffolded, running, documented via Swagger
- ✅ Conv-TasNet ported and integrated without architecture changes
- ✅ Model caching (load-once-at-startup, reused across requests)
- ✅ Sample-rate handling (resample to 8 kHz) in both `/separate` and `/pipeline`
- ✅ Automated test suite covering separation, caching, resampling, and stub-honesty (33/33 passing)
- ✅ Clear, honest architectural placeholders for KWS/ASR/command (no fake outputs)
- 🟡 Audio preprocessing utilities (mono/normalize/resample) — complete, but VAD is only a stub and not integrated
- 🟡 Speaker-count detection — implemented as a standalone module, not integrated or tested against the pipeline
- 🟡 Microphone capture — implemented, not wired into any endpoint or tested
- ❌ Keyword spotting — not implemented
- ❌ ASR — not implemented
- ❌ Command interpretation / action execution — not implemented
- ❌ Edge-hardware testing (latency, RAM, CPU, power) — not performed
- ❌ Project-specific model training/evaluation — not performed
- ❌ Streaming/real-time audio path — not implemented
- ❌ Production hardening (config system, auth, restricted CORS) — not implemented
