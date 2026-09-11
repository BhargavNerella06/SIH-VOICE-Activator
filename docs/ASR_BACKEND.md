# SIH 2026 — PS 26172: ASR Backend

**Scope:** the standalone speech-to-text module, `asr/`, used after NOVA wake-word
detection to turn a captured utterance into plain text for command interpretation. This
document covers ASR **only** — no command interpretation, no streaming/network
transport, and no KWS integration are described or changed here.

**No accuracy claim is made anywhere in this document.** The model used is generic
pretrained Whisper, not fine-tuned on NOVA or any project-specific vocabulary.

---

## 1. Selected backend: faster-whisper

**Package:** [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper) (CTranslate2
port of OpenAI Whisper), version 1.2.1, already in `requirements.txt` / `pyproject.toml`.
**Model:** `tiny.en` (English-only, smallest Whisper size), CPU, `int8` compute type.

### Why this was selected

Before choosing, the existing environment/dependencies were inspected:
`torch`, `numpy`, `librosa`, `soundfile`, `resampy`, `sounddevice`, `openwakeword`,
`fastapi` were already present; no ASR library was. Candidates considered:

| Option | Verdict |
|---|---|
| **faster-whisper** | **Selected.** Open-source, fully local/offline after a one-time model download, no cloud API call at inference time. Accepts a numpy waveform directly (`model.transcribe(ndarray, ...)`) — no `ffmpeg` dependency, unlike plain `openai-whisper`. CTranslate2's own runtime (not torch) keeps it lightweight and fast on CPU even for a laptop-class device. |
| `openai-whisper` (original) | Requires `ffmpeg` for audio decoding and carries a much heavier torch-based runtime for the same model weights; no benefit over faster-whisper for this project, which already has audio decoded as numpy arrays. |
| `vosk` | Fully offline, no first-run download-then-cache step, but recognition quality for arbitrary command phrases is noticeably weaker than Whisper in practice, and its streaming-oriented API (`KaldiRecognizer` + raw byte feeding) is a worse fit than a single `transcribe(waveform)` call for this project's one-shot-utterance use case. |
| Windows SAPI (already used for KWS TTS data synthesis) | Text-to-*speech* only; SAPI's speech-*recognition* API is a different, Windows-only, proprietary component — inappropriate for a cross-platform ASR module and unrelated to why SAPI already appears elsewhere in this repo. |

`tiny.en` (not a larger size) was chosen as the default because this project's ASR input
is short command-style utterances, not long-form dictation — `tiny.en` is small, fast to
download, and fast to run, while still producing correct transcriptions for short
commands (verified in §4). A larger model (`base.en`, etc.) can be selected via the
`model_name` parameter without any code change if accuracy on real recordings later
proves `tiny.en` insufficient.

### Architectural note: dev-machine stand-in for a remote service

This module runs **in-process on the same laptop** as everything else, for this Stage's
laptop-demo scope. The final product's intent (stated since ASR was first added) is for
ASR to run on a **separate, more capable remote/server machine**, not on the same
constrained hardware as `keyword_spotting`. Nothing here changes that intent — this
module is an honest dev-machine placeholder for that eventual remote call, not a claim
that ASR now runs "on the edge."

---

## 2. Interface

Two ways to call this module, both backed by the same cached model:

### 2.1 Simple functional interface (new, this task)

```python
from asr.transcriber import transcribe, TranscriptionError

try:
    text = transcribe(waveform, sample_rate=16000)
except TranscriptionError as exc:
    ...  # ASR unavailable or transcription failed; exc has a clear message
```

`transcribe(audio, sample_rate) -> str` — plain text in, plain text out. Raises
`TranscriptionError` (with a clear underlying reason) if the model isn't available or
transcription fails, rather than returning a sentinel value.

### 2.2 Structured interface (existing, used by `api/routers/pipeline.py`,
`api/routers/voice_command.py`, and `api/routers/stream.py`)

```python
from asr.transcriber import get_transcriber

result = get_transcriber().transcribe(waveform, sample_rate=16000)
# result.status: "OK" | "NOT_IMPLEMENTED" | "ERROR"
# result.text, result.language, result.latency_ms, result.message
```

Use this form when the caller needs to distinguish "ASR isn't available right now"
(`NOT_IMPLEMENTED` — e.g. no network on first use to fetch model weights) from "ASR ran
but failed" (`ERROR`) from a real transcription, without exception handling.

Both forms call `get_transcriber()`, a process-wide cache keyed by
`(model_name, device, compute_type)` — **the model loads at most once**, not on every
call/request, regardless of which interface is used.

### Expected input format

| | |
|---|---|
| Audio | mono `numpy.ndarray`, any float dtype/range (auto-converted); stereo is auto-downmixed via `audio_processing.utils.ensure_mono` |
| Sample rate | any positive integer; resampled internally to 16,000 Hz (faster-whisper/Whisper's native rate) if different |
| Duration | no fixed requirement — short command-style utterances (a few seconds) are the intended use case |

### Modularity

`Transcriber` wraps every faster-whisper-specific detail internally; nothing outside
`asr/transcriber.py` imports `faster_whisper` directly. Swapping the backend (e.g. to a
larger Whisper size, a different CTranslate2 model, or eventually a real remote ASR
service) means changing only this one file — every caller (`transcribe()`,
`get_transcriber().transcribe()`, and every API router that uses them) is unaffected as
long as the same `TranscriptionResult`/`str` contract is honored.

---

## 3. Errors

| Situation | Simple interface (`transcribe()`) | Structured interface (`Transcriber.transcribe()`) |
|---|---|---|
| `faster-whisper` not installed | raises `TranscriptionError` | returns `status="NOT_IMPLEMENTED"` |
| No network on first use (model download fails) | raises `TranscriptionError` | returns `status="NOT_IMPLEMENTED"` |
| Transcription itself raises (e.g. malformed audio) | raises `TranscriptionError` | returns `status="ERROR"` |
| Success, even on silence (empty text) | returns `""` | returns `status="OK"`, `text=""` |

**No fabricated transcript is ever returned** in any failure case — verified by
`tests/unit/test_asr_transcriber.py`.

---

## 4. Smoke test: audio in, text out

A minimal standalone script exercises ASR in isolation — no command interpretation, no
streaming, no API layer:

```bash
python scripts/asr_smoke_test.py --wav path/to/speech.wav

# or live from the microphone (requires an actual person to speak):
python scripts/asr_smoke_test.py --mic --duration 3
```

**Verified this session** with real (Windows-SAPI-synthesized, not recorded) speech
saying "the quick brown fox jumps over the lazy dog":

```
Loaded 3.37s of audio at 22050 Hz

Status     : OK
Transcript : 'The quick brown fox jumps over the lazy dog.'
Language   : en
Latency    : 275.2 ms (inside transcribe()) / 2240.2 ms (wall clock, incl. this script's overhead)
```

Correct transcription, correct punctuation/casing (Whisper's own normalization, not
project code), non-native input sample rate (22,050 Hz) correctly resampled internally.

---

## 5. Tests

`tests/unit/test_asr_transcriber.py` — **10 tests**, covering:
- graceful degradation (no fabricated transcript) when `faster-whisper` is missing or
  model loading raises, via mocked imports — no real model or network required;
- the process-wide model cache (same instance returned; different configs get different
  cache entries);
- the new simple `transcribe()` function: raises `TranscriptionError` on backend-missing
  and on a mocked transcription failure; returns a plain `str` on a mocked success;
  confirmed to reuse the same cache as the structured interface;
- real end-to-end transcription (silence handling, non-native sample rate) — gated
  behind actual `faster-whisper` availability via `pytest.importorskip`, so the suite
  degrades gracefully (skips, doesn't fail) in an environment without it.

Run: `python -m pytest -q tests/unit/test_asr_transcriber.py`

---

## 6. Known limitations

- **Not fine-tuned on NOVA or any command vocabulary.** Generic pretrained Whisper.
  No accuracy claim is made for any specific word, phrase, or speaker.
- **Whisper's known hallucination behavior on silence/noise** (e.g. inventing common
  phrases like "thank you for watching") is a documented general Whisper limitation, not
  something this module detects or corrects for.
- **Runs in-process on the dev laptop**, standing in for the intended remote ASR
  service — not a claim that ASR now runs "on the edge" or that its dev-machine latency
  represents the eventual remote-call latency.
- **First use per model config downloads weights from Hugging Face Hub** (a few tens of
  MB for `tiny.en`); without network access on that first use, ASR reports
  `NOT_IMPLEMENTED`/raises `TranscriptionError` rather than fabricating a result — by
  design, not a bug.
- **No command interpretation, streaming, or KWS integration is implemented, changed, or
  described in this document** — those exist elsewhere in the project (see
  `docs/NOVA_COMMAND_VOCABULARY.md`, `docs/STAGE_3_STREAMING.md`) and were explicitly out
  of scope for this task.
