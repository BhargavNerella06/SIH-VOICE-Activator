# SIH 2026 — PS 26172: Audio Session Manager

**Scope:** `audio_processing/session.py`'s `AudioSessionManager` — the laptop-side
component sitting between raw microphone chunks and one complete command-audio segment.
This document covers this component **only**. It does not implement, redesign, or
describe ASR (`docs/ASR_BACKEND.md`), the command interpreter
(`docs/COMMAND_INTERPRETER.md`), or the HTTP streaming path (`docs/STAGE_3_STREAMING.md`)
— those are separate, unmodified components this task explicitly did not touch.

**No NOVA detection accuracy is claimed anywhere in this document.** All tests use
synthetic audio and fake/mock detectors; the one check against the real
`KeywordDetector` only proves it plugs in without crashing while genuinely untrained
(`is_ready=False`) — not that it detects anything.

---

## 1. What this component does, and what it deliberately does not do

```
microphone chunks
  -> pre-roll buffer + KWS detector           (LISTENING)
  -> wake word detected -> command-audio capture
                                                (COMMAND_CAPTURE)
  -> silence / min / max duration reached
  -> one complete CommandAudioSegment          (COMPLETE)
```

`AudioSessionManager` produces exactly one thing: a `CommandAudioSegment` — a
`(waveform: np.ndarray, sample_rate: int)` pair plus some timing metadata, in precisely
the shape `asr.transcriber.transcribe()` / `get_transcriber().transcribe()` already
accept. **It does not call ASR, does not call the command interpreter, and does not read
from a microphone itself.** A caller drives it by repeatedly calling `process_chunk()`
with numpy arrays from wherever they came from (a live `MicCapture` loop, or synthetic
test data) — wiring this component to a real microphone loop and then to ASR and the
command interpreter is explicitly a separate, later task.

---

## 2. Existing infrastructure reused (unmodified)

| Component | Reused from | Purpose here |
|---|---|---|
| `AudioRingBuffer` | `audio_processing/ring_buffer.py` | Pre-roll buffer (§4) |
| `VoiceActivityDetector` | `audio_processing/vad.py` | Silence detection (§6) |
| `MicCapture` | `audio_processing/capture.py` | The intended real chunk source (not imported by this module — see `scripts/audio_session_smoke_test.py` for how a real mic loop would feed it) |

None of these three files were modified. The KWS-decoupling pattern (a duck-typed
`detector.detect_chunk(chunk) -> Optional[result]` contract, `result.status ==
"DETECTED"` signals a wake word) is the same one already established by
`api/streaming.py`'s `StreamSession` for exactly the same reason — reused as a *pattern*,
not as shared code, since this component is deliberately independent of the HTTP
streaming session (see §7).

---

## 3. Session states

```
IDLE -> LISTENING -> COMMAND_CAPTURE -> PROCESSING -> COMPLETE
                                                          |
                                                       reset()
                                                          |
                                                          v
                                                     LISTENING
```
`ERROR` is reachable from `LISTENING` or `COMMAND_CAPTURE` (see §8), with the same
`reset()` path back to `LISTENING`.

| State | Meaning |
|---|---|
| `IDLE` | Constructed but not started. No chunks accepted yet (`process_chunk()` raises `AudioSessionError`). |
| `LISTENING` | Actively feeding chunks to the pre-roll buffer and the KWS detector, waiting for a wake-word event. |
| `COMMAND_CAPTURE` | Wake word detected. Accumulating command audio, watching for silence / min / max duration (§5). |
| `PROCESSING` | Transient — set the instant an end-of-command condition fires, while the segment is assembled. In practice this is a single synchronous step (a numpy concatenation), not an async job; it exists as a distinct, observable state for callers/tests that want to see the transition, not because assembly takes meaningful time. |
| `COMPLETE` | One finished `CommandAudioSegment` is available on `.result`. No further chunks accepted until `reset()`. |
| `ERROR` | A fed chunk couldn't be interpreted as audio at all (§8). No further chunks accepted until `reset()`. |

Named `AudioSessionState` (not reusing `api.streaming.SessionState`, which has different
state names for the HTTP-streaming session) to avoid confusion if both are ever imported
in the same scope.

---

## 4. Pre-roll concept

Identical rationale to `audio_processing/ring_buffer.py`'s own docstring: a wake-word
detector has some reaction latency, so by the time it fires, the first fraction of a
second of the user's utterance has already gone by. `AudioSessionManager` feeds every
`LISTENING`-state chunk into a `AudioRingBuffer` (default capacity 0.5s) *unconditionally
and continuously*, regardless of whether a detection has happened yet. The instant a
detection fires, the buffer's current contents are read *once* and become the seed of
the command-audio capture — recovering audio that would otherwise have been clipped by
detection latency.

**Non-duplication invariant:** the chunk that triggers detection is written into the
pre-roll buffer *before* the detector is even called on it, so that chunk's samples are
already the newest samples in the pre-roll snapshot taken immediately after. That chunk
is never separately re-appended to the command buffer — verified directly by
`test_post_detection_audio_preserved_without_duplicating_boundary_chunk`.

---

## 5. Command capture and end-of-command detection

Once in `COMMAND_CAPTURE`, every subsequent chunk is:
1. Appended to the growing command-audio buffer.
2. Passed to a `VoiceActivityDetector` (`audio_processing/vad.py`, energy-threshold
   based, reused unmodified) to classify it as speech or silence.
3. Used to update a running "silence elapsed since last speech" counter — reset to zero
   on any speech chunk, incremented by the chunk's duration on a silence chunk.

The command ends (transitions to `PROCESSING` then `COMPLETE`) when **either**:
- **Silence:** at least `min_command_seconds` of command audio has been captured *and*
  the silence-elapsed counter has reached `silence_seconds`, **or**
- **Max duration:** the total captured command audio has reached
  `max_command_seconds`, regardless of silence — the "timeout" safety net for a command
  that never goes quiet (a stuck-open mic, background noise, or genuinely long speech).

`min_command_seconds` exists specifically to stop a single quiet chunk immediately after
the wake word from ending the command before the user has had a chance to speak at all.

---

## 6. Timing/configuration parameters (all configurable via the constructor)

| Parameter | Default | Meaning |
|---|---|---|
| `sample_rate` | 16000 Hz | Sample rate of chunks passed to `process_chunk()`. |
| `preroll_seconds` | 0.5 s | Pre-roll ring buffer capacity (§4). |
| `silence_seconds` | 0.8 s | Trailing silence required to end the command (§5). |
| `min_command_seconds` | 0.3 s | Minimum command-audio duration before silence-based termination is considered (§5). |
| `max_command_seconds` | 8.0 s | Hard cap / timeout on command-audio duration (§5). |
| `vad_threshold` | 0.01 | RMS energy threshold passed to `VoiceActivityDetector`. |

Constructor validation rejects non-positive `sample_rate`/`max_command_seconds`,
negative `min_command_seconds`/`silence_seconds`, and `min_command_seconds >
max_command_seconds` — fails fast on a misconfiguration rather than behaving
unpredictably at runtime.

---

## 7. Modularity: KWS detector injection

`AudioSessionManager(detector=...)` accepts anything exposing
`detect_chunk(chunk) -> Optional[result]` (a `result.status == "DETECTED"` signals a
wake word; `.confidence` is read opportunistically if present). This module never
imports `keyword_spotting` anywhere. Verified directly:

- All state-machine/pre-roll/timing tests use hand-written fake detectors
  (`FakeDetector`, `NeverDetects`, `RaisingDetector` in
  `tests/unit/test_audio_session.py`) — zero dependency on any real or trained model.
- `test_real_keyword_detector_integrates_safely_without_special_casing` plugs in the
  actual `keyword_spotting.detector.KeywordDetector()` (confirmed `is_ready=False` — no
  trained NOVA checkpoint exists in this repo) and confirms the session stays safely in
  `LISTENING`, with `detector_error is None` (its `NOT_IMPLEMENTED` result is a normal
  return value, not an exception) and zero crashes.

Swapping the mock/fake detector for the real (eventually-trained) `KeywordDetector`
requires **zero changes to `AudioSessionManager` itself** — only the object passed to
`detector=` changes, exactly matching requirement 10.

Detector *exceptions* (a buggy detector raising) are caught and recorded on
`.detector_error`, treated as "no detection this chunk" — they do **not** transition the
session to `ERROR` (that's reserved for genuinely malformed *audio*, not a detector
hiccup; see §8).

---

## 8. Error handling

| Situation | Behavior |
|---|---|
| `process_chunk()` called while `IDLE`/`PROCESSING`/`COMPLETE`/`ERROR` | Raises `AudioSessionError` — a caller lifecycle bug (forgot `start()`/`reset()`), not a runtime audio anomaly. |
| Empty chunk (0 samples) | Harmless no-op; state unchanged. Not an error. |
| Chunk that can't be converted to a numeric array | Transitions to `ERROR` (does **not** raise) with `.error_message` set — a live mic-reading loop can keep running and just check `.state`, rather than wrapping every call in try/except. |
| Detector raises an exception | Caught; recorded on `.detector_error`; treated as no detection this chunk. Session stays in its current state (`LISTENING` or `COMMAND_CAPTURE`) — never crashes, never transitions to `ERROR` for this reason. |
| `max_command_seconds` reached | Not an error — a normal, deterministic end-of-command outcome (`end_reason="max_duration"`), the designed "timeout" safety net (§5). |

---

## 9. How this connects to KWS and ASR later (not built yet)

This task deliberately stops at producing a `CommandAudioSegment`. The next task is
expected to:
1. Drive `AudioSessionManager.process_chunk()` from a real `MicCapture` read loop.
2. Pass `detector=keyword_spotting.detector.get_keyword_detector()` (the real,
   TinyKWSNet-backed detector, once trained) instead of a fake one — no change to this
   module needed, per §7.
3. On `state == COMPLETE`, take `.result.waveform` / `.result.sample_rate` and call
   `asr.transcriber.get_transcriber().transcribe(...)` (or the simpler
   `asr.transcriber.transcribe(...)` function) — both already accept exactly this shape.
4. Feed the resulting transcript text into
   `command.interpreter.CommandInterpreter().interpret(...)`.
5. Call `.reset()` to return to `LISTENING` for the next wake word.

None of steps 1–5 were implemented in this task, per explicit scope.

---

## 10. Tests and smoke test

`tests/unit/test_audio_session.py` — **29 tests**, synthetic audio and fake detectors
only: state transitions (including invalid-transition errors), pre-roll retrieval and
non-duplication at the detection boundary, command-audio accumulation, silence-based
termination (including a speech chunk resetting the silence counter), minimum-duration
protection against instant termination, maximum-duration forced termination, empty
chunks (no-op), invalid chunks (`ERROR` state, not a crash), detector exceptions
(handled safely, not an `ERROR`), result-segment sample-rate/sample-count correctness,
and the real-`KeywordDetector` safety check (§7).

Standalone smoke test (no microphone, no ASR, no command interpreter):
```bash
python scripts/audio_session_smoke_test.py
python scripts/audio_session_smoke_test.py --use-real-detector
```

Verified this session — synthetic "1s ambient silence, wake word fires, 1s simulated
speech, trailing silence": produced a 1.82s, 16000 Hz, 29,120-sample segment,
`end_reason="silence"`. With `--use-real-detector`, the session correctly stayed in
`LISTENING` throughout (untrained detector never fires).

---

## 11. Known limitations

- **Energy-threshold VAD only.** `VoiceActivityDetector` is explicitly a "Milestone 1
  stub" per its own docstring — a real WebRTC/Silero VAD would be more robust to
  background noise, but was reused as-is per this task's "do not redesign" scope for
  unrelated modules.
- **No real-time/wall-clock timeout for a stalled chunk source.** All duration logic is
  based on *sample count* (`elapsed_seconds = samples / sample_rate`), not wall-clock
  time — if a caller simply stops calling `process_chunk()` altogether (e.g. a dead mic
  thread), nothing here notices; there is no background timer. This is a deliberate
  scope limit (this component does no I/O or threading of its own, per requirement 9),
  not an oversight.
- **Single wake-word-per-session-instance model** (matching `StreamSession`'s same
  choice) — one `LISTENING → COMPLETE` cycle per `start()`, then `reset()` for another.
- **Not yet connected to a real microphone, ASR, or the command interpreter** — that
  integration is explicitly the next task (§9).
