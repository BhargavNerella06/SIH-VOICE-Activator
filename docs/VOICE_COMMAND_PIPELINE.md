# SIH 2026 — PS 26172: End-to-End Voice Command Pipeline (Laptop Demo)

**Scope:** `voice_pipeline/orchestrator.py` — wiring the existing, unmodified
`MicCapture` → `KeywordDetector`/mock trigger → `AudioSessionManager` → `asr.transcriber`
→ `command.interpreter.CommandInterpreter` components into one laptop-demo pipeline.
This is integration/orchestration only: **no component's internals were redesigned**
(Conv-TasNet, TinyKWSNet's architecture, ASR, and the command interpreter are all
untouched — see each component's own doc for its internals).

**No NOVA detection accuracy is claimed anywhere in this document.** No trained
`models/kws_nova_cnn.pt` checkpoint exists in this repo; every test and demo run either
uses a mock wake trigger (explicitly not NOVA detection) or the real, currently-untrained
`KeywordDetector` (which never fires).

---

## 1. Architecture

```
MicCapture chunks (or --wav playback)
        |
        v
  wake detection            <- MODE A: real KeywordDetector (TinyKWSNet)
        |                      MODE B: MockWakeTrigger (dev/test only)
        v
  AudioSessionManager        pre-roll + command-audio capture,
        |                    silence/min/max-duration end-of-command
        v
  asr.transcriber            cached faster-whisper, loaded once
        |
        v
  command.interpreter        deterministic rule-based parser
        |
        v
  structured PipelineResult  (transcript, intent, action, device,
                               parameters, confidence, timings)
```

Every arrow above is an existing, unmodified component's existing public interface.
`voice_pipeline/orchestrator.py` (`VoiceCommandOrchestrator`) contains no audio DSP, no
ASR logic, and no command-parsing logic of its own — it only sequences calls to the four
components already documented elsewhere:

| Component | Doc |
|---|---|
| `audio_processing.capture.MicCapture` | (Stage 2.1, no dedicated doc — see its own docstring) |
| `keyword_spotting.detector.KeywordDetector` | `docs/AUDIO_SESSION_MANAGER.md` §7, `docs/STAGE_2_EDGE_BENCHMARK.md` |
| `audio_processing.session.AudioSessionManager` | `docs/AUDIO_SESSION_MANAGER.md` |
| `asr.transcriber` | `docs/ASR_BACKEND.md` |
| `command.interpreter.CommandInterpreter` | `docs/COMMAND_INTERPRETER.md` |

---

## 2. Mock vs. real KWS mode

**The rest of the pipeline is identical after the wake event, in both modes** — this is
possible entirely because `AudioSessionManager.detector` (see `docs/
AUDIO_SESSION_MANAGER.md` §7) already accepts any duck-typed
`detect_chunk(chunk) -> Optional[result]` object. `VoiceCommandOrchestrator` just decides
*which* object to construct; nothing downstream knows or cares which one fired.

### MODE A: `WakeMode.REAL_KWS`

Uses `keyword_spotting.detector.get_keyword_detector()` — the real, cached,
TinyKWSNet-backed detector, completely unmodified. **With no trained
`models/kws_nova_cnn.pt` checkpoint (the current state of this project), it reports
`is_ready=False` and never fires `DETECTED`.** `scripts/voice_command_demo.py --real-kws`
prints an explicit warning to this effect before starting, and the resulting
`PipelineResult.status` is `"INCOMPLETE"` (audio ran out before any command was
captured) — never a fabricated command. **Once a real checkpoint exists, this mode
starts working with zero changes to this module** — that's the entire point of having
built `KeywordDetector` and `AudioSessionManager`'s detector-injection this way.

### MODE B: `WakeMode.MOCK`

Uses `MockWakeTrigger` (defined in `voice_pipeline/orchestrator.py`) — a deterministic,
manually-armed stand-in. Calling `.fire()` makes the *next* `detect_chunk()` call report
`DETECTED`, regardless of what audio is actually in that chunk. **This is explicitly not
NOVA detection**, and every log line/message mentioning it says "MOCK" (verified by
`test_orchestrator_mock_mode_logs_are_clearly_labeled`) — it exists purely so the rest of
the pipeline (capture → ASR → interpretation) can be developed, demonstrated, and tested
while real NOVA recording/training remains postponed.

`scripts/voice_command_demo.py --mock-wake` fires the mock trigger automatically once
`--mock-fire-after` seconds of audio have been processed (default 1.0s) — a scripted
convenience for a hands-free demo, not a claim of any kind about detection.

---

## 3. Data flow / result schema

```python
@dataclass
class PipelineResult:
    status: str        # "OK" | "ASR_ERROR" | "EMPTY_AUDIO" | "SESSION_ERROR" | "INCOMPLETE"
    transcript: str
    command: Optional[CommandResult]   # see docs/COMMAND_INTERPRETER.md for this schema
    end_reason: Optional[str]          # "silence" | "max_duration", from AudioSessionManager
    timings: PipelineTimings
    message: str
```

`status="OK"` covers a **successfully interpreted command regardless of whether
`CommandInterpreter` recognized it** — `action="unknown"` is a valid, non-error outcome
(matches the command interpreter's own design philosophy; see
`docs/COMMAND_INTERPRETER.md` §6). The other statuses mean the pipeline could not reach
interpretation at all:

| Status | Meaning |
|---|---|
| `OK` | A transcript was produced and interpreted (possibly as `"unknown"`). |
| `ASR_ERROR` | ASR reported `NOT_IMPLEMENTED` or `ERROR` — see `docs/ASR_BACKEND.md` §3. |
| `EMPTY_AUDIO` | The captured command segment had zero samples — ASR was never called. |
| `SESSION_ERROR` | `AudioSessionManager` hit its `ERROR` state (malformed chunk) — see `docs/AUDIO_SESSION_MANAGER.md` §8. |
| `INCOMPLETE` | The chunk source was exhausted before a wake event + command capture completed — the expected outcome for `REAL_KWS` mode with no trained checkpoint. |

---

## 4. How to run the demo

```bash
# Mock wake trigger, live microphone (fires automatically after 1s):
python scripts/voice_command_demo.py --mock-wake

# Mock wake trigger, a WAV file instead of the microphone (no recording
# required -- e.g. a file synthesized via Windows SAPI TTS, as used
# elsewhere in this project's tests/demos):
python scripts/voice_command_demo.py --mock-wake --wav path/to/command.wav

# Real KeywordDetector (TinyKWSNet) -- will not fire without a trained
# models/kws_nova_cnn.pt checkpoint; prints an explicit warning:
python scripts/voice_command_demo.py --real-kws
python scripts/voice_command_demo.py --real-kws --wav path/to/command.wav
```

### Example output (`--mock-wake --wav`, verified this session with
Windows-SAPI-synthesized — not recorded — speech saying "turn on the light")

```
Listening... (MOCK wake trigger -- NOT real NOVA detection)
Wake trigger detected (MOCK trigger)
Capturing command...
Transcribing...
Command: "Turn on the light."
Intent: device_control
Device: light
Action: turn_on

=== Result ===
Status         : OK
Transcript     : 'Turn on the light.'
Intent         : device_control
Action         : turn_on
Device         : light
Parameters     : {'device': 'light'}
Confidence     : 1.0

=== Latency (this dev machine only) ===
Command capture duration : 1676.8 ms
ASR latency              : 279.4 ms
Total wake-to-result     : 1956.2 ms
```

Note: `Action: turn_on` (not a simplified "on") — the orchestrator prints
`CommandResult.action` exactly as `command.interpreter` produces it, preserving the
documented, tested schema (`docs/COMMAND_INTERPRETER.md`) rather than inventing a second,
parallel action vocabulary purely for console display.

---

## 5. Latency instrumentation

`PipelineTimings` records a `time.time()` timestamp at each stage:
`wake_detected_ts`, `command_capture_start_ts`, `command_capture_end_ts`,
`asr_start_ts`, `asr_end_ts`, `command_interpreted_ts` — plus three derived
properties: `command_capture_duration_ms`, `asr_latency_ms`,
`total_wake_to_result_ms` (`None` if the underlying timestamps aren't both set,
never a fabricated duration).

### Measured this session (dev-machine only — see §6)

One real cycle, `--mock-wake --wav` with real-time-paced playback of synthesized speech
("turn on the light", ~1.7s of audio including trailing silence for the silence-based
end-of-command trigger):

| Metric | Value |
|---|---:|
| Command capture duration | 1,676.8 ms |
| ASR latency (faster-whisper `tiny.en`) | 279.4 ms |
| Total wake-to-result | 1,956.2 ms |

**A first-call ASR warm-up cost was also observed and is worth documenting explicitly**:
in a fresh process, the very first `transcribe()` call after model construction took
~2,200–2,370 ms; every subsequent call in the same process (verified across 3 consecutive
cycles) took ~225–280 ms. This is a one-time CTranslate2/faster-whisper backend
initialization cost, not a per-command cost — the orchestrator's `get_transcriber()`
caching (§7) means this cost is paid at most once per process, not once per voice
command.

---

## 6. What this latency data does and does not prove

- **MEASURED:** real wall-clock timings for this exact software stack, on this exact dev
  laptop, for one exact synthetic utterance.
- **NOT a SIH final-latency compliance claim.** This runs entirely in a CPython process
  on a 12-core x86-64 laptop — not the eventual edge/target hardware, not over a real
  wireless link (see `docs/STAGE_3_STREAMING.md` for the equivalent, already-stated
  caveat about the streaming path's own latency numbers), and with a generic pretrained
  Whisper model on synthesized (not real, not diverse) speech.
- **Command capture duration is bounded by the demo's own `silence_seconds`/
  `min_command_seconds`/`max_command_seconds` configuration** (`docs/
  AUDIO_SESSION_MANAGER.md` §6, defaults used here), not by anything intrinsic to the
  pipeline — a different configuration would give a different number for the same
  utterance.

---

## 7. Reuse and no-reload guarantees

- `asr.transcriber.get_transcriber()` is the SAME process-wide cache every other ASR call
  site in this project uses (`api/routers/pipeline.py`, `voice_command.py`, `stream.py`,
  and now this orchestrator) — the model loads at most once per process, not once per
  voice command. Verified by `test_reset_allows_a_second_cycle_without_reloading_asr`
  (same object, `fake_transcriber.calls == 2` after 2 cycles, not 2 separate loads).
- `MicCapture`, `AudioSessionManager`, `KeywordDetector`/`get_keyword_detector()`, and
  `CommandInterpreter` are all reused via their existing, documented public interfaces —
  none were modified for this task.
- `VoiceCommandOrchestrator.reset()` (not `start()` again — see `docs/
  AUDIO_SESSION_MANAGER.md`'s own IDLE-vs-COMPLETE/ERROR distinction, which this
  orchestrator's lifecycle faithfully mirrors) allows another wake → command cycle
  without reconstructing anything.

---

## 8. Tests

`tests/unit/test_voice_pipeline.py` — **15 tests**, synthetic audio + mocked ASR
(monkeypatched `get_transcriber`, mirroring `tests/integration/test_stream_api.py`'s
established pattern) + the real, unmodified `CommandInterpreter`:

- complete wake → command → ASR → interpreter flow, with full latency-timing assertions;
- `MockWakeTrigger` fires exactly once per `.fire()`, ignores audio content entirely, and
  every log mention of "NOVA" in MOCK mode is a disclaimer, never a claim;
- the real `KeywordDetector` (confirmed `is_ready=False`) never produces a command —
  `PipelineResult.status == "INCOMPLETE"`, ASR never even called;
- ASR failure (`ERROR` and `NOT_IMPLEMENTED` statuses) surfaces as `ASR_ERROR`, never a
  crash or a fabricated transcript;
- an unrecognized transcript is a successful (`status="OK"`) pipeline result with
  `action="unknown"` — not treated as a failure;
- empty command audio short-circuits before ASR is ever called;
- a malformed chunk triggers `SESSION_ERROR`, not a crash;
- hitting `max_command_seconds` (the built-in "timeout") still proceeds through ASR/
  interpretation normally as a valid `OK` result with `end_reason="max_duration"`;
- `reset()` supports a second full cycle, reusing the same cached ASR instance (no
  reload) and clearing previous timings.

Full project suite: **311 passed, 0 failed** (296 before this task; 15 new).

---

## 9. Known limitations

- **No real NOVA checkpoint exists.** `--real-kws` is real, working *infrastructure*
  wiring, verified to never crash and never fabricate a detection — it is not, and does
  not claim to be, a working wake-word detector yet.
- **The mock trigger's timing is scripted, not audio-content-based.** It fires after a
  configured amount of processed audio (`--mock-fire-after`) or a manual `.fire()` call
  in tests — it never "listens" for anything.
- **Console `Action:` output uses the interpreter's real action vocabulary** (e.g.
  `"turn_on"`), not a simplified on/off form — a deliberate choice to avoid a second,
  parallel action naming scheme (see §4's note).
- **No live-microphone verification was performed this session** (per explicit task
  scope: "do not record microphone samples") — the `--wav` path (synthesized speech,
  never recorded) was used to verify the full pipeline end-to-end instead. The live-mic
  code path (`_mic_chunk_source` in `scripts/voice_command_demo.py`) reuses `MicCapture`
  exactly as already tested elsewhere in this project, but was not itself exercised with
  a live device in this session.
- **Latency numbers are dev-machine-only** (§6) — not a SIH final-compliance claim.
- **Single wake → command cycle per `start()`**, mirroring `AudioSessionManager`'s own
  design — call `reset()` (not `start()`) for another cycle.

---

## 10. Recommended next step

Once a real trained `models/kws_nova_cnn.pt` checkpoint exists, re-run
`scripts/voice_command_demo.py --real-kws` (live microphone) as the first real,
non-mocked end-to-end validation — no code changes to this pipeline should be required,
per §2's design intent. Until then, `--mock-wake` remains the way to demonstrate and
develop everything downstream of wake detection.
