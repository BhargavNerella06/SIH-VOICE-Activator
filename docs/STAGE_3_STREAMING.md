# SIH 2026 — PS 26172: Stage 3 Development-Machine Audio Streaming

**Scope of this stage:** a development-machine simulation of the wireless streaming
path — microphone → edge-side buffering → HTTP chunk streaming → remote ASR → command
interpreter → structured action. **No ESP32 hardware, no real wireless link, no NOVA
training data, and no changes to Conv-TasNet/voice_separation, the command
interpreter's logic, or the frontend** were part of this stage. Every timing number in
this document was measured on one machine talking to itself over `localhost` — see
§6 for exactly what that does and does not prove.

---

## 1. What this stage builds

```
microphone/audio source
  -> edge-side chunking + pre-roll buffering     (audio_processing.pcm, .ring_buffer)
  -> streaming audio chunks over HTTP            (api/routers/stream.py)
  -> remote ASR server                           (asr.transcriber -- unmodified)
  -> transcript
  -> existing command interpreter                (command.interpreter -- unmodified)
  -> structured action
```

Three new HTTP endpoints implement the streaming transport:

| Endpoint | Purpose |
|---|---|
| `POST /stream/start` | Open a session; returns a `session_id`. |
| `POST /stream/{session_id}/chunk` | Push one small raw PCM16 audio chunk (request body = raw bytes, not multipart/WAV). |
| `POST /stream/{session_id}/stop` | Finalize: assemble every received chunk, run ASR, run the command interpreter, return the structured result plus a full latency breakdown. |

A client simulator, `scripts/stream_demo.py`, plays a WAV file (or a live microphone
capture) back through this exact HTTP path in real-sized chunks, so the whole chain can
be exercised end to end on one machine without any new hardware.

---

## 2. Component map (what's new vs. what's reused unmodified)

| Component | File | Status |
|---|---|---|
| Chunk framing + PCM16 encode/decode | `audio_processing/pcm.py` | **New this stage** |
| Circular pre-roll buffer | `audio_processing/ring_buffer.py` | **New this stage** |
| Streaming session/timing bookkeeping + KWS integration | `api/streaming.py` | **New this stage** (KWS integration added as a follow-up milestone within this stage — see §4) |
| Streaming HTTP endpoints | `api/routers/stream.py` | **New this stage** |
| Streaming response schemas | `api/schemas.py` (additions) | **New this stage** |
| Dev-machine client simulator | `scripts/stream_demo.py` | **New this stage** |
| ASR | `asr/transcriber.py` | **Unmodified** (Stage 3 ASR/command milestone) — called once per stream, at `/stop`, exactly as `api/routers/voice_command.py` already does for a single uploaded file |
| Command interpretation | `command/interpreter.py` | **Unmodified** (this stage's own prior milestone) — called with the ASR transcript, unchanged |
| Microphone capture | `audio_processing/capture.py` | **Unmodified** — `MicCapture` already existed (Stage 2.1) and is reused as-is by `--mic` mode in the client simulator |

No file under `voice_separation/`, `keyword_spotting/`, `asr/`, or the frontend was
touched, including by the later KWS-integration milestone (§4) — `keyword_spotting`
is referenced only in this document and in test docstrings, never imported by
`api/streaming.py`. `command/interpreter.py` was inspected but not modified — the
streaming endpoint imports and calls it exactly as `api/routers/voice_command.py`
already did.

---

## 3. Why the transport is separate from ASR and command interpretation

`api/routers/stream.py` and `api/streaming.py` know nothing about what a transcript or
an action is. They accumulate raw PCM chunks into a buffer and record *when* things
happened; only at `/stop` do they hand the fully assembled waveform to
`asr.transcriber.get_transcriber()` and, separately, the resulting text to
`command.interpreter.CommandInterpreter()` — the same two calls
`api/routers/voice_command.py` already made for a single uploaded file.

This separation is what makes the following true: **the eventual ESP32 client can
replace `scripts/stream_demo.py` by POSTing the same raw little-endian PCM16 bytes to
the same three endpoints, and nothing in `api/streaming.py`, `api/routers/stream.py`,
`asr/transcriber.py`, or `command/interpreter.py` would need to change.** Only the
client (what captures audio and how it's transported to the socket) changes; the wire
format and the server are agnostic to who's sending.

PCM16 (raw little-endian 16-bit signed integers), not WAV-wrapped float32, was chosen
as the wire format specifically because it's what real microphone/ADC hardware —
including a future ESP32 I2S microphone — actually produces, and because a
constrained MCU client won't want to construct WAV headers or do float conversion
before sending. `audio_processing/pcm.py` is the single place both the dev-machine
client and the server agree on this encoding.

---

## 4. KWS integration: pre-roll buffer + wake-word detection

This section covers the KWS-integration milestone added after the initial streaming
transport (§1-3, §5-9 below describe that earlier work). **No NOVA training, no audio
recording, and no changes to Conv-TasNet, ASR, or the command interpreter were part of
this milestone either.**

### 4.1 Architecture

```
audio chunk
    +--> pre-roll ring buffer      (always fed, every chunk, unconditionally)
    |
    +--> KWS detector              (fed only while WAITING_FOR_WAKE_WORD)
              |
              +-- DETECTED
                       |
                       v
                 record wake_word_detected_ts
                       |
                       v
                 assembly := preroll.read()   (one snapshot, taken once)
                       |
                       v
                 state -> COLLECTING_COMMAND
                       |
                       v
              every further chunk appended to assembly
                       |
                       v
              (at /stop) -> ASR -> command interpreter
```

The pre-roll ring buffer (`audio_processing/ring_buffer.py`'s `AudioRingBuffer`, 0.5s
capacity) and the KWS detector are two independent consumers of the same incoming
chunk stream — `StreamSession.add_chunk()` feeds both without either depending on the
other. This is what requirement 1 ("keep the transport layer independent from the KWS
implementation") means in practice: `api/streaming.py` never imports
`keyword_spotting` anywhere; it only relies on a duck-typed contract (§4.3).

### 4.2 The explicit state machine

```
WAITING_FOR_WAKE_WORD -> WAKE_WORD_DETECTED -> COLLECTING_COMMAND -> FINALIZED
```

- **`WAITING_FOR_WAKE_WORD`** — only reachable if a detector was attached at session
  creation. Every chunk feeds the pre-roll buffer and is passed to
  `detector.detect_chunk()`. Nothing is added to the ASR `assembly` buffer yet.
- **`WAKE_WORD_DETECTED`** — entered the instant a detection fires; `wake_word_detected_ts`
  (a real `time.time()` server timestamp) and `wake_word_confidence` (if the detector's
  result exposes one) are recorded here.
- **`COLLECTING_COMMAND`** — entered immediately after, in the same `add_chunk()` call.
  `assembly` is seeded with one snapshot of `preroll.read()` (see §4.4 for why exactly
  once), and every subsequent chunk is appended directly. **A session created without a
  detector (`detector=None`) starts directly in this state** — nothing to wait for, so
  every chunk is collected immediately, identical to the pre-KWS-integration behaviour.
  This is what keeps `/stop` usable for development testing without a trained NOVA
  model (requirement 7) — `api/routers/stream.py`'s three endpoints are **completely
  unchanged** by this milestone; no detector is attached by default.
- **`FINALIZED`** — set by `close_session()` (called from `/stop`), alongside the
  existing `closed = True` flag. Whatever `assembly` holds at that point (empty, if
  stopped before any detection — see §4.5) is what ASR receives.

### 4.3 How the detector is injected

`StreamSession`/`create_session(sample_rate, detector=None)` accept any object
exposing `detect_chunk(chunk) -> Optional[result]`, where a non-`None` result carries a
`.status` attribute; `"DETECTED"` signals a wake-word event, anything else does not.
`.confidence` is read opportunistically via `getattr(..., "confidence", None)` if
present. This is **exactly** the shape of
`keyword_spotting.detector.KeywordDetector.detect_chunk()` → `Optional[KeywordResult]`
— but `api/streaming.py` never imports `keyword_spotting.detector`, `KeywordResult`, or
anything else from that package. It was verified directly (see §4.6) that a real
`KeywordDetector()` instance plugs in with zero special-casing.

`api/routers/stream.py` does **not** attach a detector today (`create_session()` is
still called exactly as before, with no `detector=` argument) — deliberately, per
requirement 7: NOVA's `TinyKWSNet` remains untrained, and `KeywordDetector` reports
`NOT_IMPLEMENTED` (not `None`, not a crash) for every chunk when no trained checkpoint
exists. Wiring it in by default would be harmless (sessions would just sit in
`WAITING_FOR_WAKE_WORD` forever) but would silently break every existing dev-testing
workflow that expects `/stop` to return whatever was streamed. Attaching the real
detector by default, once a trained checkpoint exists, is a one-line change in the
router (`create_session(sample_rate=..., detector=get_keyword_detector())`) — not made
in this stage, and intentionally not made until that checkpoint is real.

### 4.4 How pre-roll and post-detection audio are combined without duplication

Every chunk is written into `preroll` (an `AudioRingBuffer`) **before** it is (while
`WAITING_FOR_WAKE_WORD`) passed to the detector. When a detection fires on a given
chunk, that same chunk's samples are therefore already the newest samples inside
`preroll`. `assembly` is then seeded with exactly one call to `preroll.read()` — a
single snapshot — and the triggering chunk is **not** separately appended a second
time. Every chunk *after* that point is appended to `assembly` directly (not via
`preroll` again), so no sample is ever counted twice. This exact non-duplication
property is asserted directly in
`tests/unit/test_streaming_session.py::test_post_detection_audio_is_preserved_without_duplicating_the_boundary_chunk`.

Per requirement 5, it is expected and acceptable that this pre-roll snapshot contains
the spoken wake word itself (e.g. "...NOVA turn on the lights" rather than a perfectly
trimmed "turn on the lights") — `command/interpreter.py` already tolerates a leading
"nova" fragment (documented and verified in
`docs/NOVA_COMMAND_VOCABULARY.md`'s ambiguity #5), so no trimming logic was added here.

### 4.5 Session behaviour when stopped before any detection

If `/stop` (`close_session()`) is called while still `WAITING_FOR_WAKE_WORD`, `assembly`
is still empty (only `preroll` was ever fed), so ASR receives an empty waveform — the
same well-tested "empty stream" path already covered for no-detector sessions in
`tests/integration/test_stream_api.py::test_stop_with_zero_chunks_does_not_crash`. No
special-casing was added for this; it falls out of the state machine directly.

### 4.6 What is actually tested vs. what remains unverified

**Tested (`tests/unit/test_streaming_session.py`, all using fake/mock detectors, zero
dependency on real NOVA accuracy):**
- Chunks reach the detector as an independent consumer, alongside (not instead of) the
  pre-roll buffer.
- No detection → no wake event, `assembly` stays empty, state stays `WAITING_FOR_WAKE_WORD`.
- A mocked detection → `wake_word_detected_ts` and `wake_word_confidence` are recorded,
  state transitions to `COLLECTING_COMMAND`.
- Pre-roll is retrieved into `assembly` at the exact moment of detection (verified with
  a small, deterministic buffer capacity so the exact expected samples can be asserted).
- Post-detection audio keeps accumulating; the detection-boundary chunk's samples
  appear exactly once (no duplication).
- Multiple chunks before *and* after detection assemble into the expected full waveform,
  and the detector stops being consulted once it has fired once per session.
- Finalizing after detection preserves everything collected; finalizing before any
  detection yields an empty command-audio buffer, cleanly (not a crash).
- A detector that raises on every call (`RaisingDetector`) never crashes `add_chunk()`;
  the exception is captured in `session.detector_error`, the session stays in
  `WAITING_FOR_WAKE_WORD`, and it can still be finalized normally afterwards.

Additionally, **verified manually against the real (untrained)
`keyword_spotting.detector.KeywordDetector`** (not part of the automated suite, since it
depends on local model-file absence being the expected state, not a fixed assertion):
attaching a real `KeywordDetector()` instance — which reports `is_ready=False` and
`load_error` set because no trained `models/nova.onnx` exists — to a session and
feeding it chunks leaves the session safely parked in `WAITING_FOR_WAKE_WORD`, with
`detector_error is None` (the detector's own `NOT_IMPLEMENTED` result is a normal
return value, not an exception) and zero crashes.

**What remains completely unverified, and is NOT claimed anywhere in this integration:**
- Whether a trained TinyKWSNet or a trained openWakeWord "NOVA" head would actually
  fire `DETECTED` on real speech containing "NOVA" — `TinyKWSNet` is untrained and no
  `models/nova.onnx` checkpoint exists in this repo. **No NOVA detection accuracy claim
  is made anywhere in this milestone.**
- Whether the 0.5s pre-roll capacity is the right amount for a real detector's real
  reaction latency — that capacity constant (`DEFAULT_PREROLL_SECONDS`, unchanged this
  stage) was chosen before any real detector existed to react to.
- End-to-end behaviour with a *real* firing detector — the "detection actually happens
  and pre-roll actually gets used" path is only exercised with fake test doubles, by
  design (requirement 8/10).

---

## 5. Chunk size and framing

`audio_processing/pcm.py` computes chunk sizes for a configurable duration in
milliseconds. At 16kHz (the ASR-native rate; see `asr/transcriber.WHISPER_SAMPLE_RATE`):

| `chunk_ms` | samples/chunk | bytes/chunk (PCM16) |
|---|---:|---:|
| 20 ms | 320 | 640 |
| 30 ms (default) | 480 | 960 |
| 40 ms | 640 | 1280 |

`scripts/stream_demo.py` defaults to 30ms and accepts `--chunk-ms` to try other values
within the requested 20-40ms range. The final chunk of a waveform that isn't an exact
multiple of the chunk size is shorter than the rest — it is not zero-padded — and both
`audio_processing/pcm.frame_waveform` and the server's chunk-receiving endpoint
tolerate this (tested explicitly; see §8).

---

## 6. Development-machine simulation vs. the actual future ESP32/Wi-Fi implementation

This distinction is the single most important thing to get right in this document, so
it's stated plainly and repeated in the relevant code docstrings too.

| | **This stage (implemented, measured)** | **Future ESP32/Wi-Fi (not implemented)** |
|---|---|---|
| Client | `scripts/stream_demo.py`, a Python process on the same machine as the server, or (via `--mic`) using this machine's own microphone through `audio_processing.capture.MicCapture` | A physical ESP32 board with an I2S microphone, connected over real Wi-Fi |
| Transport | HTTP over `localhost` (loopback network interface) — no real network hop, no radio, no packet loss, no jitter | Real 802.11 Wi-Fi to a LAN/router, then to the server — genuinely variable latency, possible packet loss/retries, real radio power draw |
| Wire format | Raw little-endian PCM16 chunks over HTTP POST — **this format is designed to be ESP32-compatible already** (§3) | Same wire format is the plan; not yet verified against a real ESP32 HTTP client library (e.g. ESP-IDF's `esp_http_client` or Arduino's `HTTPClient`) |
| Server | `api/routers/stream.py`, same FastAPI process that also hosts ASR | Same server code is expected to work unmodified; not yet tested against a non-Python client |
| ASR | `asr/transcriber.py`'s faster-whisper, running **in the same process as the API server** on this dev machine | Intended to run on a genuinely separate, more capable remote/server machine (this has been the stated intent since the Stage 3 ASR milestone — this stage does not change that intent, it just doesn't yet have a second machine to prove it against) |
| Latency measured | Same-machine loopback: HTTP overhead is real (socket, ASGI routing, JSON serialization) but network transit time is ~0 | Unknown — genuinely requires a Wi-Fi hop, and has not been measured on any hardware |
| Wake-word detection | `api/routers/stream.py` attaches **no detector by default** — verified safe against the real (untrained) `KeywordDetector` via fake/mock test doubles only (§4) | Requires a trained NOVA checkpoint (`models/nova.onnx`) before a real detector could be attached and expected to actually fire — not part of this stage's scope at all |

**Per explicit instruction: no claim of real wireless latency is made anywhere in this
document, the code, or the demo script's own output.** `scripts/stream_demo.py` prints
its own reminder of this next to every latency number it reports.

---

## 7. What is measured, what is only estimated/planned

- **MEASURED** (this stage, on this dev machine, via `scripts/stream_demo.py` against a
  running `api.main` server; see §8 for the exact numbers from one real run):
  - Chunk framing correctness (chunk count, sizes, byte encoding) — via automated tests.
  - Circular/pre-roll buffer read/write/wraparound correctness — via automated tests.
  - Stream session lifecycle (start/chunk/stop, error handling) — via automated tests
    and a live server.
  - End-to-end real-speech transcription and command interpretation through the full
    streaming path (real faster-whisper, real `CommandInterpreter`) — via a live-server
    smoke test with synthesized speech (see §8).
  - Server-side stage latencies (ASR call duration, command-parse duration, total
    server processing time) on this dev machine's loopback path.
  - KWS-as-a-stream-consumer, pre-roll retrieval at detection, and non-duplication at
    the detection boundary (§4) — via automated tests using fake/mock detectors, plus a
    manual check that the real (untrained) `KeywordDetector` plugs in without crashing
    (§4.6).
- **ESTIMATED / architectural placeholder, not measured:**
  - The pre-roll buffer's *usefulness* for recovering clipped speech in a real
    detection — no *real* detector has ever fired yet to prove this end to end (§4.6).
  - Whether 0.5s of pre-roll is the right amount for a real detector's real reaction
    latency.
- **PLANNED, not implemented, not measured, and NOT CLAIMED at all:**
  - Real Wi-Fi transport latency, jitter, and packet loss.
  - ESP32 (or any other MCU) as an actual streaming client.
  - A genuinely separate remote ASR server process/machine (today ASR runs in the same
    process as the streaming endpoint that calls it).
  - **Any real NOVA wake-word detection accuracy whatsoever** — `TinyKWSNet` is
    untrained, no `models/nova.onnx` checkpoint exists, and this milestone's KWS
    integration was verified exclusively with fake/mock detectors plus a safety check
    against the real (currently non-functional, by design) `KeywordDetector`. See §4.6.

---

## 8. How to run the development streaming demo

```bash
# Terminal 1: start the API server
uvicorn api.main:app --port 8000

# Terminal 2: stream a WAV file through it
python scripts/stream_demo.py --wav path/to/command.wav --base-url http://127.0.0.1:8000

# Or stream 3 seconds live from the microphone:
python scripts/stream_demo.py --mic --duration 3 --base-url http://127.0.0.1:8000
```

### One real run performed during this stage (MEASURED)

Using Windows SAPI text-to-speech to synthesize "turn on the lights" (the same
technique already used elsewhere in this repo, e.g. `keyword_spotting/training/synth_data.py`,
to get real — if synthetic — speech without recording anything new), streamed at 30ms
chunks with `--no-realtime-pacing` against a locally-running server (real faster-whisper
`tiny.en`, real `CommandInterpreter`, no mocking):

```
Streaming 62 chunk(s) of ~30ms each (1.84s of audio total)...

Transcript : 'Turn on the lights.'
ASR status : OK
Action     : turn_on
Parameters : {'device': 'lights'}
Command    : OK - Matched rule-based intent 'turn_on'.

Chunks received       : 62
Audio duration        : 1.84 s
ASR latency           : 258.7 ms
Command-parse latency : 0.0 ms
Server total (create->command) : 320.8 ms
Client-observed total  : 328.6 ms
```

A second run with "turn on" (no device) correctly produced
`action=turn_on, missing_parameters=["device"]` end-to-end through the streaming path,
confirming the Stage 3 command-interpreter milestone's missing-parameter handling
carries through unmodified.

**Restated per §6: these are same-machine loopback numbers.** `command-parse latency`
reads as `0.0 ms` because the rule-based interpreter is sub-millisecond and the
timestamp resolution shown is rounded to one decimal place — not because parsing was
literally instantaneous.

---

## 9. Tests

`pytest -q` (full suite): **191 passed, 0 failed** (131 before this stage's streaming
transport work; 179 after the transport milestone; 191 after the KWS-integration
milestone — 12 further tests added, all in `tests/unit/test_streaming_session.py`).
New/extended test files, mapped to the task's required coverage:

| Requirement | Test file |
|---|---|
| Chunk framing | `tests/unit/test_pcm.py` |
| Circular/pre-roll buffer behavior | `tests/unit/test_ring_buffer.py` |
| Stream lifecycle (start/chunk/stop, double-stop, chunk-after-stop) | `tests/unit/test_streaming_session.py`, `tests/integration/test_stream_api.py` |
| Malformed/incomplete audio (odd byte count, empty body) | `tests/unit/test_pcm.py`, `tests/integration/test_stream_api.py` |
| Server receiving multiple chunks | `tests/integration/test_stream_api.py` |
| ASR integration (mocked, deterministic) | `tests/integration/test_stream_api.py` (`mock_asr` fixture patches `asr.transcriber.get_transcriber`, mirroring the mocking style already used in `tests/unit/test_asr_transcriber.py`) |
| Command interpretation integration | `tests/integration/test_stream_api.py` (uses the real, unmodified `CommandInterpreter` against mocked-ASR transcripts, including the missing-parameter and unknown-action cases) |
| Chunks reaching the KWS consumer | `tests/unit/test_streaming_session.py::test_chunks_reach_the_detector_as_an_independent_consumer`, `::test_detector_receives_chunks_regardless_of_preroll_also_receiving_them` |
| No detection → no wake event | `tests/unit/test_streaming_session.py::test_no_detection_leaves_session_waiting_with_no_wake_event` |
| Mocked detection → wake event | `tests/unit/test_streaming_session.py::test_mocked_detection_fires_wake_event_and_transitions_state` |
| Pre-roll retrieval at detection | `tests/unit/test_streaming_session.py::test_preroll_is_retrieved_into_assembly_at_the_moment_of_detection` |
| Post-detection audio preserved / no duplication at the boundary | `tests/unit/test_streaming_session.py::test_post_detection_audio_is_preserved_without_duplicating_the_boundary_chunk` |
| Multiple chunks before and after detection | `tests/unit/test_streaming_session.py::test_multiple_chunks_before_and_after_detection` |
| Finalization after detection / before detection | `tests/unit/test_streaming_session.py::test_finalization_after_detection_preserves_collected_command_audio`, `::test_session_stopped_before_detection_has_empty_command_audio` |
| Detector errors handled safely | `tests/unit/test_streaming_session.py::test_detector_exception_does_not_crash_add_chunk`, `::test_detector_exception_on_every_call_keeps_session_usable` |

---

## 10. Remaining limitations

- **Sessions never expire.** A session created via `/stream/start` stays in the
  server's in-memory dict until explicitly finalized via `/stop` (or forever, if the
  client crashes mid-stream) — there is no timeout/garbage collection. Acceptable for a
  short-lived dev-machine demo; would need a TTL or explicit cleanup endpoint before any
  longer-running deployment.
- **No backpressure or flow control.** The server accepts chunks as fast as they
  arrive; a misbehaving or malicious client could send unbounded chunks to one session
  with no size cap.
- **ASR runs synchronously inside the `/stop` request.** For a ~2s utterance this is
  a few hundred ms (§8) that the HTTP client simply waits on; no streaming/partial
  transcription exists — the full audio must be received before ASR starts.
- **Single-intent, single-utterance only**, inherited unchanged from the existing
  command interpreter (documented in `docs/NOVA_COMMAND_VOCABULARY.md`).
- **No authentication/authorization** on any streaming endpoint — fine for a
  `localhost` dev demo, not acceptable as-is for any networked deployment.
- **No detector is attached by default** (§4.3) — `/stream/start` has no way to opt
  into wake-word gating today; attaching one requires calling `create_session()`
  directly with a `detector=` argument (as the tests do), not through the HTTP API.
  This was a deliberate choice (requirement 7), not an oversight.
- **Single wake-word-per-session.** Once a detector fires, it is never consulted again
  for that session (`test_multiple_chunks_before_and_after_detection` asserts this
  directly) — there is no way to detect a second wake word within one stream.
- **No resampling between the session's sample rate and the detector's expected rate.**
  If a session were created with a `sample_rate` other than the detector's own
  (`KeywordDetector.sample_rate`, 16000 by default), chunks would be fed to it
  unresampled — the caller is responsible for matching rates, exactly as `KeywordDetector.detect`
  already documents for its own one-shot interface.
- **`AudioRingBuffer`'s capacity (0.5s) was not re-derived for this milestone** — it
  was chosen before any detector was wired in and has not been reconsidered against a
  real detector's real reaction latency (§4.6, §7).

---

## 11. Exact next recommended task

**Expose an explicit opt-in path to attach a real detector through the HTTP API**, once
(and only once) a real trained NOVA checkpoint (`models/nova.onnx`) actually exists —
e.g. a `use_kws: bool = False` query parameter on `POST /stream/start` that, when true,
resolves `keyword_spotting.detector.get_keyword_detector()` and passes it to
`create_session()`. This is deliberately **not done in this milestone** (§4.3) because
doing so today, before a real checkpoint exists, would make every gated session wait
forever for a detection that cannot come. Once real NOVA training produces a
checkpoint, this is a small, well-isolated change — and only then would it become
meaningful to also design a real end-to-end demo/test that exercises actual (not
mocked) wake-word detection over the streaming path. Until that checkpoint exists, this
task should stay blocked rather than being worked around with a placeholder model, per
this stage's explicit instructions.
