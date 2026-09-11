# SIH 2026 — PS 26172: Remote ASR Audio Streaming Architecture

**Scope:** `api/routers/asr_stream.py` (server) and
`voice_pipeline/remote_asr_client.py` (client) — the network boundary between the edge
device's command-audio capture and a remote ASR service. This document covers this
boundary **only**. Conv-TasNet, TinyKWSNet, `KeywordDetector`, `AudioSessionManager`,
`asr.transcriber`, and `command.interpreter` are all reused **unmodified** — see each
component's own doc for its internals.

**No NOVA detection accuracy is claimed anywhere in this document.** No trained
`models/kws_nova_cnn.pt` checkpoint exists; wake detection happens entirely upstream of
this module, on the edge, and isn't this document's concern.

---

## 1. Edge/server architecture

```
EDGE DEVICE                                    REMOTE ASR SERVER
------------                                   -------------------
Microphone
    |
    v
KWS (KeywordDetector / TinyKWSNet)
    |
    v
NOVA detected
    |
    v
AudioSessionManager
  (pre-roll, command-audio capture,
   silence/min/max-duration endpointing)
    |
    v
CommandAudioSegment (waveform, sample_rate)
    |
    v
RemoteASRClient.send_command_audio()  --ws://host:port/asr/stream-->  api/routers/asr_stream.py
                                                                            |
                                                                            v
                                                                    asr.transcriber (cached,
                                                                    UNMODIFIED faster-whisper)
                                                                            |
                                                                            v
                                                                    command.interpreter
                                                                    (UNMODIFIED)
                                                                            |
                                                                            v
                                        <--  {"type": "result", transcript, command, timings}
```

**Architecture boundary (kept exactly as instructed):**

| Side | Responsibility |
|---|---|
| **EDGE** | KWS + audio capture/session (`KeywordDetector`, `AudioSessionManager`). Nothing here changed. |
| **REMOTE** | ASR + command interpretation. Both run **once**, server-side, in `api/routers/asr_stream.py`. **Not duplicated on the client** — `RemoteASRClient` never imports `asr.transcriber` or `command.interpreter`; it only sends audio and displays whatever the server returns. |

For this development stage, both sides run **on the same laptop** via `127.0.0.1` — the
protocol itself makes no such assumption (see §6).

---

## 2. Existing streaming infrastructure discovered (and why a new endpoint, not a
reused one)

`api/streaming.py` (`StreamSession`) and `api/routers/stream.py` (`/stream/start`,
`/stream/{id}/chunk`, `/stream/{id}/stop`) already implement HTTP-POST-per-chunk audio
streaming with KWS-detector injection, pre-roll, and ASR+command-interpreter finalization
— built for an earlier task's different framing (a single server-side session doing wake
detection *and* transcription together, over discrete HTTP requests).

This task's target architecture is different in one load-bearing way: **wake detection
and command-audio endpointing now happen entirely on the edge**, via
`AudioSessionManager` (already implemented, unmodified). The remote server in this task's
diagram never sees raw continuous audio or runs KWS at all — it only ever receives one
already-gated command-audio segment per session. Reusing `StreamSession`'s
KWS-integrated design would mean carrying detector/pre-roll machinery the remote server
doesn't need, and reusing HTTP-POST-per-chunk would fight against this task's explicit
preference for a streaming-capable protocol. A new, deliberately smaller WebSocket
endpoint fits this narrower job better, per this task's own "keep scope controlled"
instruction — **`/stream/*` is untouched, still there for its own use case.**

What genuinely *is* reused, unmodified: `audio_processing.pcm` (the exact same raw
little-endian PCM16 wire format `/stream/*` already established — this project has one
audio-chunk encoding, not two), `asr.transcriber.get_transcriber()` (the same cache every
other ASR call site uses), and `command.interpreter.CommandInterpreter`.

---

## 3. Protocol

**Transport:** WebSocket (`fastapi`'s native `WebSocket` support — no new dependency;
`websockets` was already installed transitively via `uvicorn[standard]`). Chosen over
another HTTP-POST-per-chunk session (as `/stream/*` uses) because a single persistent
connection is a more natural fit for "stream audio, then get one result back," and
because the task explicitly asked for a streaming-capable protocol where the existing
infrastructure supports it cleanly — it does, via FastAPI.

**One WebSocket connection == one command-audio session.** No session-id or server-side
session store is needed (unlike `/stream/*`'s design, which needed one to correlate
discrete stateless HTTP POSTs) — the connection itself scopes the session; all state lives
in local variables inside the single `async def asr_stream(...)` handler and is discarded
when the connection closes.

### Message sequence

```
Client                                          Server
  |--- {"type":"start", "sample_rate":16000, -->|  (validates)
  |     "channels":1, "format":"pcm16"}         |
  |<-- {"type":"ready"}  or {"type":"error",...} |
  |                                              |
  |--- <binary PCM16 chunk> --------------------->|  (repeat N times)
  |--- <binary PCM16 chunk> --------------------->|
  |         ...                                  |
  |--- {"type":"end"} --------------------------->|  (explicit completion signal)
  |                                              |
  |<-- {"type":"result", "status":..., ---------- |
  |     "transcript":..., "command":{...},        |
  |     "timings":{...}}                          |
  |                                              (connection closes)
```

- **Sample rate, channel count, audio format** are all communicated in the `"start"`
  handshake — `sample_rate` (positive int, required), `channels` (must be `1` — mono
  only), `format` (must be `"pcm16"` — the only format currently supported).
- **Start/end of command** are explicit JSON control messages (`"start"` and `"end"`),
  not inferred from silence or a timeout — that decision already happened upstream, on
  the edge, via `AudioSessionManager`. The server trusts the client's `"end"` completely.
- **Audio chunks** are raw binary WebSocket frames, never wrapped in JSON/base64 and
  never sent as one complete WAV file — this is the streaming-not-batch design the task
  asked to demonstrate.

### Result schema

```json
{
  "type": "result",
  "status": "OK",
  "transcript": "turn on the lights",
  "command": {
    "status": "OK", "action": "turn_on", "parameters": {"device": "lights"},
    "missing_parameters": [], "intent": "device_control", "device": "lights",
    "original_text": "turn on the lights", "confidence": 1.0, "message": "..."
  },
  "timings": {
    "session_created_ts": 1234.5, "stream_start_ts": 1234.5, "stream_end_ts": 1234.6,
    "n_chunks": 58, "audio_duration_s": 1.71, "asr_start_ts": 1234.6,
    "asr_end_ts": 1234.9, "asr_latency_ms": 273.0, "command_interpreted_ts": 1234.9
  },
  "message": ""
}
```

`status`: `"OK"` | `"ASR_ERROR"` | `"EMPTY_AUDIO"` — relayed from the server's own
processing; `command` is `null` for the latter two (ASR was never reached, or failed).

---

## 4. Audio format / chunk duration

| | |
|---|---|
| Sample rate | 16,000 Hz (matches `asr.transcriber.WHISPER_SAMPLE_RATE`) |
| Channels | 1 (mono only) |
| Format | Raw little-endian signed 16-bit PCM (`audio_processing.pcm`) |
| Chunk duration | Configurable; `RemoteASRClient` defaults to `audio_processing.pcm.DEFAULT_CHUNK_MS` (30 ms → 480 samples → 960 bytes/chunk at 16 kHz) |

Chosen for exactly the same reasons `/stream/*` and `docs/STAGE_3_STREAMING.md` already
documented: PCM16 is what real microphone/ADC hardware (including a future ESP32)
actually produces, is half the bytes of float32, and needs no WAV-header bookkeeping —
appropriate for a constrained edge client.

---

## 5. Localhost development setup

```bash
# Terminal 1 -- start the ASR server (same app as /pipeline, /stream, etc.):
uvicorn api.main:app --port 8000

# Terminal 2 -- stream a WAV file to it (NOT a live microphone recording):
python scripts/remote_asr_client_demo.py --wav path/to/command.wav
python scripts/remote_asr_client_demo.py --wav path/to/command.wav --host 127.0.0.1 --port 8000
```

### Verified this session (real, Windows-SAPI-synthesized — not recorded — speech
saying "turn on the light")

```
Loaded 1.71s of audio from scratch_final_demo.wav (this is a WAV-file test, not a microphone recording)
Connecting to remote ASR server at ws://127.0.0.1:8322/asr/stream ...
Streaming command audio...
Server received audio
ASR transcription: "Turn on the light."
Intent: device_control
Device: light
Action: turn_on
Parameters: {'device': 'light'}
Confidence: 1.0

=== Latency (LOCALHOST DEVELOPMENT BENCHMARK) ===
Client streaming duration   : 4.5 ms
Client<->server round trip  : 297.9 ms (includes server ASR + interpretation time)
Server ASR latency          : 296.7 ms
Server audio duration       : 1.71 s (58 chunks)
Total client wall-clock time: 309.5 ms
```

All error paths were also verified live against a running server: connection refused
(wrong port) → `CONNECTION_ERROR`; missing/invalid `sample_rate`, unsupported `format`,
wrong first-message `type`, multi-channel audio → `{"type": "error", ...}`; a malformed
(odd-byte-count) chunk mid-stream → `{"type": "error", ...}`, connection closed cleanly.

---

## 6. Localhost transport vs. eventual Wi-Fi/ESP32 transport

| | **This stage (implemented, measured)** | **Eventual ESP32 → Wi-Fi (not implemented)** |
|---|---|---|
| Client | `voice_pipeline/remote_asr_client.py`, a Python process on the same machine as the server | A physical ESP32 (or similar) edge device, running `AudioSessionManager`'s eventual embedded-C equivalent |
| Transport | WebSocket over `127.0.0.1` (loopback) — no real network hop, no radio, no packet loss, no jitter | Real 802.11 Wi-Fi to a LAN/router, then to the server — genuinely variable latency, possible reconnects/packet loss |
| Wire format | Raw little-endian PCM16 over a WebSocket binary frame — **already designed to be ESP32-compatible** (no WAV headers, no base64, minimal per-chunk overhead) | Same wire format is the plan; not yet verified against a real embedded WebSocket client library |
| Server | `api/routers/asr_stream.py`, same FastAPI process already hosting `/pipeline`, `/stream`, etc. | Same server code is expected to work unmodified — only the client and the network path change |
| Latency measured | Real wall-clock timings, but network transit time is ~0 (loopback) | Unknown — genuinely requires a Wi-Fi hop, not measured on any hardware |

**Per explicit instruction: no claim of final Wi-Fi/wireless latency is made anywhere in
this document, the code, or the demo scripts' own output.** Every latency line printed by
`scripts/remote_asr_client_demo.py` is explicitly labeled "LOCALHOST DEVELOPMENT
BENCHMARK."

---

## 7. Latency measurements (what each number means, and its limits)

| Metric | Where measured | What it captures |
|---|---|---|
| `client streaming duration` | Client, around the chunk-sending loop | Time to hand all chunks to the local OS socket buffer — on loopback this is near-instant regardless of the audio's real duration; **not** meaningful as a "how long did speaking take" measurement (that's `server audio_duration_s`, computed from sample count instead). |
| `client<->server round trip` | Client, from sending `"end"` to receiving `"result"` | Bundles network transit **and** server ASR+interpretation time together — cannot be cleanly separated from the client's vantage point alone (see `RemoteASRTimings.server_round_trip_ms`'s own docstring). |
| `server asr_latency_ms` | Server, around the `transcribe()` call only | The actual faster-whisper inference time — directly comparable to the same metric already reported in `docs/ASR_BACKEND.md` and `docs/VOICE_COMMAND_PIPELINE.md`. |
| `server audio_duration_s` | Server, from total received sample count | The real duration of the command audio, independent of transport speed. |
| `total client wall-clock time` | Client, start to finish including connection setup | The full user-facing latency for this localhost run — still not a network latency measurement, since there's no real network here. |

A first-call ASR warm-up cost (~2.2–2.4s, observed and documented in
`docs/VOICE_COMMAND_PIPELINE.md` §5) applies here identically, since this endpoint calls
the same cached `get_transcriber()` — the very first command sent to a freshly-started
server pays that cost once; subsequent commands do not.

---

## 8. Tests

`tests/integration/test_asr_stream.py` — **20 tests**, using FastAPI's synchronous
`TestClient.websocket_connect()` (no live process, no real socket, no microphone) with
mocked ASR (`monkeypatch`ing `get_transcriber`, mirroring `tests/integration/
test_stream_api.py`'s established pattern) and the real, unmodified `CommandInterpreter`:

- protocol metadata (valid start, missing/invalid `sample_rate`, wrong first message
  type, unsupported format, multi-channel rejection);
- malformed metadata beyond simple field checks (non-JSON text, malformed control
  message, unexpected message type mid-stream);
- chunk ordering (a monotonic-ramp waveform reconstructed correctly) and a complete
  session end-to-end;
- successful transcription **and** command interpretation together (both real intents
  and the missing-parameter/unknown-command cases already covered by `docs/
  COMMAND_INTERPRETER.md`);
- empty audio (no chunks sent) short-circuits before ASR is ever called;
- server ASR failure (`ERROR` and `NOT_IMPLEMENTED`) surfaces as `ASR_ERROR`, never a
  crash;
- a malformed (odd-byte-count) chunk mid-stream is rejected cleanly;
- client-side connection failure (nothing listening on the target port) reports
  `CONNECTION_ERROR`, never raises;
- ASR model caching — the same transcriber instance handles 3 sequential sessions (no
  reload per command);
- multiple sequential sessions are fully independent (no state leakage between them).

Full project suite: **331 passed, 0 failed** (311 before this task; 20 new).

---

## 9. Known limitations

- **No trained NOVA checkpoint** — this entire module operates strictly downstream of
  wake detection and makes no NOVA claim of any kind.
- **Localhost only, verified this session.** The protocol is designed to be
  transport-agnostic (§6), but has only been exercised over loopback — a real Wi-Fi hop
  (jitter, reconnection, partial-frame handling under packet loss) is unverified.
- **No reconnection/resume logic.** If the connection drops mid-stream, the session is
  simply lost — the client reports `CONNECTION_ERROR` and the caller would need to start
  a new session from scratch. Acceptable for a single short command-audio segment; would
  need revisiting for a lossier real Wi-Fi link.
- **No authentication** — matches the rest of this project's existing endpoints
  (`/pipeline`, `/stream`, `/voice-command`), none of which have any; not added here
  either, per this task's explicit "don't add unless already required" instruction.
- **Single mono PCM16 format only** — no negotiation, no alternate sample rates accepted
  from the client (the server doesn't resample; it trusts the declared `sample_rate`
  matches the actual chunk content).
- **The client and server both live in this same repository/process family** for this
  stage — running the server on a genuinely different machine has not been tested,
  though nothing in the protocol assumes same-machine operation (`host`/`port` are
  already client-configurable, defaulting to `127.0.0.1:8000`).

---

## 10. Recommended next step

Once a real trained NOVA checkpoint exists and the end-to-end local orchestrator
(`voice_pipeline.orchestrator.VoiceCommandOrchestrator`, `docs/VOICE_COMMAND_PIPELINE.md`)
is wired to call `RemoteASRClient.send_command_audio()` instead of `asr.transcriber`
directly, the natural validation step is: run the server and client on two genuinely
different machines on the same LAN (still not ESP32, but a real network hop) to get a
first non-loopback latency data point — before attempting a real wireless/ESP32 client.
