# SIH 2026 — PS 26172: Command Interpreter

**Scope:** `command/interpreter.py` — converting plain ASR transcript text into a
structured command. This document covers the interpreter **only**: its result schema,
matching logic, and normalization. It does not cover ASR (`docs/ASR_BACKEND.md`),
streaming (`docs/STAGE_3_STREAMING.md`), or the underlying command *vocabulary design*
rationale (`docs/NOVA_COMMAND_VOCABULARY.md`) — this document is about how the
interpreter works, those are about what it interprets and why those particular
commands were chosen.

**No accuracy claim is made.** This is a deterministic rule-based parser, not a
statistical or learned model — there is no accuracy to measure in the ML sense, only
whether a given input text matches one of the fixed patterns below.

---

## 1. Why deterministic parsing, not an LLM/NLU model

For the current SIH prototype's small, fixed command vocabulary (a handful of
device-control, volume, media, emergency, and session-control commands — see
`docs/NOVA_COMMAND_VOCABULARY.md`), a deterministic regex parser is deliberately
preferred over a learned NLU model or an LLM call:

- **No network, no cloud API, no training data required** — matches this project's
  local-first, offline-capable design (ASR already runs fully locally; the command
  layer should not reintroduce a network dependency for a problem regex already solves).
- **Fully inspectable and predictable** — every possible outcome for every input can be
  reasoned about by reading `_INTENT_RULES` top to bottom; a learned model's behavior on
  novel phrasing is not similarly auditable.
- **Zero latency and zero model-loading cost** — `CommandInterpreter` is stateless and
  free to construct; there is no model file, no warm-up, no inference cost.
- **Right-sized for the vocabulary's actual size.** A dozen or so intents with at most
  one required slot each does not need a slot-filling framework or an LLM — see
  `docs/NOVA_COMMAND_VOCABULARY.md`'s own "regex vs. slot-filling" conclusion, which
  reached the same verdict for the same reasons.

This is an explicit, revisitable design choice, not a limitation being worked around:
if the vocabulary grows to genuinely need free-form slot values, multi-turn
clarification, or robustness to much wider paraphrasing, that would be the point to
reconsider — not before.

---

## 2. Result schema

```python
@dataclass
class CommandResult:
    status: str                          # "OK" | "ERROR"
    action: str = "unknown"
    parameters: Dict[str, object] = {}
    missing_parameters: List[str] = []
    message: str = ""
    intent: str = "unknown"
    device: Optional[str] = None
    original_text: str = ""
    confidence: float = 0.0
```

| Field | Meaning |
|---|---|
| `status` | `"OK"` whenever interpretation ran successfully — including when nothing matched (`action="unknown"` is a valid outcome, not a failure). `"ERROR"` only if interpretation itself raised (e.g. a non-string input that can't be processed at all). |
| `action` | The specific rule that matched, e.g. `"turn_on"`, `"open"`, `"volume_up"`. `"unknown"` if nothing matched. Kept as the specific verb+effect (not collapsed to a generic "on"/"off") for backward compatibility with existing callers (`api/routers/pipeline.py`, `voice_command.py`, `stream.py`) that already key off these exact values. |
| `parameters` | Slot values captured by the matched rule, e.g. `{"device": "lights"}` or `{"amount": 20}`. Empty dict if the action takes no slots, or if a required slot is missing (see `missing_parameters`). |
| `missing_parameters` | `["device"]` if a device-control verb was clearly recognized but no valid device followed it (e.g. `"turn on"` alone) — see §5. Empty otherwise. |
| `message` | Human-readable explanation of what happened — which rule matched, or why nothing did. |
| `intent` | A coarser category derived from `action` — `"device_control"`, `"volume_control"`, `"media_control"`, `"emergency"`, `"session_control"`, or `"unknown"`. Matches the five categories documented in `docs/NOVA_COMMAND_VOCABULARY.md`. This is the field to check first if a caller only cares about the broad category, not the exact verb. |
| `device` | Convenience top-level mirror of `parameters.get("device")` — `None` for actions with no device slot (volume, media, emergency, session-control) or when the slot is missing. |
| `original_text` | The exact input string as received, **before** any normalization (§4) — preserved for logging/debugging/audit, not used for matching. |
| `confidence` | **`1.0`** if any rule matched (including a missing-parameter partial match — that's still a real, confident recognition of the verb, just an incomplete one), **`0.0`** if nothing matched or an error occurred. Deliberately binary: this is a deterministic pattern match, not a statistical model, so a fractional-looking score (e.g. `0.87`) would misrepresent what actually happened. Use `missing_parameters` to distinguish a full match from a partial one, not `confidence`. |

### Example (matches the task's own illustrative schema)

```python
>>> CommandInterpreter().interpret("turn on the light")
CommandResult(
    status="OK", action="turn_on", parameters={"device": "light"},
    missing_parameters=[], message="Matched rule-based intent 'turn_on'.",
    intent="device_control", device="light",
    original_text="turn on the light", confidence=1.0,
)
```

---

## 3. Supported vocabulary (current, small, illustrative)

| Phrase (examples) | `intent` | `action` | `device` |
|---|---|---|---|
| "turn on the light" / "switch on the lights" | device_control | `turn_on` | lights, fan, tv, television, radio, speaker, music, alarm |
| "turn off the fan" / "switch off the tv" | device_control | `turn_off` | (same enum as above) |
| "open the door" | device_control | `open` | door |
| "close the door" | device_control | `close` | door |
| "increase the volume [by N]" / "volume up" | volume_control | `volume_up` | — |
| "lower the volume [by N]" / "volume down" | volume_control | `volume_down` | — |
| "mute" | volume_control | `mute` | — |
| "unmute" | volume_control | `unmute` | — |
| "play music" / "play some music" | media_control | `play_music` | — |
| "help me" / "call for help" / "I need help" | emergency | `call_for_help` | — |
| "stop" / "cancel" / "never mind" / "abort" | session_control | `stop` | — |

The full rationale for *why these specific commands* (not a larger smart-home catalog),
category groupings, and known ambiguities is in `docs/NOVA_COMMAND_VOCABULARY.md` —
not duplicated here. `open`/`close` (door) were added this task, following the exact
same two-tier matching pattern already established for `turn_on`/`turn_off`, as a
concrete demonstration of requirement 6 (modular enough to extend without rewriting the
architecture).

---

## 4. Normalization

Before matching, input text is normalized (`command.interpreter._normalize`):

1. **Trim** leading/trailing whitespace, **lowercase**.
2. **Strip punctuation** — any character that isn't a letter, digit, underscore, or
   whitespace is replaced with a space (not deleted outright, so adjacent words don't
   get fused together — `"turn on, the light!"` → `"turn on the light"`, not
   `"turn onthe light"`).
3. **Collapse whitespace** — any run of spaces/tabs/newlines becomes a single space, and
   the result is trimmed again.

Digits are preserved throughout (needed for the `amount` slot in `"increase the volume
by 20"`). `original_text` on the result is the *pre-normalization* input, so callers can
always see exactly what was received.

This normalization was added this task; the underlying regex patterns already tolerated
repeated internal whitespace via `\s+` quantifiers, but did not tolerate punctuation
inserted between words (e.g. a comma) before this normalization step existed.

---

## 5. Matching logic

Rules are checked in a fixed order (`_INTENT_RULES`, most-specific-first):

1. **Full match** — a rule's complete pattern matches the normalized text (verb + a
   recognized device, or a device-less action like `"stop"`). First match wins; returns
   `action=<rule>`, `parameters` from the pattern's named capture groups,
   `missing_parameters=[]`, `confidence=1.0`.
2. **Verb-only fallback** (`_DEVICE_INTENT_VERB_ONLY`, only for `turn_on`/`turn_off`/
   `open`/`close`) — checked only if no full match was found. Recognizes the verb alone
   (no device, or a device outside the supported enum, e.g. `"turn on the couch"`).
   Returns `action=<rule>`, `parameters={}`, `missing_parameters=["device"]`,
   `confidence=1.0` — still a real, confident recognition of *intent*, just missing a
   required slot. See `docs/NOVA_COMMAND_VOCABULARY.md` §8 for the full rationale.
3. **Unknown** — neither tier matched. Returns `action="unknown"`, `intent="unknown"`,
   `confidence=0.0`.

---

## 6. Unknown-command behavior

Text that doesn't match any rule **never** guesses at the closest-sounding known
command. It returns `status="OK"` (interpretation itself succeeded — there's simply no
match), `action="unknown"`, `intent="unknown"`, `confidence=0.0`,
`missing_parameters=[]`. This is a deliberate, tested invariant
(`test_unmatched_text_returns_unknown_not_error`,
`test_unknown_text_has_unknown_intent_and_no_device`) — silently inventing a plausible
command for unrecognized input would be worse than admitting "I don't know what that
means."

---

## 7. Modularity: adding a new intent/device later

Adding a new device-control verb (following the `turn_on`/`open` pattern):
1. Add a full-match pattern to `_INTENT_RULES`.
2. Add its verb-only fallback to `_DEVICE_INTENT_VERB_ONLY` (only needed if the new verb
   has a required device slot).
3. Add its category to `_INTENT_CATEGORY`.

No other code changes — `CommandInterpreter.interpret()`, the result schema, and every
existing caller are unaffected. This was exercised directly this task by adding
`open`/`close` (door) this way.

---

## 8. Tests and smoke test

`tests/unit/test_command_interpreter.py` — **67 tests**: every supported command
(including the new `open`/`close`), case variations, whitespace variations (repeated
spaces, tabs/newlines), punctuation tolerance, unknown commands, empty text, malformed
(non-string) input, the missing-parameter partial-match cases, the new schema fields
(`intent`/`device`/`original_text`/`confidence`), and a documented ambiguous case
("turn on the music" matches `turn_on`, not `play_music`).

Standalone smoke test (no streaming, no API, no KWS):
```bash
# Text directly:
python scripts/command_smoke_test.py --text "turn on the light"

# Or chained after real ASR transcription:
python scripts/command_smoke_test.py --wav path/to/speech.wav
```

Verified this session, chained after real (Windows-SAPI-synthesized) speech saying
"close the door":
```
ASR transcript: 'Close the door.'

{
  "status": "OK",
  "action": "close",
  "parameters": {"device": "door"},
  "missing_parameters": [],
  "message": "Matched rule-based intent 'close'.",
  "intent": "device_control",
  "device": "door",
  "original_text": "Close the door.",
  "confidence": 1.0
}
```

---

## 9. Known limitations

- **Single intent per utterance** — "turn on the lights and play music" only matches the
  first rule found; the second clause is silently dropped (documented, not fixed —
  `docs/NOVA_COMMAND_VOCABULARY.md` ambiguity #6).
- **`confidence` is binary**, not a graded score — see §2's rationale. A caller wanting
  finer-grained partial-match information should use `missing_parameters` instead.
- **Small, illustrative vocabulary** — not a finalized command list; no real NOVA usage
  data has informed it yet (`docs/NOVA_COMMAND_VOCABULARY.md`).
- **API responses do not yet expose the new fields.** `api/routers/pipeline.py`,
  `voice_command.py`, and `stream.py` still only surface `action`/`parameters`/
  `missing_parameters` in their HTTP response schemas — `intent`/`device`/
  `original_text`/`confidence` exist on `CommandResult` but are not yet threaded through
  the API layer. Left as-is this task per explicit scope ("do not redesign unrelated API
  behavior"); a natural, minimal follow-up if the new fields prove useful to a caller.
