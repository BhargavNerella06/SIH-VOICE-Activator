# SIH 2026 — PS 26172: NOVA Command Vocabulary (Design Document)

**Status: PARTIALLY IMPLEMENTED.** This document originated as a design-only proposal;
most of it (missing-parameter handling, and now `open`/`close`) has since been
implemented in `command/interpreter.py`. Each item below is still labeled explicitly as
either describing current behavior or a still-proposed/future item — this status line
is a summary, not a substitute for reading which is which section by section.

---

## 1. Purpose of the command system

`command/interpreter.py` turns one plain-text ASR transcript into one structured
action the rest of the system (or, eventually, a downstream device/actuator layer) can
execute deterministically. It is the last stage of the voice pipeline:

```
audio → KWS ("NOVA" wake word) → ASR (faster-whisper, dev-machine) → command interpreter → structured action
```

Design goals, in order of priority for this stage:

1. **Deterministic and inspectable.** A fixed, small rule set a reviewer can read
   top-to-bottom, not a learned/black-box classifier. No network call, no model
   weights, no training data.
2. **Honest about coverage.** Text that doesn't match a known command must come back
   as a clearly-labelled "unknown" result, never a guessed or fabricated action.
3. **Small and demo-realistic**, not a general-purpose smart-home assistant. Every
   command below is chosen because it is easy to demonstrate live and easy to justify
   to a judge, not because it's part of a larger planned catalog.

### Architectural invariant: NOVA is not part of the command text

NOVA is the **wake word**, detected upstream by `keyword_spotting.detector` from raw
audio, before ASR ever runs. By the time text reaches `CommandInterpreter.interpret()`,
the wake word has already served its purpose (triggering the ASR call) and is not
expected to appear in the transcript again.

**The command grammar below must never require the word "NOVA" inside the command
phrase itself.** A user says "NOVA, turn on the lights"; only "turn on the lights" (or
similar, depending on where KWS's detection window ends and ASR's transcription window
begins) is expected to reach the interpreter. This matches the current implementation:
none of the existing `_INTENT_RULES` patterns reference "nova".

One robustness question this raises (see §7 ambiguities): what if the ASR transcript
*does* still contain a leading "nova" / "hey nova" fragment, because the audio window
handed to ASR wasn't perfectly trimmed at the wake-word boundary? This is flagged as an
open ambiguity below, not silently assumed away.

---

## 2. Supported command categories

Five categories, chosen to cover a plausible minimal "voice activation" demo
(device control + media/volume control + a safety-oriented command + session control)
without inventing appliances or rooms nobody asked for:

| # | Category | Why it's in scope for the initial vocabulary |
|---|---|---|
| 1 | **Device control** | The most universally understood "voice assistant" action; easy to demo with a simulated light/fan. |
| 2 | **Volume control** | Natural, low-ambiguity commands; easy to demo against any audio output. |
| 3 | **Media control** | Minimal media playback control; deliberately kept to a single action for now (see §10). |
| 4 | **Safety / emergency** | Directly supports an accessibility/safety narrative for the PS, which is likely to matter most in front of judges. |
| 5 | **Session control** | A generic "stop/cancel" is needed regardless of which domain command was last issued. |

All five map directly onto the 9 intents already implemented in
`command/interpreter.py`'s `_INTENT_RULES` — this document formalizes and categorizes
them, it does not introduce a sixth category or new appliances.

---

## 3–5. Canonical phrases, intents, and required parameters

For each intent: canonical example phrases, the action name, and its parameters
("slots"). `required` parameters must be present for the action to be actionable;
`optional` parameters refine it but the action is still valid without them.

### 3.1 Device control

| Intent (action) | Canonical phrases | Parameters |
|---|---|---|
| `turn_on` | "turn on the lights", "switch on the fan", "turn on the tv" | `device` (**required**, enum: `lights`, `fan`, `tv`, `television`, `radio`, `speaker`, `alarm`, `music`) |
| `turn_off` | "turn off the lights", "switch off the fan" | `device` (**required**, same enum) |
| `open` | "open the door" | `device` (**required**, enum: `door`) — **implemented** (see `docs/COMMAND_INTERPRETER.md`) |
| `close` | "close the door" | `device` (**required**, same enum) — **implemented** |

### 3.2 Volume control

| Intent (action) | Canonical phrases | Parameters |
|---|---|---|
| `volume_up` | "turn up the volume", "increase the volume", "increase the volume by 20", "volume up" | `amount` (*optional*, integer) |
| `volume_down` | "turn down the volume", "lower the volume by 5", "volume down" | `amount` (*optional*, integer) |
| `mute` | "mute" | none |
| `unmute` | "unmute" | none |

`amount`'s unit is deliberately **undefined** in the current implementation (see §7 —
this is flagged, not resolved, here).

### 3.3 Media control

| Intent (action) | Canonical phrases | Parameters |
|---|---|---|
| `play_music` | "play music", "play the music", "play some music" | none |

Deliberately the *only* media action in the initial vocabulary — see §10 for why
`pause_music` / `stop_music` are future extensions, not omissions by accident.

### 3.4 Safety / emergency

| Intent (action) | Canonical phrases | Parameters |
|---|---|---|
| `call_for_help` | "call for help", "help me", "I need help", "call emergency" | none (see §7 for why no target/contact slot yet) |

### 3.5 Session control

| Intent (action) | Canonical phrases | Parameters |
|---|---|---|
| `stop` | "stop", "cancel", "cancel that", "never mind", "abort" | none (see §7 — target of "stop" is ambiguous) |

### 3.6 Fallback (not a category, but part of the contract)

| Intent (action) | When it fires | Parameters |
|---|---|---|
| `unknown` | Transcript is empty, or matches no rule above | none |

---

## 6. Example ASR transcript → parsed command

These are the exact shapes `CommandInterpreter.interpret()` already produces today
(verified against the current implementation, not proposed):

| ASR transcript | `action` | `parameters` |
|---|---|---|
| `"turn on the lights"` | `turn_on` | `{"device": "lights"}` |
| `"please turn off the fan"` | `turn_off` | `{"device": "fan"}` |
| `"increase the volume by 20"` | `volume_up` | `{"amount": 20}` |
| `"lower the volume by 5"` | `volume_down` | `{"amount": 5}` |
| `"mute"` | `mute` | `{}` |
| `"play some music"` | `play_music` | `{}` |
| `"help me"` | `call_for_help` | `{}` |
| `"cancel that"` | `stop` | `{}` |
| `"what's the weather today"` | `unknown` | `{}` |
| `""` (empty transcript, e.g. silence after the wake word) | `unknown` | `{}` |
| `"turn on"` (verb recognized, no device named) | `turn_on` | `{}`, `missing_parameters=["device"]` |
| `"turn on the couch"` (device word not in the supported enum) | `turn_on` | `{}`, `missing_parameters=["device"]` |
| `"open the door"` | `open` | `{"device": "door"}` |
| `"close the door"` | `close` | `{"device": "door"}` |

`CommandResult` also carries `intent`/`device`/`original_text`/`confidence` fields not
shown in the table above (added when the interpreter was extended with `open`/`close`)
— see `docs/COMMAND_INTERPRETER.md` for the full schema; this document stays focused on
vocabulary/category design, not the result-object shape.

---

## 7. Unknown commands

**Current behaviour (implemented today):** if no rule matches, `interpret()` returns
`status="OK"`, `action="unknown"`, `parameters={}`, and a message stating no pattern
matched. This is treated as a **valid, successful outcome of interpretation**, not a
failure — the interpreter did its job correctly; the user just didn't say a known
command. `status="ERROR"` is reserved strictly for the interpreter itself raising
(malformed input), never for "didn't understand".

**Recommended downstream handling (not this module's concern, but worth stating for
whoever consumes `CommandResult`):** an `action="unknown"` result should surface some
kind of "sorry, I didn't understand that" feedback to the user, not silently do
nothing with no acknowledgement. That UX decision belongs to the API/frontend layer
consuming `CommandResult`, not to the interpreter itself.

---

## 8. Missing required parameters

**Implemented.** `turn_on`/`turn_off` are matched in two tiers, still with plain regex
(no slot-filling framework):

1. A **full match** (verb + a recognized device) → unchanged from before:
   `action=<intent>`, `parameters={"device": ...}`, `missing_parameters=[]`.
2. A **verb-only match** — the verb is clearly present (`"turn on"`, `"switch off"`,
   etc.) but no recognized device followed it, either because no device word was said
   at all (`"turn on"`) or because a word was said that isn't in the supported device
   enum (`"turn on the couch"`) — checked only after every full-match rule in
   `_INTENT_RULES` has failed, so a full match always takes priority. Result:

   ```json
   {"status": "OK", "action": "turn_on", "parameters": {}, "missing_parameters": ["device"]}
   ```

   so a caller can prompt "Turn on *what*?" instead of silently treating a
   half-understood command as if the user had said something unrelated.

A completely unrelated sentence (`"the weather is nice today"`) still returns
`action="unknown"`, `missing_parameters=[]` — that field only ever appears for an
intent that was clearly recognized but under-specified, never for genuinely
unrecognized text. This distinction is covered by
`tests/unit/test_command_interpreter.py` (`test_device_intent_with_missing_device_reports_missing_parameters`,
`test_full_device_match_reports_no_missing_parameters`,
`test_unmatched_text_returns_unknown_not_error`).

**Scope of this fix:** only `turn_on`/`turn_off` got a missing-parameter outcome, since
they are the only intents with a *required* slot today (§3). `volume_up`/`volume_down`'s
`amount` is optional by design, so an omitted `amount` is not a missing-parameter case.
`missing_parameters` is not populated for any other intent.

---

## 9. Commands suitable for the SIH demonstration

Ranked by demo value (clarity to a live audience + reliability of the underlying ASR
step), not by implementation effort:

1. **`call_for_help`** — the strongest narrative fit for a voice-activation PS aimed
   at accessibility/safety; a single unambiguous phrase with no slot to get wrong.
2. **`turn_on` / `turn_off`** — the most universally recognizable "voice assistant"
   demo, easy to pair with a visible simulated light/fan indicator.
3. **`stop`** — trivial to demo (interrupts whatever is currently "active"), and
   necessary to show the system isn't a one-shot toy.
4. **`volume_up` / `volume_down` / `mute` / `unmute`** — solid secondary demo material;
   the optional `amount` slot is a nice "look, it parses numbers too" moment if judges
   ask follow-up questions.
5. **`play_music`** — lowest priority of the implemented set: convincing only if there
   is an actual audio output to visibly/audibly react, otherwise it's just another
   status message on screen.

## 10. Future extensions (explicitly out of scope for now)

Listed to make clear these were considered and deliberately deferred, not overlooked:

- **`pause_music` / `resume_music` / `stop_music`** — natural companions to
  `play_music`; deferred only because the current implementation has no playback
  *state* to pause/resume against (there's no "is music playing" tracking anywhere in
  this repo yet — adding these intents without that state would let the interpreter
  return a well-formed action that has nothing real to act on downstream).
- **Explicit volume level** ("set the volume to 50") vs. the current relative-only
  `volume_up`/`volume_down` — needs the unit-of-`amount` ambiguity in §7/below
  resolved first.
- **Room/location slots** ("turn on the kitchen lights") — real smart-home scope
  creep; deliberately excluded per the task's "don't invent a huge smart-home
  assistant" instruction.
- **Emergency contact targeting** ("call for help" → *whom*, specifically) — safety-
  critical enough that it deserves its own design pass (and likely a non-voice
  confirmation step), not a quick regex slot.
- **Confirmation / multi-turn slot-filling dialogue** ("Turn on what?" → user replies
  "the lights" → command completes) — needed to make §8's proposed
  `missing_parameters` outcome actually useful in a live conversation, not just
  reported in an API response.
- **Multilingual / accent-robust phrasing** — out of scope until there's a real
  target-user population to design against.

---

## Open ambiguities found while writing this document

1. **"stop" has no target.** Stop *what* — music playback, a pending device action, an
   in-progress emergency call, the whole listening session? Today it's a single
   context-free action; a real deployment likely needs either (a) an implicit "stop
   whatever is currently active" resolved by whatever holds session state downstream
   (not this module), or (b) an explicit target slot ("stop the music").
2. **`turn_on`/`turn_off` device enum overlaps `play_music`'s domain.** "Turn on the
   music" matches `turn_on` with `device="music"`; "play music" matches `play_music`
   with no device slot. These are two different actions for what a user would consider
   the same request. Needs a product decision: collapse "music" out of the device enum
   entirely (recommended), or unify both phrasings into one intent.
3. **`amount`'s unit is undefined.** "Increase the volume by 20" parses `amount=20`,
   but 20 *of what* — a 0–100 percentage point, an absolute level, an arbitrary step
   count? This is silently ambiguous today and needs a decision before any real
   volume-control backend consumes it.
4. **`call_for_help` has no target/contact/urgency slot.** Fine for a demo ("something
   happened, alert someone"), but a real safety feature would need to know *who* gets
   notified — deferred deliberately (§10), flagged here as a known gap, not an
   oversight.
5. **Potential wake-word leakage into the transcript.** If ASR's audio window isn't
   perfectly trimmed at the KWS detection boundary, the transcript could arrive as
   "nova turn on the lights" rather than "turn on the lights". The current regexes use
   `\b...\b` word-boundary anchors and `.search()` (not `.match()`), so a leading
   "nova " would **not** break matching today (verified: `"nova turn on the lights"`
   still matches `turn_on` under the current implementation) — but this has not been
   deliberately tested end-to-end from real KWS+ASR timing, only reasoned about from
   the regex semantics. Worth an explicit test once real KWS→ASR audio windowing is
   exercised together.
6. **No case for multiple intents in one utterance.** "Turn on the lights and play
   music" only ever matches the first rule found (`turn_on`, since it's ordered first)
   and silently drops the second clause. Not addressed here; flagged so it isn't
   mistaken for a bug later — it's a known single-intent-per-utterance limitation of
   the current architecture.

---

## Recommendation: regex vs. slot-filling

**Simple regex matching remains sufficient for this vocabulary size (9 intents, at
most one required slot each with a small closed enum).** A learned or grammar-based
slot-filling system would be over-engineering for a fixed 5-category, ~9-intent demo
vocabulary — it would add a dependency, a training/maintenance burden, and non-
determinism, for a problem regex already solves deterministically and transparently.

**The one real gap regex genuinely couldn't clean up on its own — distinguishing
"unrecognized sentence" from "recognized verb, missing required slot" — has now been
closed** (§8): a two-tier match (verb pattern, then a device pattern within the same
match) is still plain regex, just organized differently; no slot-filling framework was
introduced.

- Regex remains the mechanism, unchanged in kind.
- The two-tier match + `missing_parameters` field from §8 is implemented.
- Revisit slot-filling (a real one, e.g. a small state machine for multi-turn
  clarification) only if/when a future stage adds confirmation dialogues (§10) — not
  needed for the current single-shot transcript → action contract. `missing_parameters`
  is reported in `CommandResult`, but no interpreter-side dialogue/re-prompt loop
  exists yet; that remains a downstream/future concern.
