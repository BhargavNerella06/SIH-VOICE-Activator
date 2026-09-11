"""
command.interpreter
===================
CommandInterpreter - functional rule-based intent parser (Stage 3).

Status: FUNCTIONAL (rule-based, not a learned model)
------------------------------------------------------
Maps a plain-text transcript (typically produced by
asr.transcriber.Transcriber) to one of a small fixed set of structured
actions using keyword/regex matching plus simple slot extraction.

This is intentionally a small, inspectable rule set -- not a
general-purpose NLU/LLM. No network call, no training, no external
model weights: it is deterministic and fully local by construction.
Extending the action vocabulary means adding one pattern to
_INTENT_RULES below.

Honesty notes
-------------
- status is "OK" whenever the interpreter runs successfully, whether or
  not the text matched a known action -- "no known command matched" is
  a real, valid result (action="unknown"), not a placeholder failure.
- status is "ERROR" only if interpretation itself raised (e.g. an
  unexpected input type).
- No claim is made about how well this rule set covers real spoken
  NOVA commands: no real NOVA command vocabulary has been specified or
  collected yet. The action set below is a small illustrative example
  set for the end-to-end demonstration, not a finalized command list.

Missing-parameter handling (see docs/NOVA_COMMAND_VOCABULARY.md section 8)
---------------------------------------------------------------------------
turn_on/turn_off are matched in two tiers, still with plain regex (no
slot-filling framework):

1. A full match (verb + a recognized device) -- unchanged from before:
   `action=<intent>`, `parameters={"device": ...}`.
2. A verb-only match (the verb is clearly present, but no recognized
   device followed it -- either no device word at all, or one outside
   the supported enum) -- a new outcome distinct from both success and
   "unknown": `action=<intent>`, `parameters={}`,
   `missing_parameters=["device"]`. This lets a caller respond
   "turn on *what*?" instead of silently treating a half-understood
   command as if it were unrelated to any known command.

Completely unrelated/unrecognized text still returns action="unknown"
with missing_parameters=[] -- that field only ever appears for an
intent that was clearly recognized but under-specified.

Result schema: intent / device / original_text / confidence
----------------------------------------------------------------
`action` (e.g. "turn_on") is the specific rule that matched -- kept
exactly as-is for backward compatibility with existing callers
(api/routers/pipeline.py, voice_command.py, stream.py all read
`.action`/`.parameters`/`.missing_parameters` today). `intent` is a
coarser category derived from `action` (see _INTENT_CATEGORY) --
"device_control", "volume_control", "media_control", "emergency",
"session_control", or "unknown" -- matching the five categories in
docs/NOVA_COMMAND_VOCABULARY.md. `device` mirrors
`parameters.get("device")` as a top-level convenience field.
`original_text` preserves the exact input string as received, before
normalization. `confidence` is intentionally binary (1.0 / 0.0), not a
graded probability: this is a deterministic regex parser, not a
statistical model, and a fractional-looking confidence score would
misrepresent that. See docs/COMMAND_INTERPRETER.md for the full schema
and design rationale.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Pattern, Tuple

_DEVICE_GROUP = r"(?P<device>lights?|fan|music|alarm|tv|television|radio|speaker|ac|air\s?conditioner)"
_OPENABLE_DEVICE_GROUP = r"(?P<device>door)"

# Ordered, most-specific-first. Each entry is (action, compiled_regex).
# The first pattern that matches wins; named groups become `parameters`.
_INTENT_RULES: List[Tuple[str, Pattern[str]]] = [
    ("turn_on", re.compile(
        rf"\b(?:turn|switch)\s+on\s+(?:the\s+)?{_DEVICE_GROUP}\b", re.I)),
    ("turn_off", re.compile(
        rf"\b(?:turn|switch)\s+off\s+(?:the\s+)?{_DEVICE_GROUP}\b", re.I)),
    ("open", re.compile(
        rf"\bopen\s+(?:the\s+)?{_OPENABLE_DEVICE_GROUP}\b", re.I)),
    ("close", re.compile(
        rf"\bclose\s+(?:the\s+)?{_OPENABLE_DEVICE_GROUP}\b", re.I)),
    ("volume_up", re.compile(
        r"\b(?:turn\s+up|increase|raise)\s+(?:the\s+)?volume(?:\s+by\s+(?P<amount>\d+))?\b"
        r"|\bvolume\s+up\b", re.I)),
    ("volume_down", re.compile(
        r"\b(?:turn\s+down|decrease|lower)\s+(?:the\s+)?volume(?:\s+by\s+(?P<amount>\d+))?\b"
        r"|\bvolume\s+down\b", re.I)),
    ("mute", re.compile(r"\bmute\b", re.I)),
    ("unmute", re.compile(r"\bunmute\b", re.I)),
    ("play_music", re.compile(r"\bplay\s+(?:the\s+|some\s+)?music\b", re.I)),
    ("call_for_help", re.compile(
        r"\b(?:call\s+for\s+help|call\s+emergency|i\s+need\s+help|help\s+me)\b", re.I)),
    ("stop", re.compile(r"\b(?:stop|cancel|never\s*mind|abort)\b", re.I)),
]

# Verb-only fallback for intents that require a device slot. Checked only
# after every pattern in _INTENT_RULES has failed to match, so a full
# match (verb + valid device) always wins over this.
_DEVICE_INTENT_VERB_ONLY: List[Tuple[str, Pattern[str]]] = [
    ("turn_on", re.compile(r"\b(?:turn|switch)\s+on\b", re.I)),
    ("turn_off", re.compile(r"\b(?:turn|switch)\s+off\b", re.I)),
    ("open", re.compile(r"\bopen\b", re.I)),
    ("close", re.compile(r"\bclose\b", re.I)),
]

# Coarse category per action, matching docs/NOVA_COMMAND_VOCABULARY.md's
# five command categories. "unknown" (and any future action not listed
# here) falls back to intent="unknown" via .get().
_INTENT_CATEGORY: Dict[str, str] = {
    "turn_on": "device_control",
    "turn_off": "device_control",
    "open": "device_control",
    "close": "device_control",
    "volume_up": "volume_control",
    "volume_down": "volume_control",
    "mute": "volume_control",
    "unmute": "volume_control",
    "play_music": "media_control",
    "call_for_help": "emergency",
    "stop": "session_control",
}


def _normalize(text: str) -> str:
    """Lowercase, strip harmless punctuation, and collapse whitespace.

    Punctuation (commas, periods, exclamation marks, etc.) is replaced
    with a space rather than deleted outright, so "turn on, the light!"
    normalizes to "turn on the light" instead of accidentally fusing
    adjacent words together. Word characters (letters/digits/underscore)
    and whitespace are preserved untouched -- digits survive for the
    `amount` capture group.
    """
    text = text.strip().lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class CommandResult:
    status: str                          # "OK" | "ERROR"
    action: str = "unknown"
    parameters: Dict[str, object] = field(default_factory=dict)
    missing_parameters: List[str] = field(default_factory=list)
    message: str = ""
    intent: str = "unknown"
    device: Optional[str] = None
    original_text: str = ""
    confidence: float = 0.0


class CommandInterpreter:
    """
    Rule-based text -> structured-action interpreter.

    Stateless and cheap to construct (no model to load): every call to
    interpret() runs the same fixed regex rule set.
    """

    def interpret(self, text: str) -> CommandResult:
        """
        Parse a transcript into a structured action.

        Parameters
        ----------
        text : str
            Transcribed text, e.g. from asr.transcriber.Transcriber.

        Returns
        -------
        CommandResult
            action="unknown" if no rule matched (a valid outcome, not
            an error). If a device-control verb (turn on/off/open/close)
            is clearly recognized but no valid device followed it,
            action=<intent> with missing_parameters=["device"] instead
            of "unknown" -- see docs/NOVA_COMMAND_VOCABULARY.md section 8.
            status="ERROR" only on unexpected failures. See
            docs/COMMAND_INTERPRETER.md for the full result schema
            (intent/device/original_text/confidence).
        """
        original_text = text if isinstance(text, str) else str(text)
        try:
            if not text or not text.strip():
                return CommandResult(
                    status="OK",
                    action="unknown",
                    message="Empty transcript; nothing to interpret.",
                    intent="unknown",
                    original_text=original_text,
                    confidence=0.0,
                )
            normalized = _normalize(text)
            for action, pattern in _INTENT_RULES:
                match = pattern.search(normalized)
                if match:
                    parameters: Dict[str, object] = {
                        k: v for k, v in match.groupdict().items() if v is not None
                    }
                    if "amount" in parameters:
                        parameters["amount"] = int(parameters["amount"])
                    return CommandResult(
                        status="OK",
                        action=action,
                        parameters=parameters,
                        message=f"Matched rule-based intent '{action}'.",
                        intent=_INTENT_CATEGORY.get(action, "unknown"),
                        device=parameters.get("device"),
                        original_text=original_text,
                        confidence=1.0,
                    )

            for action, pattern in _DEVICE_INTENT_VERB_ONLY:
                if pattern.search(normalized):
                    return CommandResult(
                        status="OK",
                        action=action,
                        missing_parameters=["device"],
                        message=(
                            f"Recognized '{action}' but no supported device was "
                            "named."
                        ),
                        intent=_INTENT_CATEGORY.get(action, "unknown"),
                        device=None,
                        original_text=original_text,
                        confidence=1.0,
                    )

            return CommandResult(
                status="OK",
                action="unknown",
                message="No known command pattern matched this transcript.",
                intent="unknown",
                original_text=original_text,
                confidence=0.0,
            )
        except Exception as exc:
            return CommandResult(
                status="ERROR",
                message=f"Interpretation failed: {exc}",
                intent="unknown",
                original_text=original_text,
                confidence=0.0,
            )
