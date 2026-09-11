"""
tests/unit/test_command_interpreter.py
========================================
CommandInterpreter is a real, stateless, rule-based parser (Stage 3) --
no model, no network, no training. These tests exercise the rule set
and the "no rule matched" / error-handling fallbacks directly.
"""

import pytest

from command.interpreter import CommandInterpreter, CommandResult


@pytest.fixture
def interp() -> CommandInterpreter:
    return CommandInterpreter()


@pytest.mark.parametrize(
    "text,expected_action,expected_params",
    [
        ("turn on the lights", "turn_on", {"device": "lights"}),
        ("please turn off the fan", "turn_off", {"device": "fan"}),
        ("switch on the tv", "turn_on", {"device": "tv"}),
        ("switch off the radio", "turn_off", {"device": "radio"}),
        ("increase the volume", "volume_up", {}),
        ("increase the volume by 20", "volume_up", {"amount": 20}),
        ("turn down the volume", "volume_down", {}),
        ("lower the volume by 5", "volume_down", {"amount": 5}),
        ("mute", "mute", {}),
        ("please unmute", "unmute", {}),
        ("play some music", "play_music", {}),
        ("help me", "call_for_help", {}),
        ("stop", "stop", {}),
        ("cancel that", "stop", {}),
        ("open the door", "open", {"device": "door"}),
        ("please close the door", "close", {"device": "door"}),
        ("turn on the ac", "turn_on", {"device": "ac"}),
        ("turn off the ac", "turn_off", {"device": "ac"}),
        ("turn on the air conditioner", "turn_on", {"device": "air conditioner"}),
    ],
)
def test_known_intents_match(interp, text, expected_action, expected_params):
    """Regression coverage: every previously-supported intent must still
    match with the same action/parameters, and must not report a missing
    device (they are either fully matched or device-less by design)."""
    result = interp.interpret(text)
    assert result.status == "OK"
    assert result.action == expected_action
    assert result.parameters == expected_params
    assert result.missing_parameters == []


# ---------------------------------------------------------------------------
# Missing-parameter handling (docs/NOVA_COMMAND_VOCABULARY.md section 8)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text,expected_action",
    [
        ("turn on", "turn_on"),
        ("turn off", "turn_off"),
        ("please turn on", "turn_on"),
        ("switch off", "turn_off"),
        ("turn on the couch", "turn_on"),  # verb recognized, device not in the supported enum
        ("open", "open"),
        ("close", "close"),
        ("please open", "open"),
        ("open the window", "open"),  # verb recognized, "window" not in the openable enum
    ],
)
def test_device_intent_with_missing_device_reports_missing_parameters(interp, text, expected_action):
    result = interp.interpret(text)
    assert result.status == "OK"
    assert result.action == expected_action
    assert result.parameters == {}
    assert result.missing_parameters == ["device"]


def test_missing_device_message_is_informative(interp):
    result = interp.interpret("turn on")
    assert "turn_on" in result.message
    assert result.missing_parameters == ["device"]


@pytest.mark.parametrize("text", ["turn on the lights", "switch off the fan"])
def test_full_device_match_reports_no_missing_parameters(interp, text):
    """A full match (verb + valid device) must never report missing_parameters."""
    result = interp.interpret(text)
    assert result.missing_parameters == []
    assert result.parameters.get("device") is not None


def test_unmatched_text_returns_unknown_not_error(interp):
    """A completely unrelated sentence must stay 'unknown', never be
    mistaken for a device-control intent with a missing device."""
    result = interp.interpret("the weather is nice today")
    assert result.status == "OK"
    assert result.action == "unknown"
    assert result.missing_parameters == []


def test_empty_text_returns_unknown(interp):
    result = interp.interpret("")
    assert result.status == "OK"
    assert result.action == "unknown"


def test_whitespace_only_text_returns_unknown(interp):
    result = interp.interpret("   ")
    assert result.status == "OK"
    assert result.action == "unknown"


def test_is_case_insensitive(interp):
    result = interp.interpret("TURN ON THE LIGHTS")
    assert result.action == "turn_on"
    assert result.parameters == {"device": "lights"}


def test_matching_is_deterministic_and_stateless(interp):
    """Calling interpret() repeatedly must not carry state between calls."""
    first = interp.interpret("turn on the lights")
    second = interp.interpret("some unrelated sentence")
    third = interp.interpret("turn on the lights")
    assert first.action == "turn_on"
    assert second.action == "unknown"
    assert third.action == "turn_on"


def test_does_not_fabricate_action_on_error(interp):
    """If interpretation itself raises, status must be ERROR, not a fake action."""

    class _ExplodingStr(str):
        def strip(self):
            raise RuntimeError("simulated failure")

    result = interp.interpret(_ExplodingStr("turn on the lights"))
    assert result.status == "ERROR"
    assert result.action == "unknown"


# ---------------------------------------------------------------------------
# Result schema: intent / device / original_text / confidence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text,expected_intent,expected_device",
    [
        ("turn on the lights", "device_control", "lights"),
        ("turn off the fan", "device_control", "fan"),
        ("open the door", "device_control", "door"),
        ("close the door", "device_control", "door"),
        ("increase the volume", "volume_control", None),
        ("mute", "volume_control", None),
        ("play some music", "media_control", None),
        ("help me", "emergency", None),
        ("stop", "session_control", None),
    ],
)
def test_intent_and_device_fields(interp, text, expected_intent, expected_device):
    result = interp.interpret(text)
    assert result.intent == expected_intent
    assert result.device == expected_device


def test_unknown_text_has_unknown_intent_and_no_device(interp):
    result = interp.interpret("what a nice day")
    assert result.intent == "unknown"
    assert result.device is None


def test_missing_device_has_correct_intent_but_no_device(interp):
    result = interp.interpret("turn on")
    assert result.intent == "device_control"
    assert result.device is None
    assert result.missing_parameters == ["device"]


def test_original_text_preserves_exact_input(interp):
    raw = "  TURN ON, the LIGHTS!!  "
    result = interp.interpret(raw)
    assert result.original_text == raw  # not normalized/lowercased/stripped
    assert result.action == "turn_on"   # but matching still succeeded


def test_confidence_is_1_for_full_match(interp):
    result = interp.interpret("turn on the lights")
    assert result.confidence == 1.0


def test_confidence_is_1_for_missing_parameter_match(interp):
    """A recognized-but-incomplete command is still a real match, not a guess."""
    result = interp.interpret("turn on")
    assert result.confidence == 1.0


def test_confidence_is_0_for_unknown(interp):
    result = interp.interpret("the weather is nice today")
    assert result.confidence == 0.0


def test_confidence_is_0_for_empty_text(interp):
    result = interp.interpret("")
    assert result.confidence == 0.0


def test_confidence_is_0_on_error(interp):
    class _ExplodingStr(str):
        def strip(self):
            raise RuntimeError("simulated failure")

    result = interp.interpret(_ExplodingStr("turn on the lights"))
    assert result.confidence == 0.0


def test_command_result_is_importable_from_package_root():
    """command/__init__.py must export CommandResult, not just the interpreter class."""
    from command import CommandResult as PackageCommandResult
    assert PackageCommandResult is CommandResult


# ---------------------------------------------------------------------------
# Robust normalization: punctuation, repeated/irregular whitespace
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "turn on the lights",
        "Turn On The Lights",
        "TURN ON THE LIGHTS",
        "turn   on   the   lights",       # repeated internal spaces
        "\tturn on the lights\n",          # tabs/newlines at the edges
        "turn on the lights.",             # trailing period
        "turn on the lights!",             # trailing exclamation mark
        "turn on, the lights",             # comma mid-sentence
        "turn on the lights, please",      # trailing clause after a comma
        "  turn on the lights  ",          # leading/trailing spaces
    ],
)
def test_normalization_variants_all_match_turn_on(interp, text):
    result = interp.interpret(text)
    assert result.action == "turn_on"
    assert result.parameters == {"device": "lights"}
    assert result.status == "OK"


def test_normalization_does_not_corrupt_digits_in_amount(interp):
    result = interp.interpret("increase the volume by 20!")
    assert result.action == "volume_up"
    assert result.parameters == {"amount": 20}


def test_normalization_handles_apostrophes_harmlessly(interp):
    """Punctuation stripping must not crash on characters like apostrophes,
    even though no current rule depends on them."""
    result = interp.interpret("don't turn on the lights")
    # "don't" -> "don t" after normalization; the turn_on pattern still
    # matches "turn on the lights" within the remaining text.
    assert result.action == "turn_on"


# ---------------------------------------------------------------------------
# Malformed input
# ---------------------------------------------------------------------------

def test_non_string_input_does_not_crash(interp):
    """A non-string transcript is malformed input, not a crash-worthy bug."""
    result = interp.interpret(None)
    assert result.status in ("OK", "ERROR")
    assert result.action == "unknown"


def test_integer_input_does_not_crash(interp):
    result = interp.interpret(12345)
    assert result.status in ("OK", "ERROR")
    assert result.action == "unknown"


# ---------------------------------------------------------------------------
# Ambiguous commands (documented in docs/NOVA_COMMAND_VOCABULARY.md)
# ---------------------------------------------------------------------------

def test_turn_on_music_matches_device_control_not_media_control(interp):
    """Documented ambiguity: 'turn on the music' matches turn_on/device=music
    rather than play_music -- the device-control rules are checked first.
    This test pins down the current, documented behavior so a future change
    to rule ordering doesn't silently flip it."""
    result = interp.interpret("turn on the music")
    assert result.action == "turn_on"
    assert result.parameters == {"device": "music"}
    assert result.intent == "device_control"
