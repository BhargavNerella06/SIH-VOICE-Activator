"""
tests/unit/test_device_state.py
==================================
api.device_state is the central software device-state layer standing in
for IoT relay execution (no hardware) -- see that module's docstring.
"""

import pytest

from api.device_state import canonical_device, execute_device, get_state, reset_state


@pytest.fixture(autouse=True)
def _reset():
    reset_state()
    yield
    reset_state()


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("light", "light"),
        ("lights", "light"),
        ("Light", "light"),
        ("fan", "fan"),
        ("tv", "tv"),
        ("television", "tv"),
        ("ac", "ac"),
        ("air conditioner", "ac"),
        ("door", "door"),
        ("music", None),
        ("radio", None),
        (None, None),
        ("", None),
    ],
)
def test_canonical_device(raw, expected):
    assert canonical_device(raw) == expected


def test_default_state():
    state = get_state()
    assert state == {"light": "OFF", "fan": "OFF", "tv": "OFF", "ac": "OFF", "door": "CLOSED"}


@pytest.mark.parametrize(
    "device,action,expected_state",
    [
        ("light", "turn_on", "ON"),
        ("lights", "turn_off", "OFF"),
        ("fan", "turn_on", "ON"),
        ("tv", "turn_on", "ON"),
        ("television", "turn_off", "OFF"),
        ("ac", "turn_on", "ON"),
        ("air conditioner", "turn_off", "OFF"),
        ("door", "open", "OPEN"),
        ("door", "close", "CLOSED"),
    ],
)
def test_execute_device_valid_pairings(device, action, expected_state):
    result = execute_device(device, action)
    assert result is not None
    assert result["state"] == expected_state
    assert get_state()[result["device"]] == expected_state


def test_execute_device_updates_persist_across_calls():
    execute_device("light", "turn_on")
    execute_device("fan", "turn_on")
    state = get_state()
    assert state["light"] == "ON"
    assert state["fan"] == "ON"
    assert state["tv"] == "OFF"  # untouched devices keep their default


@pytest.mark.parametrize(
    "device,action",
    [
        ("music", "turn_on"),       # not a room device this UI shows
        ("light", "open"),          # power device can't be "opened"
        ("door", "turn_on"),        # door can't be "turned on"
        ("light", "volume_up"),     # non-device-control action
        (None, "turn_on"),
        ("light", None),
        ("unknown_device", "turn_on"),
    ],
)
def test_execute_device_rejects_invalid_pairings(device, action):
    assert execute_device(device, action) is None
    # A rejected pairing must never mutate state.
    assert get_state() == {"light": "OFF", "fan": "OFF", "tv": "OFF", "ac": "OFF", "door": "CLOSED"}


def test_reset_state_restores_defaults():
    execute_device("light", "turn_on")
    execute_device("door", "open")
    assert get_state() != {"light": "OFF", "fan": "OFF", "tv": "OFF", "ac": "OFF", "door": "CLOSED"}

    reset_state()
    assert get_state() == {"light": "OFF", "fan": "OFF", "tv": "OFF", "ac": "OFF", "door": "CLOSED"}
