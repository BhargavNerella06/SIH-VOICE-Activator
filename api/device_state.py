"""
api.device_state
==================
Central, authoritative software device-state layer for the NOVA smart-room
demo.

For this laptop demo, physical IoT hardware (GPIO/ESP32/relay) is NOT
required -- see the frontend integration task's explicit instruction.
"Executing" a device command here means updating this in-memory state;
nothing pretends a physical relay was switched. Swapping in real hardware
later means only changing `execute_device()`'s body -- callers (the LIVE
voice pipeline and any future REST/WS handler) are unaffected.

This is the single source of truth for device state: both LIVE mode
(driven by api.live_session) and any backend-verified DEMO action funnel
through `execute_device()`, so the room UI never has to reconcile two
independent state copies.
"""

from __future__ import annotations

import threading
from typing import Dict, Optional

_LOCK = threading.Lock()

# The 5 devices the room UI renders. Values match what frontend/code.html's
# applyDeviceVisuals() expects per device (ON/OFF, or OPEN/CLOSED for door).
_DEFAULT_STATE: Dict[str, str] = {
    "light": "OFF",
    "fan": "OFF",
    "tv": "OFF",
    "ac": "OFF",
    "door": "CLOSED",
}

_state: Dict[str, str] = dict(_DEFAULT_STATE)

# Maps command.interpreter.CommandInterpreter's device vocabulary
# (_DEVICE_GROUP) onto this layer's 5 canonical device keys.
_DEVICE_ALIASES: Dict[str, str] = {
    "light": "light",
    "lights": "light",
    "fan": "fan",
    "tv": "tv",
    "television": "tv",
    "ac": "ac",
    "air conditioner": "ac",
    "airconditioner": "ac",
    "door": "door",
}

# action (from CommandResult.action) -> resulting device-state value.
_STATE_FOR_ACTION: Dict[str, str] = {
    "turn_on": "ON",
    "turn_off": "OFF",
    "open": "OPEN",
    "close": "CLOSED",
}

_DOOR_STATES = {"OPEN", "CLOSED"}
_POWER_STATES = {"ON", "OFF"}


def canonical_device(name: Optional[str]) -> Optional[str]:
    """Normalize a CommandInterpreter device string to one of this
    layer's 5 canonical keys, or None if it isn't one of them (e.g.
    'music', 'radio', 'speaker', 'alarm' -- real words the interpreter
    recognizes for other intents, but not room devices this UI shows)."""
    if not name:
        return None
    return _DEVICE_ALIASES.get(name.strip().lower())


def get_state() -> Dict[str, str]:
    with _LOCK:
        return dict(_state)


def execute_device(device: Optional[str], action: Optional[str]) -> Optional[Dict[str, object]]:
    """
    Apply one (device, action) pair -- as produced by
    command.interpreter.CommandInterpreter.interpret() -- to the
    software device-state layer.

    Returns
    -------
    dict with keys {"device", "state", "full_state"} describing exactly
    what changed, or None if this isn't a controllable room-device
    action (unknown device, or an action/device pairing that doesn't
    make sense -- e.g. "open" on the light, or an action from a
    non-device-control intent like volume/media/emergency). Callers
    should not broadcast a device_update when this returns None.
    """
    canon = canonical_device(device)
    new_value = _STATE_FOR_ACTION.get(action or "")
    if canon is None or new_value is None:
        return None
    if canon == "door" and new_value not in _DOOR_STATES:
        return None
    if canon != "door" and new_value not in _POWER_STATES:
        return None

    with _LOCK:
        _state[canon] = new_value
        return {"device": canon, "state": new_value, "full_state": dict(_state)}


def reset_state() -> Dict[str, str]:
    """Reset every device to its default state. Used by tests and by a
    fresh DEMO-mode session start."""
    global _state
    with _LOCK:
        _state = dict(_DEFAULT_STATE)
        return dict(_state)
