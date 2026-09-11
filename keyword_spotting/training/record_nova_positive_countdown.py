"""
keyword_spotting/training/record_nova_positive_countdown.py
=============================================================
Records COUNT positive "NOVA" wake-word clips with an explicit
"Recording X/COUNT -- Say: NOVA" prompt before each clip and a 2-second
recording window to say the word. No countdown between clips -- the
next prompt appears immediately after a clip is saved.

Before recording starts, verifies that the system default input device
is the known-good "Microphone Array (Realtek" device -- a prior session
recorded mostly silence because the default input had silently become a
disconnected/idle Bluetooth headset, so this check aborts early with a
clear message rather than wasting another 10-clip session.

Reuses the existing recording/saving machinery from record_data.py
unchanged: MicCapture, record_one_clip(), unique_path() (never
overwrites an existing file), update_metadata() (appends a session
entry to data/kws/real/positive/metadata.json). Saves only into the
existing positive directory. Never touches the negative directory and
never deletes anything.

Usage
-----
    python -m keyword_spotting.training.record_nova_positive_countdown
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from audio_processing.capture import MicCapture, MicCaptureError
from keyword_spotting.training.record_data import (
    TARGET_SR,
    record_one_clip,
    unique_path,
    update_metadata,
)
import uuid

COUNT = 5
LABEL = "positive"
OUT_DIR = Path("data/kws/real") / LABEL
# Wider than the 1.0s CLIP_SECONDS used elsewhere in the KWS pipeline, to give
# a real reaction window after the "Say: NOVA" prompt appears. NOTE: dataset
# loading (keyword_spotting/features.py fixed_length_waveform) trims a longer
# clip down to its *first* window_seconds by default, so say the word as soon
# as the prompt appears, not partway through the 2s window, or it may get cut
# off during feature extraction later.
RECORD_SECONDS = 2.0
EXPECTED_MIC_SUBSTRING = "Microphone Array (Realtek"


def check_default_microphone() -> tuple[bool, str]:
    """
    Look up the current system default input device name and report
    whether it matches the known-good Realtek microphone array.

    Returns (ok, name). If the device can't be queried at all (e.g.
    sounddevice/PortAudio unavailable), returns (False, "<error text>")
    so the caller can print it and abort rather than silently recording
    from an unknown/unverified device.
    """
    try:
        import sounddevice as sd  # type: ignore
        idx = sd.default.device[0]
        name = sd.query_devices(idx)["name"]
    except Exception as exc:  # pragma: no cover - environment-dependent
        return False, f"<could not query default input device: {exc}>"
    return EXPECTED_MIC_SUBSTRING in name, name


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    session_id = uuid.uuid4().hex[:8]

    print(f"Session ID    : {session_id}")
    print(f"Label         : {LABEL} (positive only -- no negatives will be recorded)")
    print(f"Output dir    : {OUT_DIR} (existing files are never overwritten or deleted)")
    print(f"Sample rate   : {TARGET_SR} Hz")
    print(f"Clip length   : {RECORD_SECONDS:.1f}s each, {COUNT} clips")
    print("Inter-clip delay: none -- next prompt appears immediately")
    print("Press Ctrl+C at any time to stop safely -- clips already saved are kept.")

    mic_ok, mic_name = check_default_microphone()
    print(f"Default input device: {mic_name}")
    if not mic_ok:
        print(
            f"ERROR: default input device does not look like the known-good "
            f"{EXPECTED_MIC_SUBSTRING!r} device. Refusing to record -- a prior "
            f"session silently recorded noise floor for this exact reason "
            f"(a disconnected/idle Bluetooth headset was default).\n"
            f"Fix: open Windows Sound settings -> Input and select "
            f"'Microphone Array (Realtek(R) Audio)' as the default device, "
            f"then re-run this script."
        )
        return 1
    print("Microphone check passed -- default input is the known-good Realtek array.")

    mic = MicCapture(sample_rate=TARGET_SR)
    try:
        mic.start()
    except MicCaptureError as exc:
        print(f"ERROR: could not start microphone: {exc}")
        return 1

    print("\nGet ready -- recording starts in ", end="", flush=True)
    for remaining in range(3, 0, -1):
        print(f"{remaining}...", end="", flush=True)
        time.sleep(1)
    print(flush=True)

    saved = 0
    try:
        for i in range(COUNT):
            print(f"\nRecording {i + 1}/{COUNT} -- Say: NOVA", flush=True)
            audio = record_one_clip(mic, RECORD_SECONDS)
            out_path = unique_path(OUT_DIR, session_id, i)
            sf.write(str(out_path), audio, TARGET_SR, subtype="PCM_16")
            saved += 1
            print(f"  saved -> {out_path.name}  (peak={float(np.max(np.abs(audio))):.3f})")
    except KeyboardInterrupt:
        print("\nStopped early by user -- all clips saved so far are kept.")
    except MicCaptureError as exc:
        print(f"ERROR while recording: {exc}")
    finally:
        mic.stop()

    if saved > 0:
        update_metadata(OUT_DIR, LABEL, session_id, saved, RECORD_SECONDS)

    print(f"\nDone. Saved {saved}/{COUNT} clip(s) to {OUT_DIR} (session {session_id}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
