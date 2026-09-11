"""
keyword_spotting/training/record_data.py
============================================
Interactive real-microphone data collection for the "NOVA" wake word.

Records short clips from the default microphone one at a time, saving
each as a WAV file into data/kws/real/positive/ or
data/kws/real/negative/, plus a metadata.json describing each recording
session. This script needs an actual person present to speak -- it
cannot be run unattended.

Usage
-----
    python -m keyword_spotting.training.record_data --label positive --count 60
    python -m keyword_spotting.training.record_data --label negative --count 60
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

from audio_processing.capture import MicCapture, MicCaptureError
from keyword_spotting.features import DEFAULT_SAMPLE_RATE, DEFAULT_WINDOW_SECONDS

TARGET_SR = DEFAULT_SAMPLE_RATE  # keeps recordings at the exact rate the KWS pipeline expects
# Derived from features.py (not a separate literal) so a clip's raw length can
# never silently drift from the fixed-length window training/inference pad/trim
# to -- see keyword_spotting.features.fixed_length_waveform.
CLIP_SECONDS = DEFAULT_WINDOW_SECONDS

# Shown one at a time (cycling) for negative recordings, so each clip is a
# genuinely different word/phrase rather than the user repeating one word.
NEGATIVE_PROMPTS = [
    "hello", "computer", "assistant", "weather", "music", "light",
    "kitchen", "window", "morning", "please", "thank you", "goodbye",
    "yesterday", "tomorrow", "coffee", "table", "picture", "elephant",
    "umbrella", "notebook", "(stay silent -- say nothing this time)",
]


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


def countdown(seconds: float, label: str = "Next recording in") -> None:
    whole = int(round(seconds))
    print(f"{label} ", end="", flush=True)
    for remaining in range(whole, 0, -1):
        print(f"{remaining}...", end="", flush=True)
        time.sleep(1)
    print(flush=True)


def record_one_clip(mic: MicCapture, clip_seconds: float = CLIP_SECONDS) -> np.ndarray:
    # Discard chunks buffered while nothing was reading (e.g. during the
    # pause/countdown since the previous clip), so this capture is the
    # live 1.0s starting now rather than stale pre-prompt audio -- see
    # MicCapture.drain().
    mic.drain()
    n_needed = int(round(mic.sample_rate * clip_seconds))
    collected = []
    total = 0
    while total < n_needed:
        chunk = mic.read_chunk(timeout=5.0)
        collected.append(chunk)
        total += len(chunk)
    return np.concatenate(collected)[:n_needed].astype(np.float32)


def unique_path(out_dir: Path, session_id: str, index: int) -> Path:
    """Build a filename that can never silently overwrite an existing clip,
    even across separate recording sessions."""
    path = out_dir / f"real_{session_id}_{index:04d}.wav"
    suffix = 0
    while path.exists():
        suffix += 1
        path = out_dir / f"real_{session_id}_{index:04d}_{suffix}.wav"
    return path


def update_metadata(out_dir: Path, label: str, session_id: str, count: int, clip_seconds: float) -> None:
    meta_path = out_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    else:
        meta = {"type": "real", "label": label, "sample_rate": TARGET_SR, "sessions": []}
    meta["sessions"].append(
        {
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "count": count,
            "clip_seconds": clip_seconds,
        }
    )
    meta["total_recorded"] = sum(s["count"] for s in meta["sessions"])
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", choices=["positive", "negative"], required=True)
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--out-dir", type=str, default="data/kws/real")
    parser.add_argument("--clip-seconds", type=float, default=CLIP_SECONDS)
    parser.add_argument("--pause-seconds", type=float, default=1.2,
                         help="Pause between clips so you can prepare to speak.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    session_id = uuid.uuid4().hex[:8]

    print(f"Session ID    : {session_id}")
    print(f"Sample rate   : {TARGET_SR} Hz")
    print(f"Clip length   : {args.clip_seconds:.1f}s each, up to {args.count} clips")
    print(f"Inter-clip pause: {args.pause_seconds:.1f}s with a visible countdown")
    if args.label == "positive":
        print("Say the wake word: NOVA")
    else:
        print("A different word/phrase will be shown before each clip (mixed with some silence).")
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
    print("Starting in 3 seconds...")
    time.sleep(3)

    mic = MicCapture(sample_rate=TARGET_SR)
    try:
        mic.start()
    except MicCaptureError as exc:
        print(f"ERROR: could not start microphone: {exc}")
        return 1

    saved = 0
    try:
        for i in range(args.count):
            prompt = "NOVA" if args.label == "positive" else NEGATIVE_PROMPTS[i % len(NEGATIVE_PROMPTS)]
            print(f"[{i + 1}/{args.count}] Say: {prompt!r} ...", flush=True)
            audio = record_one_clip(mic, args.clip_seconds)
            out_path = unique_path(out_dir, session_id, i)
            sf.write(str(out_path), audio, TARGET_SR, subtype="PCM_16")
            saved += 1
            print(f"  saved -> {out_path.name}  (peak={float(np.max(np.abs(audio))):.3f})")
            if i < args.count - 1:
                countdown(args.pause_seconds)
    except KeyboardInterrupt:
        print("\nStopped early by user -- all clips saved so far are kept.")
    except MicCaptureError as exc:
        print(f"ERROR while recording: {exc}")
    finally:
        mic.stop()

    if saved > 0:
        update_metadata(out_dir, args.label, session_id, saved, args.clip_seconds)

    print(f"Done. Saved {saved} clip(s) to {out_dir} (session {session_id}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
