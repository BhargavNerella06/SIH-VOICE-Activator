"""
keyword_spotting/training/record_nova_burst.py
=================================================
Records one continuous 10-second take of the "NOVA" wake word, repeated
several times with a short gap between each, then saves the raw burst
for offline splitting into individual clips.

Unlike record_nova_positive_countdown.py (discrete prompt-per-clip),
this captures one continuous stream -- simpler to time correctly when
run interactively, since there's only one "start now" moment instead of
several. Run this yourself (not via an automated tool call) so you see
the countdown live and can time your speech to it.

Verifies the system default input device is the known-good Realtek
microphone array before recording, same check as
record_nova_positive_countdown.py.

Saves only a single raw WAV file -- it does NOT write into
data/kws/real/positive/ and does NOT touch any existing recordings.
Splitting into individual ~1s clips is a separate, offline step.

Usage
-----
    python -m keyword_spotting.training.record_nova_burst
    python -m keyword_spotting.training.record_nova_burst --seconds 15 --out data/kws/raw/burst_02.wav
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from audio_processing.capture import MicCapture, MicCaptureError

TARGET_SR = 16000
DEFAULT_SECONDS = 10.0
DEFAULT_OUT = Path("data/kws/raw/nova_burst.wav")
EXPECTED_MIC_SUBSTRING = "Microphone Array (Realtek"


def check_default_microphone() -> tuple[bool, str]:
    try:
        import sounddevice as sd  # type: ignore
        idx = sd.default.device[0]
        name = sd.query_devices(idx)["name"]
    except Exception as exc:  # pragma: no cover - environment-dependent
        return False, f"<could not query default input device: {exc}>"
    return EXPECTED_MIC_SUBSTRING in name, name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=DEFAULT_SECONDS)
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    args = parser.parse_args()

    out_path = Path(args.out)
    if out_path.exists():
        print(f"ERROR: {out_path} already exists -- refusing to overwrite. "
              f"Pass --out with a different filename.")
        return 1
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mic_ok, mic_name = check_default_microphone()
    print(f"Default input device: {mic_name}")
    if not mic_ok:
        print(
            f"ERROR: default input device does not look like the known-good "
            f"{EXPECTED_MIC_SUBSTRING!r} device. Refusing to record.\n"
            f"Fix: open Windows Sound settings -> Input and select "
            f"'Microphone Array (Realtek(R) Audio)' as the default device, "
            f"then re-run this script."
        )
        return 1
    print("Microphone check passed.")

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

    print(f"\n>>> RECORDING NOW for {args.seconds:.0f}s -- say 'NOVA', pause briefly, "
          f"repeat <<<", flush=True)
    mic.drain()
    n_needed = int(round(TARGET_SR * args.seconds))
    collected = []
    total = 0
    start_wall = time.time()
    next_tick = 1
    try:
        while total < n_needed:
            chunk = mic.read_chunk(timeout=5.0)
            collected.append(chunk)
            total += len(chunk)
            elapsed = time.time() - start_wall
            if elapsed >= next_tick:
                remaining_s = max(0, args.seconds - elapsed)
                print(f"  ...{remaining_s:.0f}s left", flush=True)
                next_tick += 1
    except MicCaptureError as exc:
        print(f"ERROR while recording: {exc}")
        mic.stop()
        return 1
    finally:
        mic.stop()

    print("<<< RECORDING STOPPED >>>\n")

    audio = np.concatenate(collected)[:n_needed].astype(np.float32)
    sf.write(str(out_path), audio, TARGET_SR, subtype="PCM_16")
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    print(f"Saved -> {out_path}  duration={len(audio)/TARGET_SR:.2f}s  peak={peak:.4f}  rms={rms:.5f}")
    print("Next step: ask Claude to voice-activity-split this file into individual clips.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
