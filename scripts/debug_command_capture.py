#!/usr/bin/env python
"""
scripts/debug_command_capture.py
====================================
TEMPORARY diagnostic tool that traces the exact audio stream from the
live microphone through wake detection into command capture, chunk by
chunk, so a truncated/corrupted/empty captured_command.wav can be
diagnosed instead of guessed at.

Drives audio_processing.session.AudioSessionManager and
keyword_spotting.detector.KeywordDetector directly (the same public
`process_chunk()` / `detect_chunk()` calls voice_pipeline.orchestrator
makes) -- it does NOT reimplement or modify either. This only adds
instrumentation *around* the existing, unmodified calls.

For every chunk it logs peak amplitude and RMS, and prints device
identity once at startup, so these questions are answered from actual
measurements rather than assumption:
  - Is the same microphone stream used before and after wake detection?
  - Does real audio keep arriving after NOVA fires (not silence/noise)?
  - Is a second stream accidentally opened?
  - Does the preroll snapshot handed into command capture contain real
    signal, or near-zero noise?
  - Are the exact command chunks appended into the WAV the same ones
    read from the mic (same peak/RMS), or has something corrupted them
    in between?

Usage
-----
    python scripts/debug_command_capture.py --kws-threshold 0.40
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from audio_processing.session import AudioSessionManager, AudioSessionState  # noqa: E402
from audio_processing.utils import save_wav_int16  # noqa: E402

SAMPLE_RATE = 16000


def _peak_rms(chunk: np.ndarray) -> "tuple[float, float]":
    if chunk.size == 0:
        return 0.0, 0.0
    peak = float(np.max(np.abs(chunk)))
    rms = float(np.sqrt(np.mean(np.square(chunk, dtype=np.float64))))
    return peak, rms


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kws-threshold", type=float, default=None,
                         help="Override KeywordDetector's confidence threshold for this run only "
                              "(does not change keyword_spotting/detector.py's shipped default).")
    parser.add_argument("--listen-timeout", type=float, default=20.0)
    parser.add_argument("--out-wav", type=str,
                         default=str(Path(__file__).resolve().parent / "_debug_output" / "captured_command.wav"))
    args = parser.parse_args()

    from keyword_spotting.detector import get_keyword_detector
    detector_kwargs = {}
    if args.kws_threshold is not None:
        detector_kwargs["threshold"] = args.kws_threshold
    detector = get_keyword_detector(**detector_kwargs)

    print("=" * 78)
    print("STAGE 0: detector")
    print("=" * 78)
    print(f"  is_ready              = {detector.is_ready}")
    print(f"  threshold             = {detector.threshold}")
    print(f"  consecutive_frames    = {detector.consecutive_frames}")
    print(f"  window_seconds        = {detector.window_seconds}")
    print(f"  hop_seconds           = {detector.hop_seconds}")
    if not detector.is_ready:
        print(f"  ERROR: {detector.load_error}")
        return 1

    session = AudioSessionManager(sample_rate=SAMPLE_RATE, detector=detector)
    print()
    print("=" * 78)
    print("STAGE 0: session config")
    print("=" * 78)
    print(f"  preroll_seconds       = {session.preroll_seconds}  ({session.preroll.capacity_samples} samples)")
    print(f"  min_command_seconds   = {session.min_command_seconds}")
    print(f"  silence_seconds       = {session.silence_seconds}")
    print(f"  max_command_seconds   = {session.max_command_seconds}")
    print(f"  vad_threshold         = {session.vad_threshold}")

    from audio_processing.capture import MicCapture, MicCaptureError
    import sounddevice as sd

    try:
        default_idx = sd.default.device[0]
        dev_info = sd.query_devices(default_idx)
        print()
        print("=" * 78)
        print("STAGE 1: microphone device")
        print("=" * 78)
        print(f"  default input index   = {default_idx}")
        print(f"  device name           = {dev_info['name']}")
        print(f"  device default sr     = {dev_info['default_samplerate']}")
        print(f"  device max_input_ch   = {dev_info['max_input_channels']}")
    except Exception as exc:
        print(f"  (could not query default device info: {exc})")

    mic = MicCapture(sample_rate=SAMPLE_RATE, chunk_size=480)
    try:
        mic.start()
    except MicCaptureError as exc:
        print(f"ERROR: could not start microphone: {exc}")
        return 1
    print(f"  MicCapture id()       = {id(mic)}  (single instance -- same object used for both "
          f"wake detection and command capture below)")
    print(f"  sample_rate           = {mic.sample_rate}  chunk_size = {mic.chunk_size}  "
          f"channels = {mic.channels}")

    print()
    print("=" * 78)
    print("Listening... say 'NOVA', pause briefly, then say your command.")
    print("=" * 78)
    print()

    session.start()
    t_start = time.perf_counter()
    n_listen_chunks = 0
    n_command_chunks = 0
    last_print = 0.0
    wake_seed_logged = False

    try:
        while True:
            if time.perf_counter() - t_start >= args.listen_timeout:
                print(f"\n(listen-timeout of {args.listen_timeout}s reached -- stopping)")
                break

            chunk = mic.read_chunk(timeout=5.0)
            chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
            peak, rms = _peak_rms(chunk)

            prev_state = session.state
            new_state = session.process_chunk(chunk)

            if prev_state == AudioSessionState.LISTENING:
                n_listen_chunks += 1
                now = time.perf_counter() - t_start
                if now - last_print >= 0.5:  # throttle LISTENING-phase spam
                    print(f"[LISTEN  #{n_listen_chunks:>4}] t={now:6.2f}s dtype={chunk.dtype} "
                          f"n={chunk.size} peak={peak:.4f} rms={rms:.5f}")
                    last_print = now

            if prev_state == AudioSessionState.LISTENING and new_state == AudioSessionState.COMMAND_CAPTURE:
                seed = session.preroll.read()
                s_peak, s_rms = _peak_rms(seed)
                print()
                print(f"*** WAKE DETECTED *** t={time.perf_counter() - t_start:.3f}s  "
                      f"confidence={session.wake_word_confidence}")
                print(f"    preroll seed handed into capture: n={seed.size} "
                      f"({seed.size / SAMPLE_RATE:.3f}s)  peak={s_peak:.4f}  rms={s_rms:.5f}")
                print()
                wake_seed_logged = True

            if new_state == AudioSessionState.COMMAND_CAPTURE:
                n_command_chunks += 1
                now = time.perf_counter() - t_start
                print(f"[CAPTURE #{n_command_chunks:>4}] t={now:6.2f}s n={chunk.size} "
                      f"peak={peak:.4f} rms={rms:.5f} "
                      f"silence_elapsed={session._silence_elapsed_seconds:.2f}s")

            if new_state == AudioSessionState.COMPLETE:
                break
            if new_state == AudioSessionState.ERROR:
                print(f"SESSION ERROR: {session.error_message}")
                break
    finally:
        mic.stop()

    print()
    print("=" * 78)
    print("RESULT")
    print("=" * 78)
    if not wake_seed_logged:
        print("Wake word was never detected in this run -- no command was captured.")
        return 0

    segment = session.result
    if segment is None:
        print("Session ended without completing a segment.")
        return 0

    full_peak, full_rms = _peak_rms(segment.waveform)
    print(f"end_reason            = {segment.end_reason}")
    print(f"sample count          = {segment.waveform.size}")
    print(f"duration              = {segment.duration_seconds:.3f}s")
    print(f"whole-segment peak    = {full_peak:.4f}")
    print(f"whole-segment rms     = {full_rms:.5f}")

    out_path = Path(args.out_wav)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_wav_int16(str(out_path), segment.waveform, segment.sample_rate)
    print(f"Saved WAV to          : {out_path.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
