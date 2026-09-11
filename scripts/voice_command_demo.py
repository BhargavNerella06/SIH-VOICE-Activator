#!/usr/bin/env python
"""
scripts/voice_command_demo.py
================================
End-to-end laptop-demo console driver:

    microphone (or --wav) -> wake trigger -> AudioSessionManager
      -> ASR (faster-whisper) -> CommandInterpreter -> structured result

Two wake-trigger modes -- the rest of the pipeline is IDENTICAL after
the wake event either way (see voice_pipeline/orchestrator.py):

  --mock-wake   A deterministic, scripted trigger fires automatically
                after --mock-fire-after seconds of audio have been
                processed. THIS IS NOT NOVA DETECTION -- it exists
                purely so the rest of the pipeline can be demonstrated
                and tested while real NOVA recording/training remains
                postponed. Every log line involving it says "MOCK".

  --real-kws    Uses the real keyword_spotting.detector.KeywordDetector
                (TinyKWSNet-backed). With no trained
                models/kws_nova_cnn.pt checkpoint in this repo (the
                current, honest state of this project), it will listen
                forever and never fire -- this script prints a clear
                warning to that effect rather than pretending
                otherwise. Once a real checkpoint exists, this mode
                starts working with NO changes to this script.

Audio source
------------
By default, reads from the real microphone via
audio_processing.capture.MicCapture (requires `sounddevice` and an
actual input device). Pass --wav to instead play back a WAV file
(e.g. one synthesized via Windows SAPI TTS, as used elsewhere in this
project's demos/tests -- NOT a live recording) -- useful for
demonstrating or verifying this script without a microphone or a
person present.

Usage
-----
    python scripts/voice_command_demo.py --mock-wake
    python scripts/voice_command_demo.py --mock-wake --wav path/to/command.wav
    python scripts/voice_command_demo.py --real-kws
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from audio_processing.pcm import frame_waveform  # noqa: E402
from audio_processing.utils import ensure_mono, resample_if_needed  # noqa: E402
from voice_pipeline.orchestrator import (  # noqa: E402
    MockWakeTrigger,
    VoiceCommandOrchestrator,
    WakeMode,
)

SAMPLE_RATE = 16000
CHUNK_MS = 30.0


def _wav_chunk_source(
    path: str, mock_trigger: "MockWakeTrigger | None", fire_after_s: float, realtime_pacing: bool = True,
):
    """Yield chunks from a WAV file, auto-firing the mock trigger (if
    given) once `fire_after_s` seconds of audio have been yielded.

    Paced at real playback speed by default (like scripts/stream_demo.py)
    so "command capture duration" reflects real elapsed time, not the
    near-zero time it takes to just iterate the file's chunks in memory.
    """
    import soundfile as sf

    y, sr = sf.read(path)
    waveform = ensure_mono(np.asarray(y, dtype=np.float32))
    if sr != SAMPLE_RATE:
        waveform = resample_if_needed(waveform, sr, SAMPLE_RATE)

    chunks = frame_waveform(waveform, SAMPLE_RATE, chunk_ms=CHUNK_MS)
    elapsed_s = 0.0
    fired = False
    for chunk in chunks:
        if mock_trigger is not None and not fired and elapsed_s >= fire_after_s:
            mock_trigger.fire()
            fired = True
        yield chunk
        chunk_seconds = chunk.size / float(SAMPLE_RATE)
        if realtime_pacing:
            time.sleep(chunk_seconds)
        elapsed_s += chunk_seconds


def _mic_chunk_source(
    mock_trigger: "MockWakeTrigger | None", fire_after_s: float, chunk_size: int = 480,
    listen_timeout: "float | None" = None,
):
    """Yield chunks from a live microphone, auto-firing the mock trigger
    (if given) once `fire_after_s` seconds of real time have elapsed.

    `listen_timeout`, if given, stops yielding (a clean generator return,
    not an exception) once that many seconds have elapsed with no wake
    event -- so a --real-kws session where the checkpoint's confidence
    never crosses the threshold ends with a normal INCOMPLETE
    PipelineResult (see VoiceCommandOrchestrator._run_capture_phase)
    instead of hanging forever and needing Ctrl+C."""
    from audio_processing.capture import MicCapture, MicCaptureError

    mic = MicCapture(sample_rate=SAMPLE_RATE, chunk_size=chunk_size)
    try:
        mic.start()
    except MicCaptureError as exc:
        print(f"ERROR: could not start microphone: {exc}")
        raise SystemExit(1)

    start = time.perf_counter()
    fired = False
    try:
        while True:
            if listen_timeout is not None and (time.perf_counter() - start) >= listen_timeout:
                print(f"\n(listen-timeout of {listen_timeout}s reached with no wake event -- stopping)")
                return
            chunk = mic.read_chunk(timeout=5.0)
            if mock_trigger is not None and not fired and (time.perf_counter() - start) >= fire_after_s:
                mock_trigger.fire()
                fired = True
            yield chunk
    finally:
        mic.stop()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    wake_group = parser.add_mutually_exclusive_group(required=True)
    wake_group.add_argument("--mock-wake", action="store_true",
                             help="Use the deterministic mock wake trigger (NOT NOVA detection).")
    wake_group.add_argument("--real-kws", action="store_true",
                             help="Use the real KeywordDetector (TinyKWSNet). Will not fire "
                                  "without a trained models/kws_nova_cnn.pt checkpoint.")
    parser.add_argument("--wav", type=str, default=None,
                         help="Play back a WAV file instead of the live microphone.")
    parser.add_argument("--no-realtime-pacing", action="store_true",
                         help="With --wav, send all chunks back-to-back instead of pacing "
                              "them at real playback speed.")
    parser.add_argument("--mock-fire-after", type=float, default=1.0,
                         help="Seconds of audio before the mock trigger fires (--mock-wake only). Default: 1.0")
    parser.add_argument("--listen-timeout", type=float, default=15.0,
                         help="Max seconds to wait for a wake event on --real-kws with a live "
                              "microphone before giving up (WAV mode ignores this -- it stops "
                              "when the file ends). Default: 15.0")
    parser.add_argument("--kws-threshold", type=float, default=None,
                         help="Explicit override for the real KeywordDetector's confidence "
                              "threshold (--real-kws only). Does NOT change "
                              "keyword_spotting.detector's shipped default (0.85) -- this is an "
                              "opt-in, session-only override for demo purposes when the trained "
                              "checkpoint's measured confidence doesn't clear 0.85 on this small "
                              "dataset. Omit to use the production default.")
    args = parser.parse_args()

    wake_mode = WakeMode.MOCK if args.mock_wake else WakeMode.REAL_KWS
    mock_trigger = MockWakeTrigger() if wake_mode == WakeMode.MOCK else None

    detector = None
    if wake_mode == WakeMode.REAL_KWS:
        from keyword_spotting.detector import get_keyword_detector
        detector_kwargs = {}
        if args.kws_threshold is not None:
            detector_kwargs["threshold"] = args.kws_threshold
        detector = get_keyword_detector(**detector_kwargs)
        if args.kws_threshold is not None:
            print(f"NOTE: using --kws-threshold={args.kws_threshold} override "
                  f"(production default in keyword_spotting/detector.py is unchanged).")
        if not detector.is_ready:
            print("=" * 70)
            print("WARNING: --real-kws was requested, but no trained NOVA checkpoint")
            print(f"exists ({detector.load_error})")
            print("This session will listen but is NOT expected to ever detect a wake")
            print("word. Use --mock-wake for a working end-to-end demonstration.")
            print("=" * 70)
            print()

    orchestrator = VoiceCommandOrchestrator(
        wake_mode=wake_mode,
        detector=detector or mock_trigger,
        sample_rate=SAMPLE_RATE,
    )
    orchestrator.start()

    if args.wav:
        chunks = _wav_chunk_source(
            args.wav, mock_trigger, args.mock_fire_after,
            realtime_pacing=not args.no_realtime_pacing,
        )
    else:
        chunks = _mic_chunk_source(
            mock_trigger, args.mock_fire_after,
            listen_timeout=args.listen_timeout if wake_mode == WakeMode.REAL_KWS else None,
        )

    result = orchestrator.run_cycle(chunks)

    print()
    print("=== Result ===")
    print(f"Status         : {result.status}")
    if result.message:
        print(f"Message        : {result.message}")
    if result.command is not None:
        print(f"Transcript     : {result.transcript!r}")
        print(f"Intent         : {result.command.intent}")
        print(f"Action         : {result.command.action}")
        if result.command.device:
            print(f"Device         : {result.command.device}")
        if result.command.parameters:
            print(f"Parameters     : {result.command.parameters}")
        print(f"Confidence     : {result.command.confidence}")

    t = result.timings
    print()
    print("=== Latency (this dev machine only -- see docs/VOICE_COMMAND_PIPELINE.md) ===")
    print(f"Command capture duration : {t.command_capture_duration_ms} ms")
    print(f"ASR latency              : {t.asr_latency_ms} ms")
    print(f"Total wake-to-result     : {t.total_wake_to_result_ms} ms")
    print()
    print("NOTE: no claim of SIH final latency/hardware compliance is made -- this")
    print("runs entirely on this laptop's CPU, not the eventual edge target.")

    return 0 if result.status == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
