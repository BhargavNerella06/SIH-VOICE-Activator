#!/usr/bin/env python
"""
scripts/stream_demo.py
========================
Stage 3 development-machine streaming client simulator.

Plays a WAV file (or a live microphone capture) back as a sequence of
small PCM16 chunks, POSTed one at a time to the /stream/* endpoints on
a running api.main FastAPI server, then prints the transcript,
structured command, and the full server-side latency breakdown.

IMPORTANT -- what this does and does not demonstrate
-------------------------------------------------------
This is a same-machine (usually localhost) HTTP client talking to a
server process on the SAME machine. It exercises the real chunk-
framing / streaming-session / ASR / command-interpretation code path
end to end, but it does NOT exercise a real wireless link, a real
ESP32 client, or real network latency/jitter/packet loss. Any timing
this script prints is a development-machine loopback measurement, not
a wireless latency claim. See docs/STAGE_3_STREAMING.md.

Usage
-----
    # Terminal 1:
    uvicorn api.main:app --port 8000

    # Terminal 2, stream a WAV file:
    python scripts/stream_demo.py --wav path/to/command.wav

    # Or stream `--duration` seconds live from the microphone:
    python scripts/stream_demo.py --mic --duration 3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from audio_processing.pcm import DEFAULT_CHUNK_MS, float32_to_pcm16_bytes, frame_waveform  # noqa: E402
from audio_processing.utils import ensure_mono, resample_if_needed  # noqa: E402

STREAM_SAMPLE_RATE = 16000  # matches asr.transcriber.WHISPER_SAMPLE_RATE


def _load_wav(path: str) -> np.ndarray:
    y, sr = sf.read(path)
    y = ensure_mono(np.asarray(y, dtype=np.float32))
    if sr != STREAM_SAMPLE_RATE:
        y = resample_if_needed(y, sr, STREAM_SAMPLE_RATE)
    return y.astype(np.float32)


def _record_from_mic(duration_s: float) -> np.ndarray:
    from audio_processing.capture import MicCapture, MicCaptureError

    mic = MicCapture(sample_rate=STREAM_SAMPLE_RATE)
    try:
        mic.start()
    except MicCaptureError as exc:
        print(f"ERROR: could not start microphone: {exc}")
        raise SystemExit(1)

    print(f"Recording {duration_s:.1f}s from the microphone...")
    chunks = []
    start = time.perf_counter()
    try:
        while (time.perf_counter() - start) < duration_s:
            chunks.append(mic.read_chunk(timeout=2.0))
    finally:
        mic.stop()
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)


def run(
    waveform: np.ndarray,
    base_url: str,
    chunk_ms: float,
    realtime_pacing: bool,
) -> dict:
    client = httpx.Client(base_url=base_url, timeout=30.0)

    t_client_start = time.time()
    resp = client.post("/stream/start", params={"sample_rate": STREAM_SAMPLE_RATE})
    resp.raise_for_status()
    session = resp.json()
    session_id = session["session_id"]
    print(f"Stream session started: {session_id}")

    chunks = frame_waveform(waveform, STREAM_SAMPLE_RATE, chunk_ms=chunk_ms)
    print(f"Streaming {len(chunks)} chunk(s) of ~{chunk_ms:.0f}ms each "
          f"({len(waveform) / STREAM_SAMPLE_RATE:.2f}s of audio total)...")

    for chunk in chunks:
        capture_ts = time.time()  # "audio capture/chunk creation" timestamp
        payload = float32_to_pcm16_bytes(chunk)
        send_ts = time.time()     # "stream send" timestamp
        resp = client.post(
            f"/stream/{session_id}/chunk",
            params={"capture_ts": capture_ts, "send_ts": send_ts},
            content=payload,
            headers={"Content-Type": "application/octet-stream"},
        )
        resp.raise_for_status()
        if realtime_pacing:
            # Simulate a live microphone: don't send faster than the audio
            # itself plays out. Dev-machine-only pacing, not a network
            # simulation -- see this script's module docstring.
            time.sleep(chunk_ms / 1000.0)

    resp = client.post(f"/stream/{session_id}/stop")
    resp.raise_for_status()
    result = resp.json()
    t_client_done = time.time()
    result["_client_observed_total_ms"] = (t_client_done - t_client_start) * 1000.0
    return result


def _print_summary(result: dict) -> None:
    print()
    print("=== Result ===")
    print(f"Transcript : {result['transcript']!r}")
    print(f"ASR status : {result['asr']['status']}")
    print(f"Action     : {result['action']}")
    print(f"Parameters : {result['parameters']}")
    if result.get("missing_parameters"):
        print(f"Missing    : {result['missing_parameters']}")
    print(f"Command    : {result['command']['status']} - {result['command']['message']}")

    timing = result["timing"]
    print()
    print("=== Server-side latency breakdown (dev-machine / loopback only) ===")
    print(f"Chunks received       : {timing['n_chunks']}")
    print(f"Audio duration        : {timing['audio_duration_s']:.2f} s")
    print(f"ASR latency           : {timing['asr_latency_ms']:.1f} ms")
    print(f"Command-parse latency : {timing['command_latency_ms']:.1f} ms")
    print(f"Server total (create->command) : {timing['server_total_ms']:.1f} ms")
    print(f"Client-observed total  : {result['_client_observed_total_ms']:.1f} ms")
    print()
    print("NOTE: all of the above is a same-machine HTTP loopback measurement.")
    print("It is not a wireless/ESP32 latency measurement -- see")
    print("docs/STAGE_3_STREAMING.md.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wav", type=str, default=None, help="WAV file to stream.")
    parser.add_argument("--mic", action="store_true", help="Record live from the microphone instead.")
    parser.add_argument("--duration", type=float, default=3.0, help="Seconds to record if --mic is given.")
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--chunk-ms", type=float, default=DEFAULT_CHUNK_MS,
                         help=f"Chunk duration in ms (default: {DEFAULT_CHUNK_MS}).")
    parser.add_argument("--no-realtime-pacing", action="store_true",
                         help="Send all chunks back-to-back instead of pacing them "
                              "at real playback speed.")
    args = parser.parse_args()

    if args.mic:
        waveform = _record_from_mic(args.duration)
    elif args.wav:
        waveform = _load_wav(args.wav)
    else:
        parser.error("Provide either --wav <file> or --mic")
        return 2

    if waveform.size == 0:
        print("No audio to stream (empty waveform).")
        return 1

    result = run(
        waveform,
        base_url=args.base_url,
        chunk_ms=args.chunk_ms,
        realtime_pacing=not args.no_realtime_pacing,
    )
    _print_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
