#!/usr/bin/env python
"""
scripts/remote_asr_client_demo.py
=====================================
Remote-ASR streaming client demo: reads a WAV file (NOT a microphone
recording -- e.g. one synthesized via Windows SAPI TTS, as used
elsewhere in this project's demos/tests) and streams it to the remote
ASR server (api/routers/asr_stream.py) exactly as an edge device would
stream a command-audio segment from AudioSessionManager.

This is a WAV-FILE test of the streaming protocol, not a microphone or
wireless-network test -- see docs/REMOTE_ASR_STREAMING.md for what is
and is not demonstrated by this script.

Usage
-----
    Terminal 1:
        uvicorn api.main:app --port 8000

    Terminal 2:
        python scripts/remote_asr_client_demo.py --wav path/to/command.wav
        python scripts/remote_asr_client_demo.py --wav path/to/command.wav --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from audio_processing.utils import ensure_mono, resample_if_needed  # noqa: E402
from voice_pipeline.remote_asr_client import (  # noqa: E402
    DEFAULT_HOST,
    DEFAULT_PORT,
    RemoteASRClient,
)

SAMPLE_RATE = 16000


def _load_wav(path: str) -> np.ndarray:
    y, sr = sf.read(path)
    waveform = ensure_mono(np.asarray(y, dtype=np.float32))
    if sr != SAMPLE_RATE:
        waveform = resample_if_needed(waveform, sr, SAMPLE_RATE)
    return waveform


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wav", type=str, required=True, help="WAV file to stream (not a live recording).")
    parser.add_argument("--host", type=str, default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--chunk-ms", type=float, default=30.0)
    args = parser.parse_args()

    waveform = _load_wav(args.wav)
    print(f"Loaded {len(waveform) / SAMPLE_RATE:.2f}s of audio from {args.wav} "
          f"(this is a WAV-file test, not a microphone recording)")

    client = RemoteASRClient(host=args.host, port=args.port)
    print(f"Connecting to remote ASR server at {client.url} ...")
    print("Streaming command audio...")

    result = client.send_command_audio(waveform, sample_rate=SAMPLE_RATE, chunk_ms=args.chunk_ms)

    if result.status == "CONNECTION_ERROR":
        print(f"ERROR: {result.message}")
        print("(Is the server running? Terminal 1: uvicorn api.main:app --port "
              f"{args.port})")
        return 1
    if result.status == "PROTOCOL_ERROR":
        print(f"ERROR: {result.message}")
        return 1

    print("Server received audio")

    if result.status == "EMPTY_AUDIO":
        print("Server reported EMPTY_AUDIO -- no audio was actually sent/received.")
        return 1
    if result.status == "ASR_ERROR":
        print(f"ASR error: {result.message}")
        return 1

    print(f'ASR transcription: "{result.transcript}"')
    cmd = result.command or {}
    print(f"Intent: {cmd.get('intent')}")
    if cmd.get("device"):
        print(f"Device: {cmd.get('device')}")
    print(f"Action: {cmd.get('action')}")
    if cmd.get("parameters"):
        print(f"Parameters: {cmd.get('parameters')}")
    print(f"Confidence: {cmd.get('confidence')}")

    ct = result.client_timings
    st = result.server_timings
    print()
    print("=== Latency (LOCALHOST DEVELOPMENT BENCHMARK -- see "
          "docs/REMOTE_ASR_STREAMING.md; NOT a Wi-Fi/wireless latency claim) ===")
    print(f"Client streaming duration   : {ct.streaming_duration_ms} ms")
    print(f"Client<->server round trip  : {ct.server_round_trip_ms} ms "
          "(includes server ASR + interpretation time, not network-only)")
    print(f"Server ASR latency          : {st.get('asr_latency_ms')} ms")
    print(f"Server audio duration       : {st.get('audio_duration_s')} s "
          f"({st.get('n_chunks')} chunks)")
    print(f"Total client wall-clock time: {ct.total_client_ms} ms")

    return 0 if result.status == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
