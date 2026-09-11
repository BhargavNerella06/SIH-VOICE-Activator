#!/usr/bin/env python
"""
scripts/command_smoke_test.py
================================
Minimal command-interpreter smoke test: text in, structured command out.

Two modes:
  --text "turn on the lights"   Interpret text directly (no ASR involved).
  --wav path/to/speech.wav      Run asr.transcriber first, then feed its
                                 transcript into the command interpreter --
                                 the "ASR text -> command interpreter ->
                                 structured command" chain.

Deliberately does NOT touch streaming, the API layer, KWS, or the
frontend -- this is a standalone demonstration of the command
interpreter (optionally chained after ASR), matching the "command
interpreter only" scope of the task this script was written for.

Usage
-----
    python scripts/command_smoke_test.py --text "turn on the lights"
    python scripts/command_smoke_test.py --wav path/to/speech.wav
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from command.interpreter import CommandInterpreter  # noqa: E402


def _transcribe_wav(path: str) -> str:
    import numpy as np
    import soundfile as sf

    from asr.transcriber import get_transcriber
    from audio_processing.utils import ensure_mono

    y, sr = sf.read(path)
    waveform = ensure_mono(np.asarray(y, dtype=np.float32))

    transcriber = get_transcriber()
    if not transcriber.is_ready:
        print(f"ASR model not available: {transcriber.load_error}")
        raise SystemExit(1)

    result = transcriber.transcribe(waveform, sample_rate=sr)
    if result.status != "OK":
        print(f"ASR failed: {result.status} - {result.message}")
        raise SystemExit(1)

    print(f"ASR transcript: {result.text!r}")
    return result.text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--text", type=str, default=None, help="Text to interpret directly.")
    parser.add_argument("--wav", type=str, default=None, help="WAV file to transcribe via ASR first.")
    args = parser.parse_args()

    if args.wav:
        text = _transcribe_wav(args.wav)
    elif args.text is not None:
        text = args.text
    else:
        parser.error("Provide either --text \"...\" or --wav <file>")
        return 2

    result = CommandInterpreter().interpret(text)

    print()
    print(json.dumps(asdict(result), indent=2))
    return 0 if result.status == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
