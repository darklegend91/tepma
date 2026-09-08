#!/usr/bin/env python3
"""Download and warm every model the kiosk needs, so the first interview is not the slow one.

Run by `scripts/tepma.sh setup`; safe to run again at any time - everything it does is
cached, so a second run is a no-op that simply verifies the machine is ready.

    .venv/bin/python scripts/prepare_models.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from assistant_script import spoken_lines
from engines import STT_BACKEND, synthesize_wav, transcribe_audio


def step(label: str):
    print(f"  - {label} ... ", end="", flush=True)
    return time.perf_counter()


def done(started: float):
    print(f"{time.perf_counter() - started:.0f}s")


def main() -> int:
    print(f"Preparing models (STT backend: {STT_BACKEND})")

    # Whisper downloads on first use - a couple of gigabytes for the MLX large-v3 weights.
    # One second of silence is enough to force the download and the model load.
    t = step("speech recognition")
    try:
        transcribe_audio(np.zeros(16000, dtype=np.float32), fast=True)
        done(t)
    except Exception as exc:
        print(f"FAILED\n    {exc}")
        return 1

    # Kokoro downloads its weights and voices, once per language pipeline.
    for language in ("en", "hi"):
        t = step(f"speech synthesis ({language})")
        try:
            synthesize_wav("Ready.", language)
            done(t)
        except Exception as exc:
            print(f"FAILED\n    {exc}")
            return 1

    # Every scripted line the kiosk speaks, cached to disk so sessions never synthesize
    # the greeting again. The server warms this at startup too; doing it here means the
    # very first visitor after an install is greeted instantly.
    t = step("caching scripted speech")
    failures = 0
    for text, language in spoken_lines():
        try:
            synthesize_wav(text, language)
        except Exception:
            failures += 1
    done(t)
    if failures:
        print(f"    ({failures} line(s) could not be synthesized)")

    print("\nModels ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
