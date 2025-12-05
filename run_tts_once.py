#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-shot TTS worker for VideoAutomation.

This script is launched by watcher.py when there are pending .txt scripts
in PutScript. It builds the Gemini TTS engine + options and then delegates
all actual processing to watcher.process_all_pending_scripts().

It exits when there are no more pending scripts in this batch.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# --- Project bootstrap ---
VA_ROOT = Path(os.environ.get("VA_ROOT", str(Path.home() / "VideoAutomation")))
SYS     = VA_ROOT / "System"
SYSTTS  = SYS / "SysTTS"

# Ensure SysTTS is importable so we can import watcher
if str(SYSTTS) not in sys.path:
    sys.path.insert(0, str(SYSTTS))

import watcher  # type: ignore


def build_synth_opts() -> dict:
    """Build TTS synth options from environment, mirroring watcher.py."""
    return {
        "primary": os.environ.get("TTS_PRIMARY", "gemini-2.5-pro-tts"),
        "voice": os.environ.get("TTS_VOICE_NAME", "Enceladus"),
        "lang": os.environ.get("TTS_LANG", "en-US"),
        "rate": float(os.environ.get("TTS_RATE", "1.035")),
        "pitch": float(os.environ.get("TTS_PITCH", "0.025")),
        "sr": int(os.environ.get("TTS_SR", "22050")),
        # chunk & byte caps
        "byte_cap": int(os.environ.get("BYTE_CAP", "3000")),
        "target_words": int(os.environ.get("TARGET_WORDS", "450")),
        "max_words": int(os.environ.get("MAX_WORDS", "600")),
        # continuity / de-click (engine first, then our post-fix)
        "start_silence_ms": int(watcher.START_SILENCE_MS),
        "fade_ms": int(watcher.FADE_MS),
    }


def main() -> None:
    # Instantiate the same TTS engine watcher used previously
    engine = watcher.GeminiEngine(watcher.SYSTTS)
    synth_opts = build_synth_opts()

    # Delegate the actual processing to watcher.process_all_pending_scripts
    watcher.log("▶ TTS worker starting (run_tts_once.py)…")
    watcher.process_all_pending_scripts(engine, synth_opts)
    watcher.log("✔ TTS worker finished (run_tts_once.py).")


if __name__ == "__main__":
    main()