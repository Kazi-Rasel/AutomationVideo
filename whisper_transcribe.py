# File: /Users/kazirasel/VideoAutomation/System/SysTTS/whisper_transcribe.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run WhisperX on a WAV file to produce real word-level timestamps.

Usage (from whisper-env):
    python3 whisper_transcribe.py /path/to/audio.wav /path/to/output.json

Output JSON schema:
{
  "words": [
    { "text": "This", "t0": 0.12, "t1": 0.45 },
    { "text": "is",   "t0": 0.45, "t1": 0.62 },
    ...
  ]
}
"""

import sys
import json
from pathlib import Path

import os
import datetime

import warnings
import logging

# Silence noisy third-party warnings/logs; keep our own [WHISPER] prints clean.
warnings.filterwarnings("ignore", category=UserWarning)

for name in [
    "whisperx.asr",
    "whisperx.vads.pyannote",
    "pyannote.audio",
    "speechbrain",
    "lightning",
    "pytorch_lightning",
]:
    logging.getLogger(name).setLevel(logging.WARNING)

import torch
import whisperx

# --- Central logging setup ---
VA_ROOT = Path(os.environ.get("VA_ROOT", str(Path.home() / "VideoAutomation")))
SYS     = VA_ROOT / "System"
LOG_DIR = SYS / "Logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "whisper_transcribe.log"


def wlog(msg: str) -> None:
    """Log Whisper events to terminal and central log file."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[WHISPER {timestamp}] {msg}"
    # print to terminal
    print(line, flush=True)
    # append to central log file
    try:
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def main() -> None:
    if len(sys.argv) != 3:
        wlog("Usage: whisper_transcribe.py <input.wav> <output.json>")
        sys.exit(1)

    wav_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])

    if not wav_path.exists():
        wlog(f"Input audio not found: {wav_path}")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    wlog(f"Using device: {device}")

    wlog(f"Loading audio: {wav_path}")
    audio = whisperx.load_audio(str(wav_path))

    # 1) ASR model (no word_timestamps arg here)
    wlog("Loading ASR model (small, int8)…")
    model = whisperx.load_model("small", device=device, compute_type="int8")

    wlog("Transcribing…")
    result = model.transcribe(audio, batch_size=16)

    lang = result.get("language", "en")
    wlog(f"Detected language: {lang}")

    # 2) Alignment model for word-level timings
    wlog("Loading alignment model…")
    align_model, metadata = whisperx.load_align_model(
        language_code=lang,
        device=device,
    )

    wlog("Aligning to get word-level timestamps…")
    aligned = whisperx.align(
        result["segments"],
        align_model,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    # 3) Extract words into simple schema
    words: list[dict] = []
    for seg in aligned.get("segments", []):
        for w in seg.get("words", []):
            text = str(w.get("word", "")).strip()
            start = w.get("start", None)
            end   = w.get("end", None)
            if not text or start is None or end is None:
                continue
            words.append(
                {
                    "text": text,
                    "t0": float(start),
                    "t1": float(end),
                }
            )

    if not words:
        wlog("No word-level timestamps found; nothing to write.")
        sys.exit(0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"words": words}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    wlog(f"Saved word timings → {out_path.name}")


if __name__ == "__main__":
    main()