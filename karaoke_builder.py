#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import re
import json
import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# --------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------
VA_ROOT = Path(os.environ.get("VA_ROOT", str(Path.home() / "VideoAutomation")))
SYSTTS  = VA_ROOT / "System" / "SysTTS"
SUBTITLES_DIR = SYSTTS / "Subtitles"

SYS     = VA_ROOT / "System"
LOG_DIR = SYS / "Logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "karaoke_builder.log"

# --------------------------------------------------------------------
# ------------- GLOBAL SETTINGS (edit here) -------------
FONT_NAME   = "Montserrat"
FONT_SIZE   = 83                # main text size

# Colours in BGR order (no alpha here – we inject it in the style)
BASE_RGB    = "F8F8F8"          # off-white base (white as before)
HI_RGB      = "30C8FF"          # amber/yellow highlight as before
OUTLINE_RGB = "000000"          # black outline
BACK_RGB    = "202020"          # soft shadow/glow

MARGIN_V    = 130               # bottom margin

OUTLINE     = 2.5               # outline thickness (restored)
SHADOW      = 3.0               # shadow/glow size (restored)
BOLD        = -1                # -1 = bold, 0 = normal

GLOBAL_OFFSET       = 0.0       # shift all subs (seconds)
PAUSE_THRESHOLD     = 0.40      # new group if gap > this
MAX_WORDS_PER_GROUP = 4
MAX_CHARS_PER_GROUP = 22
MIN_SEG_DURATION    = 0.05      # minimum segment length

# Global transparency: 0x00 = fully solid, 0xFF = fully invisible
# This is the alpha that gave you the "it's happening now" transparency.
GLOBAL_ALPHA = 0x5B             # ~37% visible A0, ≈44% visible 90, ≈50% visible 80, ≈60% visible 67, ≈65% visible 5B, ≈70% visible 4C 

def _style_color(rgb: str) -> str:
    """
    Build an ASS style colour with global alpha.
    ASS format is &HAABBGGRR; rgb is BGR, e.g. "F8F8F8".
    """
    return f"&H{GLOBAL_ALPHA:02X}{rgb}"

# Style colours (with alpha)
BASE_COLOR    = _style_color(BASE_RGB)
HI_COLOR      = _style_color(HI_RGB)
OUTLINE_COLOR = _style_color(OUTLINE_RGB)
BACK_COLOR    = _style_color(BACK_RGB)

# BGR-only versions for override tags (so they don't override alpha)
BASE_BGR = BASE_RGB
HI_BGR   = HI_RGB

# ------------- Logging & model -------------
def _log(msg: str) -> None:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[KARAOKE {timestamp}] {msg}"
    # print to terminal
    print(line, flush=True)
    # append to central log file
    try:
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

@dataclass
class Word:
    text: str
    t0: float
    t1: float

# ------------- Time formatting -------------
def _ass_time(t: float) -> str:
    t = max(0.0, float(t))
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    cs = int((t % 1.0) * 100 + 0.5)
    return f"{h:01d}:{m:02d}:{s:02d},{cs:02d}"

# ------------- ASS Header -------------
def _ass_header() -> str:
    return f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
ScaledBorderAndShadow: yes
Collisions: Normal
WrapStyle: 2

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: KARAOKE,{FONT_NAME},{FONT_SIZE},{BASE_COLOR},{HI_COLOR},{OUTLINE_COLOR},{BACK_COLOR},{BOLD},0,0,0,100,100,0,0,1,{OUTLINE},{SHADOW},2,25,25,{MARGIN_V},1

[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
"""

# Pre-built colour tags (BGR only, alpha comes from the style)
BASE_TAG = f"{{\\c&H{BASE_BGR}&}}"
HI_TAG   = f"{{\\c&H{HI_BGR}&}}"

# --------------------------------------------------------------------
# Load Whisper words
# --------------------------------------------------------------------
def _load_whisper_words(day: str, job_id: str) -> List[Word]:
    path = SUBTITLES_DIR / day / f"{job_id}_whisper.json"
    if not path.exists():
        _log(f"⚠ whisper json not found: {path}")
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        _log(f"⚠ failed to read {path}: {e}")
        return []
    words: List[Word] = []
    for w in data.get("words", []):
        try:
            txt = str(w.get("text", "")).strip()
            t0  = float(w.get("t0", 0.0))
            t1  = float(w.get("t1", 0.0))
        except Exception:
            continue
        if txt and t1 > t0:
            words.append(Word(txt, t0, t1))
    return words

# --------------------------------------------------------------------
# Token cleaning & grouping
# --------------------------------------------------------------------
def _clean_display_token(raw: str) -> str:
    # keep alnum + common currency/symbol chars, strip punctuation
    parts = re.findall(r"[A-Za-z0-9$€£¥%&'/-]+", raw)
    if not parts:
        return ""
    return "".join(parts).upper()

def _is_strong_punct(raw: str) -> bool:
    return any(ch in raw for ch in ".!?")

def _should_break(prev: Word, cur: Word, cur_len: int, cur_count: int) -> bool:
    if cur_count == 0:
        return False
    # long pause
    if cur.t0 - prev.t1 > PAUSE_THRESHOLD:
        return True
    # sentence boundary
    if _is_strong_punct(prev.text):
        return True
    # too many words
    if cur_count >= MAX_WORDS_PER_GROUP:
        return True
    # too wide text
    next_len = cur_len + 1 + len(_clean_display_token(cur.text))
    return next_len > MAX_CHARS_PER_GROUP

def _group_words_dynamic(words: List[Word]) -> List[List[Word]]:
    groups: List[List[Word]] = []
    cur: List[Word] = []
    cur_len = 0
    for w in words:
        disp = _clean_display_token(w.text)
        if not disp:
            # punctuation-only token can terminate group
            if cur and _is_strong_punct(w.text):
                groups.append(cur)
                cur, cur_len = [], 0
            continue

        if cur and _should_break(cur[-1], w, cur_len, len(cur)):
            groups.append(cur)
            cur, cur_len = [], 0

        cur.append(w)
        cur_len += (0 if cur_len == 0 else 1) + len(disp)

    if cur:
        groups.append(cur)
    return groups

# --------------------------------------------------------------------
# Build coloured line for one highlight index
# --------------------------------------------------------------------
def _build_colored_line(tokens: List[str], hi_idx: int) -> str:
    """
    Full phrase, base-coloured, with only tokens[hi_idx] in highlight.
    If hi_idx < 0, everything is base colour (used before first highlight).
    """
    parts: List[str] = [BASE_TAG]
    for i, tok in enumerate(tokens):
        if i > 0:
            parts.append(" ")
        if i == hi_idx:
            parts.append(f"{HI_TAG}{tok}{BASE_TAG}")
        else:
            parts.append(tok)
    return "".join(parts)

# --------------------------------------------------------------------
# Build events for one group (single-layer per time slice)
# --------------------------------------------------------------------
def _build_group_events(grp: List[Word]) -> List[str]:
    tokens = [_clean_display_token(w.text) for w in grp]
    tokens = [t for t in tokens if t]
    if not tokens:
        return []

    group_start = grp[0].t0 + GLOBAL_OFFSET
    group_end   = grp[-1].t1 + GLOBAL_OFFSET
    if group_end <= group_start:
        return []

    events: List[str] = []

    # For each word, create a slice where that word is highlighted.
    for idx, w in enumerate(grp):
        # Determine segment start/end
        start = w.t0 + GLOBAL_OFFSET
        if idx + 1 < len(grp):
            next_t0 = grp[idx + 1].t0 + GLOBAL_OFFSET
            end = next_t0
        else:
            end = group_end

        # Clamp and enforce minimum duration
        start = max(group_start, start)
        end   = max(start + MIN_SEG_DURATION, min(end, group_end))

        text = _build_colored_line(tokens, hi_idx=idx)
        events.append(
            f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},KARAOKE,,0,0,{MARGIN_V},,{text}"
        )

    return events

# --------------------------------------------------------------------
# Main builder
# --------------------------------------------------------------------
def build_karaoke_ass(channel: str, day: str, job_id: str) -> Optional[Path]:
    _log(f"▶ karaoke {channel}/{day}/{job_id}")
    words = _load_whisper_words(day, job_id)
    if not words:
        _log("⚠ no whisper words; skipping karaoke")
        return None

    groups = _group_words_dynamic(words)
    _log(f"🎤 {len(groups)} lines from {len(words)} words")

    job_root = (VA_ROOT / "Channels" / channel / "Build" / "Ready" / day / job_id)
    job_root.mkdir(parents=True, exist_ok=True)
    ass_path = job_root / f"{job_id}.ass"

    out_lines: List[str] = [_ass_header()]

    for grp in groups:
        events = _build_group_events(grp)
        out_lines.extend(events)

    ass_path.write_text("\n".join(out_lines), encoding="utf-8")
    _log(f"✅ ASS → {ass_path}")
    return ass_path

# --------------------------------------------------------------------
# CLI entry
# --------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        _log("Usage: karaoke_builder.py <channel> <YYYY-MM-DD> <job_id>")
        sys.exit(1)
    ch, day, jid = sys.argv[1], sys.argv[2], sys.argv[3]
    build_karaoke_ass(ch, day, jid)