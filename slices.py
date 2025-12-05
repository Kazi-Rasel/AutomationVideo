#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
import json


@dataclass
class SliceDef:
    """
    A single time slice over the audio.

    Fields:
      index   : 0-based index of the slice
      t0, t1  : start/end time in seconds (global audio timeline)
      target  : preferred length for this slice (e.g. 5 / 6 / 7, from schedule)
      text    : caption/prompt text assigned later
    """
    index: int
    t0: float
    t1: float
    target: float
    text: str = ""


# ---------------------------------------------------------------------------
# Schedule parsing
# ---------------------------------------------------------------------------

def parse_schedule(spec: str) -> List[Tuple[float, float, float]]:
    """
    Parse a schedule string like '0-240:5,240-540:6,540-inf:7' into a list of
    (start, end, target_length) tuples in seconds.

    Example:
      '0-240:5,240-540:6,540-inf:7'
        -> [(0.0,240.0,5.0), (240.0,540.0,6.0), (540.0,∞,7.0)]
    'inf' as the end means +∞.
    """
    parts: List[Tuple[float, float, float]] = []
    if not spec:
        return parts

    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        rng, val = chunk.split(":")
        start_s, end_s = [x.strip() for x in rng.split("-")]
        start = float(start_s)
        end = float("inf") if end_s == "inf" else float(end_s)
        target = float(val)
        parts.append((start, end, target))

    parts.sort(key=lambda r: r[0])
    return parts


def target_for_t(t: float, schedule: List[Tuple[float, float, float]]) -> float:
    """
    Look up the preferred slice length at time t from the schedule.
    If t is beyond the last schedule bucket, reuse the last bucket's target.
    """
    for s0, s1, targ in schedule:
        if s0 <= t < s1:
            return targ
    return schedule[-1][2] if schedule else 5.0  # default to 5s if no schedule


# ---------------------------------------------------------------------------
# Slice building
# ---------------------------------------------------------------------------

def build_slices(
    audio_dur: float,
    schedule_spec: str,
    min_len: float = 3.0,
    plus2: float = 2.0,
) -> List[SliceDef]:
    """
    Build time slices that cover [0, audio_dur] using a schedule and your rules.

    Rules:
      - The schedule string (e.g. "0-240:5,240-540:6,540-inf:7") defines a
        preferred target length (e.g. 5 / 6 / 7) for any time t.
      - Each slice starts where the previous one ended.
      - For a given target length L, the allowed length band is:
            [max(min_len, L - plus2), L + plus2]
        With min_len=3, plus2=2 this yields, for example:
            L=5 -> [3,7]
            L=6 -> [4,8]
            L=7 -> [5,9]
      - We never create a slice shorter than min_len. If the final leftover
        would be shorter than its lower bound, we extend the previous slice
        to the end of the audio.
    """
    if audio_dur <= 0:
        return []

    sched = parse_schedule(schedule_spec)
    if not sched:
        # Fallback: single slice covering the whole audio
        return [SliceDef(index=0, t0=0.0, t1=audio_dur, target=audio_dur)]

    slices: List[SliceDef] = []
    t = 0.0
    idx = 0

    while t < audio_dur:
        target = target_for_t(t, sched)

        # Allowed band for this target
        lo = max(min_len, target - plus2)
        hi = target + plus2

        remaining = audio_dur - t
        if remaining <= lo:
            # Remaining audio is too short to form a valid new slice:
            # extend the last slice instead of creating a tiny one.
            if slices:
                last = slices[-1]
                last.t1 = audio_dur
            else:
                # Only one slice covering everything
                slices.append(SliceDef(index=0, t0=0.0, t1=audio_dur, target=target))
            break

        # If what's left fits entirely into the band, make a final slice and stop.
        if lo <= remaining <= hi:
            slices.append(SliceDef(index=idx, t0=t, t1=audio_dur, target=target_for_t(t, sched)))
            break

        # Otherwise, create a slice clamped within [lo, hi].
        length = max(lo, min(hi, target))
        t1 = t + length
        if t1 > audio_dur:
            t1 = audio_dur

        slices.append(SliceDef(index=idx, t0=t, t1=t1, target=target))
        idx += 1
        t = t1

    return slices


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def save_slices(path: Path, audio_dur: float, schedule_spec: str, slices: List[SliceDef]) -> None:
    """Write slices to a JSON file for reuse by other components."""
    data = {
        "audio_duration": audio_dur,
        "schedule": schedule_spec,
        "slices": [
            {
                "index": s.index,
                "t0": round(s.t0, 3),
                "t1": round(s.t1, 3),
                "target": s.target,
                "text": s.text,
            }
            for s in slices
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_slices(path: Path) -> Optional[List[SliceDef]]:
    """Load slices from a JSON file created by save_slices()."""
    if not path.exists():
        return None
    try:
        raw = json.load(path.open("r", encoding="utf-8"))
    except Exception:
        return None

    out: List[SliceDef] = []
    for item in raw.get("slices", []):
        try:
            out.append(
                SliceDef(
                    index=int(item.get("index", len(out))),
                    t0=float(item.get("t0", 0.0)),
                    t1=float(item.get("t1", 0.0)),
                    target=float(item.get("target", 5.0)),
                    text=str(item.get("text", "")),
                )
            )
        except Exception:
            continue
    return out