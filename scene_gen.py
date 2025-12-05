#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scene_gen.py — Generates image scenes based on time slices.

This version:
  - Uses Shared/slices.py to build time slices (5/7/9 with ±2s rules)
  - Splits script text and assigns text to each slice
  - Saves a canonical slices.json (and prompts.json style)
  - Generates EXACTLY one image per slice (Fal / Flux / any AI)
  - Completely decouples image logic from video/tts logic

NOTE: This file no longer decides durations. Slices decide everything.
"""

from __future__ import annotations
import os, json, re, sys
from pathlib import Path
from typing import List

# --------------------------------------------------------------
# Imports
# --------------------------------------------------------------

VA_ROOT = Path(os.environ.get("VA_ROOT", str(Path.home() / "VideoAutomation")))
SYSVIS  = VA_ROOT / "System" / "SysVisuals"
SHARED  = SYSVIS / "Shared"

if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))

import slices as slice_mod    # <-- NEW
from slices import SliceDef
from scene_engine import SceneEngine
from concept_extractor import extract_concept
from narrative_planner import build_narrative_plan, save_narrative_plan
from scene_planner import build_scene_plan, save_scene_plan

ENG_IMAGE = SYSVIS / "Engines" / "image"
if str(ENG_IMAGE) not in sys.path:
    sys.path.insert(0, str(ENG_IMAGE))

from ai_engine import AIImageEngine
_engine = AIImageEngine()

# Paths helper
try:
    import paths
except:
    paths = None

try:
    from prompt_style import apply_prompt_style_env
except Exception:
    apply_prompt_style_env = None


# --------------------------------------------------------------
# Script processing helpers
# --------------------------------------------------------------

_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')
_MIN_CHARS = 20


def script_sentences(text: str) -> List[str]:
    """Split script into clean sentences."""
    text = re.sub(r'\s+', ' ', text).strip()
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p.strip()]
    out = []
    buf = ""

    for s in parts:
        if len(buf) + len(s) < _MIN_CHARS:
            buf = (buf + " " + s).strip()
        else:
            if buf:
                out.append(buf)
                buf = ""
            out.append(s)

    if buf:
        out.append(buf)

    return out


def text_for_slice(sentences: List[str], s: SliceDef, total_slices: int) -> str:
    """
    Assign text to each slice sequentially.
    Later we can do smart relevance matching; for now it's linear.
    """
    if not sentences:
        return ""

    idx = s.index
    if idx < len(sentences):
        return sentences[idx]
    return sentences[-1]  # fallback


# --------------------------------------------------------------
# Main scene generation
# --------------------------------------------------------------

def run_scene(channel: str, day: str, job_id: str,
              wav_path: Path, script_path: Path, config: dict):

    # Load schedule
    schedule = os.environ.get("IMG_PROMPT_SCHEDULE", "0-240:5,240-540:7,540-inf:9")

    # Load script text
    script_txt = script_path.read_text(encoding="utf-8", errors="ignore")
    if callable(apply_prompt_style_env):
        apply_prompt_style_env(channel)

    # Initialize scene engine after channel style is applied
    engine = SceneEngine.from_env(default_root=SYSVIS)

    sentences = script_sentences(script_txt)

    # Determine audio duration
    from subprocess import run, PIPE
    pr = run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nokey=1:noprint_wrappers=1", str(wav_path)],
        stdout=PIPE, stderr=PIPE, text=True
    )
    try:
        audio_dur = float(pr.stdout.strip())
    except:
        audio_dur = 30.0  # fallback

    # Build slices
    slices = slice_mod.build_slices(audio_dur, schedule)

    # Prepare output dirs early (job_root is needed for plans)
    job_root = paths.build_ready_job_root(channel, job_id, day)
    scenes_dir = job_root / "Scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)

    # Assign text to slices
    for sl in slices:
      sl.text = text_for_slice(sentences, sl, len(slices))

    # Build narrative and scene plans for this job
    narrative_plan = build_narrative_plan(script_txt, slices)
    narrative_path = save_narrative_plan(narrative_plan, job_root, job_id)
    print(f"[SCENE] 📖 narrative plan saved → {narrative_path}")

    scene_plan = build_scene_plan(script_txt, slices)
    scene_plan_path = save_scene_plan(scene_plan, job_root, job_id)
    print(f"[SCENE] 🧩 scene plan saved → {scene_plan_path}")

    # Build scene prompts for each slice using the universal scene engine
    prompts = []
    for sl in slices:
        raw_text = sl.text or ""
        # Merge text-derived concepts with scene spec
        concept = extract_concept(raw_text)
        spec = scene_plan.get(sl.index, {}) if isinstance(scene_plan, dict) else {}
        # Do not override already-inferred fields in concept
        for k, v in spec.items():
            concept.setdefault(k, v)
        slice_summary = concept.get("slice_summary", raw_text)
        prompt_text, negative_text, meta = engine.build_prompt(
            slice_text=raw_text,
            slice_summary=slice_summary,
            **concept,
        )
        prompts.append({
            "index": sl.index,
            "t0": sl.t0,
            "t1": sl.t1,
            "text": raw_text,
            "prompt": prompt_text,
            "negative_prompt": negative_text,
            "engine_rule_id": meta.get("rule_id"),
            "engine_template": meta.get("template"),
            "scene_spec": spec,
        })
    # Quick lookup by index for generation/manifest
    prompts_by_index = {p["index"]: p for p in prompts}

    # Short summary log for slices
    print(f"[SCENE] 🎞 {len(slices)} slices (audio {audio_dur:.2f}s, schedule={schedule})")

    # Save canonical slices JSON
    pjson = paths.prompts_json_path(channel, day, job_id)
    slice_mod.save_slices(pjson, audio_dur, schedule, slices)

    # Generate one image per slice (compact progress logging)
    total = len(slices)
    for idx, s in enumerate(slices):
        out_img = scenes_dir / f"scene_{s.index+1:03d}.jpg"
        try:
            entry = prompts_by_index.get(s.index)
            prompt = entry["prompt"] if entry else s.text or ""
            print(f"[IMG] 🖼 [{idx+1}/{total}] {out_img.name}")
            _engine.generate(prompt, out_img, seed=None)
        except Exception as e:
            print(f"[IMG] ⚠ slice {s.index} failed: {e}")

    # Save engine prompts alongside slices for debugging and analysis
    prompts_out = job_root / f"{job_id}_final.prompts.json"
    prompts_out.write_text(
        json.dumps(prompts, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[SCENE] 🧠 prompts saved → {prompts_out}")

    # Save manifest.json
    manifest = {
        "job": job_id,
        "audio_duration": audio_dur,
        "schedule": schedule,
        "frames": [
            {
                "index": s.index,
                "t0": s.t0,
                "t1": s.t1,
                "file": f"Scenes/scene_{s.index+1:03d}.jpg",
                "text": s.text,
                "rule": prompts_by_index.get(s.index, {}).get("engine_rule_id"),
            }
            for s in slices
        ]
    }

    (job_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[SCENE] ✅ {len(slices)} scenes ready in {job_root}")


# --------------------------------------------------------------
# CLI
# --------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("Usage: scene_gen.py <Channel> <Day> <job_id> <wav_path> <script_path>")
        sys.exit(64)

    ch, day, jid, wav_p, scr_p = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
    run_scene(ch, day, jid, Path(wav_p), Path(scr_p), {})