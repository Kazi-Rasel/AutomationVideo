# File: /Users/kazirasel/VideoAutomation/System/SysVisuals/Orchestrators/scene_orchestrator.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scene_orchestrator.py — orchestrate scene (image) generation based on time slices.

This version:
  - Uses Shared/slices.py to compute time slices (5/7/9s with ±2s and min 3s)
  - Assigns script text to each slice in order
  - Saves a canonical prompts/slices JSON that video_orchestrator.py can use
  - Generates exactly one image per slice via the AI image engine
  - Writes a manifest.json with frames, t0/t1, and text for compatibility
"""

from __future__ import annotations

import os
import sys
import json
import re
import subprocess
import shutil
import datetime
from types import SimpleNamespace
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from typing import Any

VA_ROOT = Path(os.environ.get("VA_ROOT", str(Path.home() / "VideoAutomation")))
SYS     = VA_ROOT / "System"
SYSVIS  = SYS / "SysVisuals"
SHARED  = SYSVIS / "Shared"
ENG_IMG = SYSVIS / "Engines" / "image"
TOOLS   = SYSVIS / "Tools"
LLM     = SYSVIS / "LLM"

# --- Central log directory ---
LOG_DIR = SYS / "Logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "scene_orchestrator.log"

# LLM sentinel: if present, we should not start/continue LLM-driven scene jobs
LLM_SENTINEL = LLM / "LLM_DISABLED"

def _llm_disabled() -> bool:
    """Return True if the LLM_DISABLED sentinel exists.

    When present, the scene orchestrator should not start or continue
    LLM-based scene generation for this job. This mirrors the behavior in
    watcher.py and auto_on_job.py and prevents partial/corrupt outputs
    when the LLM layer is unavailable.
    """
    try:
        if LLM_SENTINEL.exists():
            try:
                reason = LLM_SENTINEL.read_text(encoding="utf-8", errors="ignore").strip()
            except Exception:
                reason = ""
            if not reason:
                reason = "LLM disabled by sentinel file."
            _log(f"🚫 LLM_DISABLED sentinel present at {LLM_SENTINEL}")
            _log(f"🚫 Reason: {reason}")
            return True
    except Exception as e:
        # Logging uses _log, which is defined below; we avoid calling it here
        # and let callers handle failure conservatively.
        print(f"[SCENE] ⚠ Failed to inspect LLM sentinel: {e}", flush=True)
    return False

# Ensure search paths
for p in (SHARED, ENG_IMG, TOOLS, LLM):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Shared slices (time logic)
import slices as slice_mod
from slices import SliceDef

# AI image engine
from ai_engine import AIImageEngine
_engine = AIImageEngine()

# VisualLibrary and image pipeline helpers
import visual_library
from clip_utils import CategoryIndex
from image_pipeline import process_slice

# Integrate prompt building
from prompt_style import (
    analyse_job,
    classify_role,
    choose_topic_for_slice,
    assign_concept_id,
    build_scene_prompt_semantic,
    analyse_job_llm,
    build_scene_prompt_llm,
    load_channel_policy,
)

# LLM error type
from LLM.gpt_client import LLMQuotaError

# Paths helper
try:
    import paths
except Exception:
    paths = None

# --------------------------------------------------------------
# Helpers
# --------------------------------------------------------------

def _infer_chapters(job_profile: Any, total_slices: int) -> List[str]:
    """Generic chapter assignment.
    For now, simple proportional slicing:
        - first 15% = HOOK
        - next 35% = CONTEXT
        - next 35% = DEVELOPMENT
        - final 15% = RESOLUTION
    Channels may override via channel_policy.refine_llm_scene_context.
    Returns: list of chapter labels aligned to slice indices.
    """
    chapters = []
    n = total_slices
    for i in range(n):
        frac = i / max(1, n - 1)
        if frac < 0.15:
            chapters.append("HOOK")
        elif frac < 0.50:
            chapters.append("CONTEXT")
        elif frac < 0.85:
            chapters.append("DEVELOPMENT")
        else:
            chapters.append("RESOLUTION")
    return chapters

_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')
_MIN_CHARS  = 20

SCENE_MIN   = float(os.environ.get("IMG_SCENE_MIN", "4.0"))
SCENE_TARGET = float(os.environ.get("IMG_SCENE_TARGET", "5.5"))
SCENE_MAX   = float(os.environ.get("IMG_SCENE_MAX", "9.0"))
SKIP_IF_EXISTS = os.environ.get("IMG_SKIP_IF_EXISTS", "1") == "1"
def _is_hero_portrait_concept(concept_id: str) -> bool:
    """Return True if this concept_id looks like a hero/person portrait anchor."""
    if not concept_id:
        return False
    parts = concept_id.split("|")
    if len(parts) < 2:
        return False
    entity, motif = parts[0], parts[1]
    if entity.startswith("person:") and motif in ("portrait", "hook_tableau"):
        return True
    return False

def _log(msg: str) -> None:
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[SCENE {timestamp}] {msg}"
    # print to terminal
    print(line, flush=True)
    # append to central log file
    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(line + "\n")
    except Exception:
        pass


def _script_sentences(text: str) -> List[str]:
    """
    Split the script into reasonably sized sentences/chunks.
    Later we can replace this with a smarter semantic segmenter.
    """
    text = re.sub(r"\s+", " ", text).strip()
    raw = [p.strip() for p in _SENT_SPLIT.split(text) if p.strip()]
    out: List[str] = []
    buf = ""

    for s in raw:
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


def _audio_duration_sec(wav: Path) -> float:
    """
    Use ffprobe to get the duration of the final merged WAV in seconds.
    """
    try:
        pr = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nokey=1:noprint_wrappers=1", str(wav)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False
        )
        d = float(pr.stdout.strip())
        return max(0.5, d)
    except Exception:
        return 30.0  # safe fallback


def _text_for_slice(sentences: List[str], s: SliceDef) -> str:
    """
    Assign prompt text to a slice. For now, we map slices to sentences linearly:
    slice 0 -> sentence 0, slice 1 -> sentence 1, etc.,
    and if we run out, we reuse the last sentence.
    """
    if not sentences:
        return ""
    if s.index < len(sentences):
        return sentences[s.index]
    return sentences[-1]


# ---------------- Category helpers for segment routing ----------------
def _load_categories(cat_file: Path) -> List[Dict[str, Any]]:
    """Load categories config from JSON file for text-based routing."""
    try:
        raw = cat_file.read_text(encoding="utf-8")
        data = json.loads(raw)
        if isinstance(data, list):
            return [c for c in data if isinstance(c, dict) and "id" in c and "prompt" in c]
        return []
    except Exception:
        return []

def _best_category_for_text(text: str, categories: List[Dict[str, Any]]) -> str:
    """Choose the best category id for a given text using simple keyword overlap."""
    text_l = (text or "").lower()
    best_id = ""
    best_score = 0
    for cat in categories:
        prompt = str(cat.get("prompt", ""))
        # Split prompt into simple tokens
        tokens = [t for t in re.split(r"[^a-z0-9]+", prompt.lower()) if t]
        if not tokens:
            continue
        score = 0
        for t in tokens:
            if t and t in text_l:
                score += 1
        if score > best_score:
            best_score = score
            best_id = str(cat.get("id", ""))
    return best_id


@dataclass
class VisualSegment:
    index: int
    concept_id: str
    topic: str
    role: str
    t0: float
    t1: float
    slice_indices: List[int]
    caption_text: str
    prompt: str = ""
    negative: str = ""
    file: str = ""


# --------------------------------------------------------------
# Main entry: run_scene
# --------------------------------------------------------------

def run_scene(channel: str,
              day: str,
              job_id: str,
              wav_path: Path,
              script_path: Path,
              config: dict) -> None:
    """
    Main entry called by auto_on_job.py.

    Parameters:
      channel     : channel name (folder under Channels/)
      day         : YYYY-MM-DD
      job_id      : unique job id (e.g. something_CC_final)
      wav_path    : path to final merged TTS WAV in Input/Audio/<day>/
      script_path : path to script .txt in Input/Script/<day>/
      config      : future hook for per-channel overrides (unused for now)
    """
    # Hard guard: if the LLM sentinel is present, do not start this job.
    if _llm_disabled():
        _log(f"🛑 Aborting scene job for {channel}/{day}/{job_id} due to LLM_DISABLED sentinel.")
        return

    if not paths:
        _log("paths module not available; aborting.")
        return

    if not wav_path.exists():
        _log(f"WAV not found: {wav_path}")
        return

    if not script_path.exists():
        _log(f"Script not found: {script_path}")
        return

    # 1) Load script text and convert to sentences/chunks
    script_txt = script_path.read_bytes().decode("utf-8", errors="ignore")
    sentences = _script_sentences(script_txt)
    if not sentences:
        _log("No sentences found in script; aborting.")
        return

    # 2) Determine audio duration
    audio_dur = _audio_duration_sec(wav_path)

    # 3) Build time slices using schedule + 5/7/9 ±2s + min 3s rule
    schedule_spec = os.environ.get("IMG_PROMPT_SCHEDULE", "0-240:5,240-540:7,540-inf:9")
    slice_defs: List[SliceDef] = slice_mod.build_slices(audio_dur, schedule_spec)
    if not slice_defs:
        _log("No slices built; aborting.")
        return

    _log(f"🎞 {len(slice_defs)} slices (audio {audio_dur:.2f}s, schedule={schedule_spec})")

    # 4) Assign text to each slice
    for s in slice_defs:
        s.text = _text_for_slice(sentences, s)

    try:
        job_profile = analyse_job_llm(script_txt, channel, job_id)
    except LLMQuotaError as e:
        _log(f"❌ LLM quota error while analyzing job {job_id}: {e}")
        return
    except Exception as e:
        _log(f"⚠ LLM analyse_job_llm failed, falling back to heuristic analyse_job: {e}")
        job_profile = analyse_job(script_txt, channel, job_id)

    policy = load_channel_policy(channel)
    # Prepare chapters now based on slice count
    chapter_labels = _infer_chapters(job_profile, len(slice_defs))

    prev_concept_id: Optional[str] = None
    for s in slice_defs:
        s.caption = s.text  # keep original for subtitles/debug
        role = classify_role(s.text or "", job_profile)
        topic = choose_topic_for_slice(s.text or "", job_profile, channel)
        # Assign chapter label
        s.chapter = chapter_labels[s.index] if s.index < len(chapter_labels) else "CONTEXT"
        s.role = role
        s.topic = topic
        s.concept_id = assign_concept_id(s.text or "", topic, job_profile, prev_concept_id)
        prev_concept_id = s.concept_id

    # 5) Save canonical prompts/slices JSON for later (video + SRT)
    prompts_path = paths.prompts_json_path(channel, day, job_id)  # type: ignore[attr-defined]
    slice_mod.save_slices(prompts_path, audio_dur, schedule_spec, slice_defs)
    _log(f"Saved slices/prompts → {prompts_path}")

    # 6) Build semantic visual segments from slices (concept blocks + duration constraints)
    segments: List[VisualSegment] = []
    if slice_defs:
        # group into concept blocks
        blocks: List[List[SliceDef]] = []
        current_block: List[SliceDef] = [slice_defs[0]]
        for s in slice_defs[1:]:
            if getattr(s, "concept_id", None) == getattr(current_block[-1], "concept_id", None):
                current_block.append(s)
            else:
                blocks.append(current_block)
                current_block = [s]
        blocks.append(current_block)

        seg_index = 0
        for block in blocks:
            block_t0 = block[0].t0
            block_t1 = block[-1].t1
            # Greedy split by SCENE_MAX while trying to stay near SCENE_TARGET
            seg_slices: List[SliceDef] = []
            seg_start_t = block[0].t0
            for s in block:
                seg_slices.append(s)
                tentative_t1 = s.t1
                dur = tentative_t1 - seg_start_t
                if dur >= SCENE_TARGET or dur >= SCENE_MAX:
                    # close this segment
                    seg = VisualSegment(
                        index=seg_index,
                        concept_id=getattr(block[0], "concept_id", ""),
                        topic=getattr(block[0], "topic", ""),
                        role=getattr(block[0], "role", "MACRO_STATEMENT"),
                        t0=seg_start_t,
                        t1=tentative_t1,
                        slice_indices=[x.index for x in seg_slices],
                        caption_text=" ".join((x.caption if hasattr(x, "caption") else x.text) for x in seg_slices),
                    )
                    segments.append(seg)
                    seg_index += 1
                    seg_slices = []
                    seg_start_t = s.t1
            # flush remainder if any
            if seg_slices:
                seg = VisualSegment(
                    index=seg_index,
                    concept_id=getattr(block[0], "concept_id", ""),
                    topic=getattr(block[0], "topic", ""),
                    role=getattr(block[0], "role", "MACRO_STATEMENT"),
                    t0=seg_start_t,
                    t1=seg_slices[-1].t1,
                    slice_indices=[x.index for x in seg_slices],
                    caption_text=" ".join((x.caption if hasattr(x, "caption") else x.text) for x in seg_slices),
                )
                segments.append(seg)
                seg_index += 1

    if not segments:
        _log("No segments built; aborting.")
        return

    _log(f"🎨 {len(segments)} visual segments built from {len(slice_defs)} slices")

    # 7) Prepare job directory and Scenes folder
    job_root   = paths.build_ready_job_root(channel, job_id, day)  # type: ignore[attr-defined]
    scenes_dir = job_root / "Scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)

    # Initialise VisualLibrary, CategoryIndex, and used_phashes for image pipeline
    vlib_root = SYSVIS / "VisualLibrary"
    vlib_root.mkdir(parents=True, exist_ok=True)
    vlib = visual_library.vlib_init(vlib_root)
    # expose root path so image_pipeline can derive images/ location
    vlib["root"] = vlib_root

    cat_index = None
    cat_path = vlib_root / "categories.json"
    if cat_path.is_file():
        try:
            cat_index = CategoryIndex.from_config(cat_path)
            _log(f"Loaded CategoryIndex from {cat_path}")
        except Exception as e:
            _log(f"⚠ Failed to load categories.json: {e}")
            cat_index = None

    categories_cfg: List[Dict[str, Any]] = []
    if cat_path.is_file():
        cats = _load_categories(cat_path)
        if cats:
            _log(f"Loaded {len(cats)} text categories from {cat_path}")
            categories_cfg = cats

    used_phashes: List[str] = []

    # Optional: refine segment topics using categories.json text routing
    if 'categories_cfg' in locals() and categories_cfg:
        for seg in segments:
            cat_id = _best_category_for_text(seg.caption_text, categories_cfg)
            if cat_id:
                seg.topic = cat_id

    # 8) Generate or reuse one image per visual segment via VisualLibrary-aware pipeline
    sticky_assets: Dict[str, str] = {}
    results: List[Any] = []
    for seg in segments:
        img_path = scenes_dir / f"scene_{seg.index+1:03d}.jpg"
        seg.file = f"Scenes/scene_{seg.index+1:03d}.jpg"
        _log(f"[segment {seg.index+1}/{len(segments)}] {seg.t0:.2f}-{seg.t1:.2f}s concept={seg.concept_id}")

        # If an image already exists for this segment and skip-if-exists is enabled, keep it
        if SKIP_IF_EXISTS and img_path.exists():
            _log(f"↷ Keeping existing image for segment {seg.index+1}: {img_path.name}")
            results.append(SimpleNamespace(reused=True, chosen_phash=None, attempts=0, bright_drops=0))
            continue

        concept_key = seg.concept_id or ""
        # Per-job sticky hero reuse: if we already generated a hero/portrait for this concept, copy it
        if concept_key and concept_key in sticky_assets and _is_hero_portrait_concept(concept_key):
            src_path = Path(sticky_assets[concept_key])
            if src_path.is_file():
                shutil.copy2(src_path, img_path)
                _log(f"🧬 Sticky hero reuse for segment {seg.index+1} concept={concept_key}")
                results.append(SimpleNamespace(reused=True, chosen_phash=None, attempts=0, bright_drops=0))
                continue

        # Use the first slice in the segment as representative for engine/pipeline
        rep_slice = next((s for s in slice_defs if s.index == seg.slice_indices[0]), None)
        if rep_slice is None:
            _log(f"⚠ No representative slice found for segment {seg.index}; skipping")
            continue

        # If LLM has been disabled mid-run, stop this job immediately
        if _llm_disabled():
            _log(f"🛑 Stopping scene generation mid-job for {channel}/{day}/{job_id} due to LLM_DISABLED sentinel.")
            break

        # previous-scene continuity
        prev_scene_text = ""
        if seg.index > 0:
            prev_scene_text = segments[seg.index - 1].prompt or ""

        approx_role = getattr(rep_slice, "role", "") or ""
        approx_topic = seg.topic or (getattr(rep_slice, "topic", "") or "")
        # Build extended context: chapter + previous scene
        slice_extended_text = f"[CHAPTER: {getattr(rep_slice, 'chapter', 'CONTEXT')}] " \
                              f"[PREV_SCENE: {prev_scene_text}] " \
                              f"{rep_slice.text or ''}"
        # Build LLM-driven scene prompt, with semantic fallback
        try:
            role, positive, negative = build_scene_prompt_llm(
                channel=channel,
                slice_text=slice_extended_text,
                job_profile=job_profile,
                approx_role=approx_role,
                approx_topic=approx_topic,
            )
        except LLMQuotaError as e:
            _log(f"❌ LLM quota error during segment {seg.index}: {e}; aborting job {job_id}")
            return
        except Exception as e:
            _log(f"⚠ LLM build_scene_prompt_llm failed for segment {seg.index}: {e}; falling back to semantic builder")
            positive, negative = build_scene_prompt_semantic(
                channel,
                approx_role or "MACRO_STATEMENT",
                approx_topic or "",
                rep_slice.text or "",
                job_profile,
            )
            role = approx_role or "MACRO_STATEMENT"

        seg.role = role
        seg.prompt = positive
        seg.negative = negative

        # Pass positive prompt into the slice_def for process_slice
        rep_slice.text = positive

        try:
            res = process_slice(
                engine=_engine,
                vlib=vlib,
                cat_index=cat_index,
                channel=channel,
                day=day,
                job_id=job_id,
                slice_def=rep_slice,
                out_path=img_path,
                used_phashes=used_phashes,
                negative=negative,
            )
            results.append(res)
            # If this segment represents a hero/person portrait concept, remember it as an anchor
            if concept_key and _is_hero_portrait_concept(concept_key):
                sticky_assets.setdefault(concept_key, str(img_path))
            if getattr(res, "chosen_phash", None):
                used_phashes.append(res.chosen_phash)
        except Exception as e:
            _log(f"ERROR generating image for segment {seg.index}: {e}")

    # 9) Write manifest.json for compatibility (frames list)
    frames = []
    for seg in segments:
        frames.append({
            "index": seg.index,
            "t0": seg.t0,
            "t1": seg.t1,
            "file": seg.file or f"Scenes/scene_{seg.index+1:03d}.jpg",
            "text": seg.caption_text,
            "caption": seg.caption_text,
            "prompt": seg.prompt,
            "topic": seg.topic,
            "role": seg.role,
            "concept_id": seg.concept_id,
            "chapter": getattr(seg, "chapter", ""),
            "prev_prompt": segments[seg.index-1].prompt if seg.index > 0 else "",
        })

    manifest = {
        "job": job_id,
        "audio_duration": audio_dur,
        "schedule": schedule_spec,
        "frames": frames,
    }

    manifest_path = job_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    # Summary
    total = len(results)
    reused = sum(1 for r in results if getattr(r, "reused", False))
    new_used = total - reused
    attempts = sum(getattr(r, "attempts", 0) for r in results)
    bright_drop = sum(getattr(r, "bright_drops", 0) for r in results)
    _log(f"Summary | slices={total} reused={reused} new_used={new_used} attempts={attempts} bright_drop={bright_drop}")

    _log(f"✅ {len(segments)} scenes ready in {job_root}")


# --------------------------------------------------------------
# CLI entry (optional manual run)
# --------------------------------------------------------------

if __name__ == "__main__":
    """
    Optional manual usage:

      python scene_orchestrator.py <Channel> <YYYY-MM-DD> <job_id> <wav_path> <script_path>

    This is mostly for debugging. The real pipeline calls run_scene from auto_on_job.py.
    """
    if len(sys.argv) < 6:
        print("Usage: scene_orchestrator.py <Channel> <YYYY-MM-DD> <job_id> <wav_path> <script_path>")
        sys.exit(64)

    ch   = sys.argv[1]
    day  = sys.argv[2]
    jid  = sys.argv[3]
    wavp = Path(sys.argv[4])
    scrp = Path(sys.argv[5])

    run_scene(ch, day, jid, wavp, scrp, {})