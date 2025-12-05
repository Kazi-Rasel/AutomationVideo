#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""image_pipeline.py

Shared scene-image generation & VisualLibrary integration logic.

This module is **provider-agnostic**. It does not know about Fal/Imagen/Flux
or any specific AI vendor. It only talks to an abstract AIImageEngine
(interface defined in ai_engine.py) and the VisualLibrary / CLIP utilities.

Responsibilities:
  - For job-time scene generation:
      * For each SliceDef (t0, t1, text, index), obtain a final scene image
        at a given output path (Scenes/scene_XXX.jpg).
      * Try to reuse existing images from VisualLibrary based on CLIP
        similarity + phash diversity + label filters.
      * If reuse fails, call AIImageEngine.generate() to create 1..N
        candidate images, run brightness & CLIP checks, assign categories,
        archive and store metadata in VisualLibrary, choose a winner.
  - For manual ingest (ImgToVL):
      * For each user-supplied image in ImgToVL/PutImg, run the same
        brightness & CLIP/category logic and store into VisualLibrary.

This file is the core of your VisualLibrary behavior. It is used by:
  - Scene orchestrators (scene_orchestrator.py, scene_gen.py)
  - ImgToVL/process_to_vl.py for manual imports

All parameters (brightness thresholds, CLIP thresholds, max retries, etc.)
are expressed in generic terms and may come from environment vars or
pipeline.json (see below).
"""

from __future__ import annotations

import json
import os
import shutil
import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Any

# Type alias for VisualLibrary handle
VLib = Dict[str, Any]

from ai_engine import AIImageEngine
from clip_utils import encode_text, encode_image, cosine_similarity, CategoryIndex
from visual_library import (
    vlib_init,
    vlib_store,
    vlib_search,
    vlib_find_by_concept,
    vlib_find_entity_anchor,
    _entity_from_concept_id,
)
from slices import SliceDef
import paths
from prompt_style import load_channel_policy
from PIL import Image, ImageFilter
import numpy as np

# --- central logging setup ---
VA_ROOT = Path(os.environ.get("VA_ROOT", str(Path.home() / "VideoAutomation")))
SYS     = VA_ROOT / "System"
LOG_DIR = SYS / "Logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "image_pipeline.log"

def _log(msg: str) -> None:
    """Log IMG/VL messages to terminal and central log file."""
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[IMG {timestamp}] {msg}"
    # print to terminal
    print(line, flush=True)
    # append to central log file
    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(line + "\n")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

DEFAULT_SCHED = "0-240:5,240-480:7,540-inf:9"

# Brightness: we operate in [0, 1] (normalised luma), but use env/pipeline
# configured values if available.
BRIGHT_MIN = float(os.environ.get("IMG_BRIGHT_MIN", "0.18"))
BRIGHT_MAX = float(os.environ.get("IMG_BRIGHT_MAX", "0.98"))
BRIGHT_TARGET = float(os.environ.get("IMG_BRIGHT_TARGET", "0.62"))


# CLIP similarity thresholds for reuse/new candidates
CLIP_SIM_GOOD = float(os.environ.get("IMG_CLIP_SIM_GOOD", "0.25"))

# Minimum CLIP similarity for a "good enough" newly generated image.
# If an attempt reaches this, we early-stop further generation for that
# slice, but still store all brightness-OK candidates in VisualLibrary.
GEN_MIN_SIM = float(os.environ.get("IMG_GEN_MIN_SIM", "0.20"))

# Diversity in phash Hamming distance
JOB_DIVERSITY_PHASH = int(os.environ.get("IMG_JOB_DIVERSITY_PHASH", "6"))
LIB_DIVERSITY_PHASH = int(os.environ.get("IMG_LIB_DIVERSITY_PHASH", "4"))

# Max number of new candidates to try per slice when reuse fails
MAX_ATTEMPTS = int(os.environ.get("IMG_MAX_ATTEMPTS", "3"))

# Whether to keep all brightness-passing candidates in VisualLibrary or only
# the winner.
STORE_ALL_PASS_MIN = os.environ.get("IMG_STORE_ALL_PASS_MIN", "1") == "1"


def _normalize_to_four_three(path: Path, target_ratio: float = 4.0 / 3.0, tol: float = 0.03) -> None:
    """Ensure the image at `path` is approximately 4:3.

    If the aspect ratio is already within `tol` of 4:3, do nothing.
    If it is moderately off, perform a center crop to exact 4:3.
    If it is very far off, create a blurred-padded 4:3 canvas and paste
    the scaled image onto it. This runs only at ingest/generation time
    as a safety guard and should very rarely trigger.
    """
    try:
        img = Image.open(path)
        fmt = (img.format or "").upper()
    except Exception:
        return

    try:
        img = img.convert("RGB")
        w, h = img.size
        if w <= 0 or h <= 0:
            return
        ratio = float(w) / float(h)

        # Close enough to 4:3.
        if abs(ratio - target_ratio) <= tol:
            # If already a real JPEG, leave as-is; otherwise re-encode once
            # to JPEG so VisualLibrary always contains consistent JPEG files.
            if fmt != "JPEG":
                try:
                    img.save(path, format="JPEG", quality=95)
                except Exception:
                    return
            return

        # If moderately close to 4:3, center-crop to exact 4:3
        if abs(ratio - target_ratio) <= 0.25:
            if ratio > target_ratio:
                # too wide → crop horizontally
                new_w = int(round(h * target_ratio))
                left = max(0, (w - new_w) // 2)
                right = left + new_w
                img = img.crop((left, 0, right, h))
            else:
                # too tall → crop vertically
                new_h = int(round(w / target_ratio))
                top = max(0, (h - new_h) // 2)
                bottom = top + new_h
                img = img.crop((0, top, w, bottom))
        else:
            # Very different aspect ratio → blurred pad to 4:3
            target_h = h
            target_w = int(round(target_h * target_ratio))
            if target_w <= 0 or target_h <= 0:
                return

            # Scale image to fit within target canvas while preserving aspect
            scale = min(float(target_w) / float(w), float(target_h) / float(h))
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))

            fg = img.resize((new_w, new_h), Image.LANCZOS)

            # Create blurred background from original image
            bg = img.resize((target_w, target_h), Image.LANCZOS).filter(ImageFilter.GaussianBlur(radius=24))

            x = (target_w - new_w) // 2
            y = (target_h - new_h) // 2
            bg.paste(fg, (x, y))
            img = bg

        # Save back over original path as JPEG (consistent format in VL)
        img.save(path, format="JPEG", quality=95)
    except Exception:
        # Safety: never let aspect fix crash the pipeline
        return


@dataclass
class SliceResult:
    index: int
    t0: float
    t1: float
    prompt: str
    file: Optional[Path]  # final scene image path, or None if no image
    reused: bool
    attempts: int
    bright_drops: int
    chosen_sim: float
    chosen_label: Optional[str]
    chosen_phash: Optional[str] = None


# ---------------------------------------------------------------------------
# Core: generate or reuse scene image for a single slice
# ---------------------------------------------------------------------------


def process_slice(
    engine: AIImageEngine,
    vlib: VLib,
    cat_index: Optional[CategoryIndex],
    channel: str,
    day: str,
    job_id: str,
    slice_def: SliceDef,
    out_path: Path,
    used_phashes: Sequence[str],
    negative: Optional[str] = None,
) -> SliceResult:
    """
    Generate or reuse an image for a single slice.

    - engine: AIImageEngine implementation (Fal, Imagen, etc.)
    - vlib: VisualLibrary handle (from vlib_init)
    - cat_index: CategoryIndex or None if no categories.json is available
    - channel/day/job_id: for metadata
    - slice_def: SliceDef with t0/t1/index/text
      (may now be called with a semantic prompt representative of a visual segment;
       slice_def may carry extra attributes like `topic`, `role`, and `concept_id`
       used for logging/debugging but not required)
    - out_path: final scene image path under Scenes/
    - used_phashes: collection of phash strings already used in this job

    Returns a SliceResult with details of what happened.

    This may generate up to MAX_ATTEMPTS new images when reuse fails, but will
    stop early for a slice once a candidate reaches GEN_MIN_SIM CLIP similarity.
    """
    prompt = slice_def.text or ""
    topic = getattr(slice_def, "topic", "")
    concept_id = getattr(slice_def, "concept_id", "")
    if topic or concept_id:
        _log(f"slice {slice_def.index} topic={topic} concept={concept_id}")
    role = getattr(slice_def, "role", "")
    reused = False
    attempts = 0
    bright_drops = 0
    chosen_sim = 0.0
    chosen_label: Optional[str] = None

    # Per-topic CLIP similarity threshold (may be refined by channel policy)
    min_clip = CLIP_SIM_GOOD
    try:
        policy = load_channel_policy(channel)
    except Exception:
        policy = None
    if policy and hasattr(policy, "topic_clip_threshold"):
        try:
            override = policy.topic_clip_threshold(topic or "")  # type: ignore[misc]
            if isinstance(override, (int, float)) and override > 0.0:
                min_clip = float(override)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # -1) Hero/entity anchor reuse: keep faces/buildings consistent across jobs
    # ------------------------------------------------------------------
    entity_key = _entity_from_concept_id(concept_id or "")
    if entity_key:
        try:
            ent_hit = vlib_find_entity_anchor(vlib, entity_key, channel=channel)
        except Exception:
            ent_hit = None
        if ent_hit and ent_hit.get("path"):
            ent_src = Path(ent_hit["path"])
            ent_ph = ent_hit.get("phash")
            # Avoid exact-duplicate phash within this job if requested
            if ent_ph and ent_ph in used_phashes:
                ent_hit = None
            else:
                if ent_src.is_file():
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(ent_src, out_path)
                    chosen_label = ent_hit.get("label")
                    chosen_sim = float(ent_hit.get("score", 0.0)) if "score" in ent_hit else 0.0
                    _log(f"🧬 ENTITY ANCHOR REUSE {out_path.name} from {ent_src.name} (entity={entity_key}, label={chosen_label})")
                    return SliceResult(
                        index=slice_def.index,
                        t0=slice_def.t0,
                        t1=slice_def.t1,
                        prompt=prompt,
                        file=out_path,
                        reused=True,
                        attempts=0,
                        bright_drops=0,
                        chosen_sim=chosen_sim,
                        chosen_label=chosen_label,
                        chosen_phash=ent_ph,
                    )

    # ------------------------------------------------------------------
    # 0) Hard reuse by deterministic concept key (cross-job hero/anchor reuse)
    # ------------------------------------------------------------------
    if concept_id:
        try:
            hit = vlib_find_by_concept(vlib, concept_id, channel=channel)
        except Exception:
            hit = None
        if hit and hit.get("path"):
            src_path = Path(hit["path"])
            ph = hit.get("phash")
            # Avoid exact-duplicate phash within this job if requested
            if ph and ph in used_phashes:
                hit = None
            else:
                if src_path.is_file():
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_path, out_path)
                    chosen_label = hit.get("label")
                    chosen_sim = float(hit.get("score", 0.0)) if "score" in hit else 0.0
                    _log(f"🧬 CONCEPT REUSE {out_path.name} from {src_path.name} (concept={concept_id}, label={chosen_label})")
                    return SliceResult(
                        index=slice_def.index,
                        t0=slice_def.t0,
                        t1=slice_def.t1,
                        prompt=prompt,
                        file=out_path,
                        reused=True,
                        attempts=0,
                        bright_drops=0,
                        chosen_sim=chosen_sim,
                        chosen_label=chosen_label,
                        chosen_phash=ph,
                    )

    # ------------------------------------------------------------------
    # 1) Try reuse from VisualLibrary using CLIP + phash constraints
    # ------------------------------------------------------------------
    text_vec = encode_text(prompt) if prompt else None

    if text_vec is not None:
        # Improved reuse logic: try candidate labels from CategoryIndex, or all if not available.
        if cat_index is not None:
            # Get candidate label ids from CategoryIndex, fallback to [None] if empty.
            label_sims = cat_index.labels_for_text_vec(text_vec, top_k=15, min_sim=min_clip)
            labels_to_try = [lbl for lbl, _ in label_sims] if label_sims else [None]
        else:
            labels_to_try = [None]

        best_hit = None
        best_score = float("-inf")
        best_label_id = None
        for label_id in labels_to_try:
            hit = vlib_search(
                vlib,
                prompt=prompt,
                topk=1,
                min_phash_dist=JOB_DIVERSITY_PHASH,
                exclude_hashes=list(used_phashes),
                channel=channel,
                label=label_id,
                topic=topic or None,
                role=role or None,
                prompt_clip=text_vec,
                min_clip=min_clip,
                max_candidates=0,
            )
            if hit and hit.get("path"):
                score = float(hit.get("score", 0.0))
                if score > best_score:
                    best_hit = hit
                    best_score = score
                    best_label_id = label_id
        if best_hit and best_hit.get("path"):
            src_path = Path(best_hit["path"])
            if src_path.is_file():
                out_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, out_path)
                chosen_label = best_hit.get("label") or best_label_id
                chosen_sim = float(best_hit.get("score", 0.0))
                ph = best_hit.get("phash")
                _log(f"🖼 REUSE {out_path.name} from {src_path.name} (label={chosen_label}, sim={chosen_sim:.3f})")
                return SliceResult(
                    index=slice_def.index,
                    t0=slice_def.t0,
                    t1=slice_def.t1,
                    prompt=prompt,
                    file=out_path,
                    reused=True,
                    attempts=0,
                    bright_drops=0,
                    chosen_sim=chosen_sim,
                    chosen_label=chosen_label,
                    chosen_phash=ph,
                )

    # ------------------------------------------------------------------
    # 2) No reuse: generate new candidates via AI engine
    # ------------------------------------------------------------------
    # Candidate records: (archive_path, sim, phash, label). We try up to
    # MAX_ATTEMPTS, but stop early for this slice if an attempt reaches
    # GEN_MIN_SIM CLIP similarity.
    candidates: List[Tuple[Path, float, Optional[str], Optional[str]]] = []
    best_gen_sim = float("-inf")

    # Where we put temporary candidates before archiving
    job_root = paths.build_ready_job_root(channel, job_id, day)
    cand_dir = job_root / "SceneCandidates"
    cand_dir.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        attempts += 1
        tmp_name = f"{job_id}_s{slice_def.index:03d}_a{attempt:02d}.jpg"
        tmp_path = cand_dir / tmp_name

        # Skip unexpected non-image files (extra safety)
        ext = tmp_path.suffix.lower()
        if ext not in [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"]:
            _log(f"⚠ slice {slice_def.index} attempt {attempt}: unsupported file type, skipping")
            continue

        try:
            engine.generate(prompt, tmp_path, seed=None, negative=negative)
        except Exception as e:  # engine is provider-specific; we just log & continue
            _log(f"⚠ slice {slice_def.index} attempt {attempt} failed: {e}")
            continue

        # Basic sanity: file must exist and be non-empty
        if not tmp_path.is_file() or tmp_path.stat().st_size <= 0:
            _log(f"⚠ slice {slice_def.index} attempt {attempt} produced no file")
            continue

        # Brightness check
        try:
            with Image.open(str(tmp_path)) as im:
                im = im.convert("RGB")
                arr = np.array(im).astype("float32")
                luma = float((0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]).mean())
        except Exception:
            _log(f"⚠ slice {slice_def.index} attempt {attempt} brightness check failed")
            bright_drops += 1
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            continue

        luma_norm = luma / 255.0
        if not (BRIGHT_MIN <= luma_norm <= BRIGHT_MAX):
            _log(f"⚠ slice {slice_def.index} {tmp_name} luma={luma_norm:.3f} outside [{BRIGHT_MIN:.3f},{BRIGHT_MAX:.3f}] → drop")
            bright_drops += 1
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            continue

        # Archive path in VisualLibrary/images/<label_id>/…
        vlib_root: Path = vlib["root"]
        # We don't yet know the label; use a temporary UNCATEGORIZED folder and
        # correct label below after CLIP/category assignment.
        temp_label_dir = vlib_root / "images" / "UNCATEGORIZED"
        temp_label_dir.mkdir(parents=True, exist_ok=True)
        archive_name = f"TMP_{job_id}_s{slice_def.index:03d}_a{attempt:02d}.jpg"
        archive_path = temp_label_dir / archive_name

        shutil.move(str(tmp_path), str(archive_path))

        # Aspect ratio safety guard: ensure archive image is effectively 4:3.
        _normalize_to_four_three(archive_path)

        # CLIP embedding
        img_vec = encode_image(archive_path)
        if img_vec is None:
            _log(f"⚠ slice {slice_def.index} CLIP encode failed for {tmp_name}")
            try:
                archive_path.unlink(missing_ok=True)
            except Exception:
                pass
            continue

        # Compute similarity vs prompt text
        sim = 0.0
        if text_vec is not None:
            sim = cosine_similarity(text_vec, img_vec)

        # Determine label via CategoryIndex if available
        label_id: Optional[str] = None
        if cat_index is not None:
            label_id = cat_index.best_label_for_image_vec(img_vec) or "UNCATEGORIZED"
        else:
            label_id = "UNCATEGORIZED"

        # Move normalized archive into its final label-based directory/name
        vlib_root: Path = vlib["root"]
        img_dir = vlib_root / "images" / label_id
        img_dir.mkdir(parents=True, exist_ok=True)
        final_name = f"{label_id}_{job_id}_s{slice_def.index:03d}_a{attempt:02d}.jpg"
        final_path = img_dir / final_name

        shutil.move(str(archive_path), str(final_path))

        # store metadata in VisualLibrary
        vlib_store(
            vlib,
            final_path,
            prompt,
            channel,
            job_id,
            day,
            label=label_id,
            topic=topic or None,
            role=role or None,
            concept_id=concept_id or None,
        )

        # We may want to compute phash here only for local job diversity;
        # we can fetch it from vlib, or recompute using vlib["phash"]
        phash = vlib["phash_of"](final_path) if "phash_of" in vlib else None
        candidates.append((final_path, sim, phash, label_id))

        # Track best similarity for this slice and apply CLIP early-stop:
        # if this attempt is semantically strong enough (>= GEN_MIN_SIM),
        # we stop generating more candidates for this slice.
        if sim > best_gen_sim:
            best_gen_sim = sim
        if text_vec is not None and sim >= GEN_MIN_SIM:
            # Good enough semantic match; no need for further attempts.
            break

    # Decide on winner
    if not candidates:
        # No usable image
        return SliceResult(
            index=slice_def.index,
            t0=slice_def.t0,
            t1=slice_def.t1,
            prompt=prompt,
            file=None,
            reused=False,
            attempts=attempts,
            bright_drops=bright_drops,
            chosen_sim=0.0,
            chosen_label=None,
            chosen_phash=None,
        )

    # sort by similarity; if no text_vec, all sims may be 0.0 but we still pick one
    candidates.sort(key=lambda x: x[1], reverse=True)
    best_path, best_sim, best_phash, best_label = candidates[0]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_path, out_path)
    _log(f"🏆 slice {slice_def.index} winner {best_path.name} (label={best_label}, sim={best_sim:.3f})")

    # record diversity into caller's used_phashes list if desired (caller updates)
    return SliceResult(
        index=slice_def.index,
        t0=slice_def.t0,
        t1=slice_def.t1,
        prompt=prompt,
        file=out_path,
        reused=False,
        attempts=attempts,
        bright_drops=bright_drops,
        chosen_sim=best_sim,
        chosen_label=best_label,
        chosen_phash=best_phash,
    )


# ---------------------------------------------------------------------------
# Manual ingest: process curated user-provided images in ImgToVL/PutImg
# ---------------------------------------------------------------------------


def ingest_manual_images(
    vlib: VLib,
    cat_index: Optional[CategoryIndex],
    img_dir: Path,
    channel: str = "MANUAL",
    job_id: str = "manual_import",
    day: Optional[str] = None,
) -> None:
    """Ingest all images under img_dir into VisualLibrary.

    - Applies the same brightness / CLIP / category logic as job-time images.
    - Uses the base filename (without extension) as the prompt.
    - Re-encodes inputs to JPEG and stores under VisualLibrary/images/<label_id>/…
      and records metadata via vlib_store.
    """
    import datetime

    if day is None:
        day = datetime.date.today().isoformat()

    vlib_root: Path = vlib["root"]

    total = 0
    stored = 0
    dropped = 0

    for p in sorted(img_dir.glob("*")):
        if not p.is_file():
            continue
        # Only allow real image files by extension
        ext = p.suffix.lower()
        if ext not in [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"]:
            _log(f"[VL] ⚠ {p.name}: unsupported file type, skipping")
            # Do NOT delete or move it — leave the file as-is
            dropped += 1
            continue
        total += 1
        prompt = p.stem.replace("_", " ")

        # Brightness check
        try:
            with Image.open(str(p)) as im:
                im = im.convert("RGB")
                arr = np.array(im).astype("float32")
                luma = float((0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]).mean())
        except Exception:
            _log(f"[VL] ⚠ {p.name}: failed to read/compute brightness, skipping")
            dropped += 1
            continue

        luma_norm = luma / 255.0
        if not (BRIGHT_MIN <= luma_norm <= BRIGHT_MAX):
            _log(f"[VL] ⚠ {p.name}: luma={luma_norm:.3f} outside [{BRIGHT_MIN:.3f},{BRIGHT_MAX:.3f}] → skip")
            dropped += 1
            continue

        # Prepare archive path in VisualLibrary for this manual image
        archive_dir = vlib_root / "images"
        archive_dir.mkdir(parents=True, exist_ok=True)
        # Temporary generic name; final label subdir resolved after CLIP/category
        temp_archive = archive_dir / f"TMP_manual_{day}_{p.stem}.jpg"

        # Re-encode to jpeg to normalise format
        try:
            with Image.open(str(p)) as im:
                im = im.convert("RGB")
                im.save(str(temp_archive), format="JPEG", quality=95)
        except Exception as e:
            _log(f"[VL] ⚠ {p.name}: failed to convert/save -> {e}")
            dropped += 1
            continue

        # Aspect ratio safety guard for manual imports
        _normalize_to_four_three(temp_archive)

        img_vec = encode_image(temp_archive)
        if img_vec is None:
            _log(f"[VL] ⚠ {p.name}: CLIP encode failed, skipping")
            dropped += 1
            try:
                temp_archive.unlink()
            except Exception:
                pass
            continue

        label_id: Optional[str] = None
        if cat_index is not None:
            label_id = cat_index.best_label_for_image_vec(img_vec) or "UNCATEGORIZED"
        else:
            label_id = "UNCATEGORIZED"

        # Move normalized archive into its final label-based directory/name
        label_dir = vlib_root / "images" / label_id
        label_dir.mkdir(parents=True, exist_ok=True)
        final_name = f"{label_id}_manual_{day}_{p.stem}.jpg"
        final_path = label_dir / final_name
        shutil.move(str(temp_archive), str(final_path))

        # remove original file after successful import
        try:
            p.unlink()
        except Exception:
            pass

        vlib_store(
            vlib,
            final_path,
            prompt,
            channel,
            job_id,
            day,
            label=label_id,
        )

        stored += 1
        _log(f"[VL] ✅ stored {final_path} (label={label_id}, luma={luma_norm:.3f})")

    _log(f"[VL] Summary | files={total} stored={stored} dropped={dropped}")
