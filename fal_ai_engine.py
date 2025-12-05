# File: /Users/kazirasel/VideoAutomation/System/SysVisuals/Engines/image/fal_ai_engine.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fal_ai_engine.py — Fal.ai / Flux image generation engine.

Responsibilities:
  - Provide image generation for slices via Fal.ai / Flux.
  - Use centralized key management (key_loader.py) to obtain the FAL API key.
  - Expose FAL_KEY via environment variable for downstream Fal clients.
"""

from __future__ import annotations
import os, sys, json, hashlib, time
from pathlib import Path
from typing import Optional, Tuple

# Root paths for centralized key handling
VA_ROOT = Path.home() / "VideoAutomation"
ALLKEYS = VA_ROOT / "System" / "AllKeys"
KEYLOADER_DIR = ALLKEYS / "KeyLoader"

# Ensure we can import key_loader from System/AllKeys/KeyLoader (single, canonical location).
if str(KEYLOADER_DIR) not in sys.path:
    sys.path.insert(0, str(KEYLOADER_DIR))

from key_loader import load_key  # uses ~/VideoAutomation/System/AllKeys/<name>.key

# Expose FAL_KEY, loaded via key_loader (System/AllKeys/fal.key), to any downstream client via env var.
os.environ["FAL_KEY"] = load_key("fal")

# Assume vlib_store and vlib_search are already imported:
# from visual_library import vlib_store, vlib_search

# Other imports and existing code...

def process_slice(slice_def, vlib, out_dir, channel, job_id, day, label, prompt, prompt_clip, used_phashes, GEN_MIN_SIM):
    # Existing code...

    # Locate reuse search in VisualLibrary:
    match = vlib_search(
        vlib,
        prompt=prompt,
        topk=5,
        min_phash_dist=10,
        exclude_hashes=set(used_phashes),
        channel=channel,
        label=label,
        prompt_clip=prompt_clip,
        min_clip=GEN_MIN_SIM,
        max_candidates=20,
        topic=getattr(slice_def, "topic", None),
        role=getattr(slice_def, "role", None),
        concept_id=getattr(slice_def, "concept_id", None),
    )

    # Existing logic to handle match...

    # After generating a new image:
    wh, best_luma = (None, None)  # placeholder for actual values
    best_clip = None

    # Save new image to VisualLibrary with semantic metadata:
    vlib_store(
        vlib,
        out_path,
        prompt=prompt,
        channel=channel,
        job=job_id,
        day=day,
        label=label,
        luma=best_luma,
        clip=best_clip,
        width=wh[0] if wh else None,
        height=wh[1] if wh else None,
        topic=getattr(slice_def, "topic", None),
        role=getattr(slice_def, "role", None),
        concept_id=getattr(slice_def, "concept_id", None),
    )

    # Rest of existing code...