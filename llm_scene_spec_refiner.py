#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
llm_scene_spec_refiner.py — LLM-powered refinement of SceneSpec for SysVisuals.

This module allows the planner to take a baseline, heuristic SceneSpec
(e.g. from infer_scene_spec) and refine it using the LLM, so that
scene_type, metaphor, camera, scale, chart usage/style, and required /
avoid elements better match the slice text and channel's theme map.

It is designed to be OPTIONAL:
  - If the LLM fails or returns invalid JSON, the original spec is used.
  - It only refines fields; it does not try to generate a brand-new spec
    from scratch.

The refiner is channel-agnostic; allowed scene_types and metaphors are
derived from metaphor_bank.json, and pacing constraints are handled
upstream by scene_planner.py.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, List

import json

try:
    from gpt_client import run_gpt_json
except Exception:
    # Fallback stub so imports still work if gpt_client is unavailable
    def run_gpt_json(prompt: str, schema: Dict[str, Any], model: str = "gpt-4.1") -> Dict[str, Any]:
        return {}


from config_loader import load_metaphor_bank, get_default_channel


# ---------------------------------------------------------------------------
# Allowed values pulled from metaphor_bank
# ---------------------------------------------------------------------------

def _load_allowed_scene_types_and_metaphors() -> Dict[str, List[str]]:
    channel = get_default_channel()
    mb = load_metaphor_bank(channel)
    themes = mb.get("themes", {}) or {}
    scene_types: List[str] = []
    metaphors: List[str] = []

    for theme_cfg in themes.values():
        for st in theme_cfg.get("scene_type_priority", []) or []:
            if st not in scene_types:
                scene_types.append(st)
        for m in theme_cfg.get("metaphors", []) or []:
            if m not in metaphors:
                metaphors.append(m)

    return {
        "scene_types": scene_types,
        "metaphors": metaphors,
    }


_ALLOWED = _load_allowed_scene_types_and_metaphors()


# ---------------------------------------------------------------------------
# JSON schema for refined SceneSpec
# ---------------------------------------------------------------------------

def _refine_schema() -> Dict[str, Any]:
    """
    Schema for the LLM's JSON output. It is intentionally loose but
    disciplined enough for us to merge safely into the base spec.
    """
    return {
        "type": "object",
        "properties": {
            "scene_type": {"type": "string", "nullable": True},
            "metaphor": {"type": "string", "nullable": True},
            "era": {"type": "string", "nullable": True},
            "focus": {"type": "string", "nullable": True},
            "camera": {"type": "string", "nullable": True},
            "scale": {"type": "string", "nullable": True},
            "chart_usage": {"type": "string", "nullable": True},
            "chart_style": {"type": "string", "nullable": True},
            "requires": {
                "type": "array",
                "items": {"type": "string"},
                "nullable": True,
            },
            "avoid": {
                "type": "array",
                "items": {"type": "string"},
                "nullable": True,
            },
            "nice_to_have": {
                "type": "array",
                "items": {"type": "string"},
                "nullable": True,
            },
        },
        "required": [],
    }


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _build_prompt(slice_text: str, base_spec: Dict[str, Any], theme: Optional[str]) -> str:
    allowed_scene_types = ", ".join(_ALLOWED.get("scene_types", []))
    allowed_metaphors = ", ".join(_ALLOWED.get("metaphors", []))

    base_json = json.dumps(base_spec, ensure_ascii=False, indent=2)

    return f"""
You are a senior storyboard artist and director for a long-form financial, geopolitical and macro-economic documentary channel.

Your job is to refine a baseline SCENE SPEC so that the scene visually matches the script slice in a clear, symbolic, and non-repetitive way.

INPUT SLICE:
\"\"\"{slice_text}\"\"\"

CURRENT THEME (may be null):
{theme}

BASELINE SCENE SPEC (from heuristic engine):
```json
{base_json}
```

ALLOWED scene_type values (prefer one of these if possible):
[{allowed_scene_types}]

ALLOWED metaphor values (prefer one of these if possible):
[{allowed_metaphors}]

TASK:
- Gently improve the scene_type, metaphor, focus, camera, scale, and chart usage/style so they tell the story more clearly.
- You must stay within the channel's tone: painterly, symbolic, consistent with long-form documentary visuals.
- Do NOT invent wild new metaphors; choose from the allowed ones or leave metaphor null.
- You may add or remove items in requires[], avoid[], nice_to_have[] to better match the slice.
- If the baseline fields are already good, keep them.

OUTPUT:
Return ONLY valid JSON, matching this shape:
{{
  "scene_type": "<string-or-null>",
  "metaphor": "<string-or-null>",
  "era": "<string-or-null>",
  "focus": "<string-or-null>",
  "camera": "<string-or-null, e.g. 'wide' | 'overhead' | 'medium' | 'closeup'>",
  "scale": "<string-or-null, e.g. 'macro' | 'household' | 'street'>",
  "chart_usage": "<string-or-null, e.g. 'none' | 'device' | 'wall' | 'map_overlay' | 'integrated'>",
  "chart_style": "<string-or-null, e.g. 'candles' | 'smooth_line' | 'pie' | 'gauge'>",
  "requires": ["optional", "visual elements"],
  "avoid": ["optional", "elements_to_avoid"],
  "nice_to_have": ["optional", "secondary_elements"]
}}
If you do not want to change a field, set it to null or omit it.
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def refine_scene_spec_llm(
    slice_text: str,
    base_spec: Dict[str, Any],
    theme: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Refine a baseline SceneSpec using the LLM.

    Args:
        slice_text: raw transcript for this slice
        base_spec: dict derived from SceneSpec.to_dict()
        theme: optional theme string (from LLM or heuristic)

    Returns:
        A new dict representing the refined SceneSpec. If refinement
        fails, base_spec is returned unchanged.
    """
    if not slice_text or not slice_text.strip():
        return base_spec

    prompt = _build_prompt(slice_text, base_spec, theme)
    schema = _refine_schema()

    try:
        resp = run_gpt_json(prompt, schema=schema)
        if not isinstance(resp, dict):
            return base_spec

        refined: Dict[str, Any] = dict(base_spec)

        # Merge scalar fields if provided and non-empty
        for key in ["scene_type", "metaphor", "era", "focus", "camera", "scale", "chart_usage", "chart_style"]:
            val = resp.get(key, None)
            if isinstance(val, str) and val.strip():
                refined[key] = val.strip()

        # Merge list fields if provided and non-empty
        for key in ["requires", "avoid", "nice_to_have"]:
            lst = resp.get(key, None)
            if isinstance(lst, list) and lst:
                # Normalize to unique strings
                cleaned = [str(x).strip() for x in lst if str(x).strip()]
                if cleaned:
                    if key in refined and isinstance(refined[key], list):
                        merged = list({*refined[key], *cleaned})
                    else:
                        merged = list(set(cleaned))
                    refined[key] = merged

        return refined
    except Exception:
        return base_spec
