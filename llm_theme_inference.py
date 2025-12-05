#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
llm_theme_inference.py — LLM‑powered theme inference for SysVisuals.

This module replaces the weak keyword-based `_infer_theme_for_slice`
with a robust JSON-structured inference using your existing LLM
pipeline (gpt_client). It produces a single theme label aligned with
metaphor_bank.json's keys.

The output is intentionally simple: one theme per slice.
ScenePlanner then uses pacing_policy + metaphor_bank to determine
scene_type, metaphors, etc.

All logic here is channel-agnostic; the channel-specific theme list
comes from metaphor_bank.json.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, List
import json
import os

# Import your LLM wrapper
try:
    from gpt_client import run_gpt_json
except Exception:
    # Fallback: define a stub so file still imports for testing
    def run_gpt_json(prompt: str, schema: Dict[str, Any], model: str = "gpt-4.1") -> Dict[str, Any]:
        return {"theme": None}


# ---------------------------------------------------------------------------
# Load metaphor bank (for allowed theme list)
# ---------------------------------------------------------------------------

from config_loader import load_metaphor_bank, get_default_channel


def _load_allowed_themes() -> List[str]:
    channel = get_default_channel()
    mb = load_metaphor_bank(channel)
    themes = list(mb.get("themes", {}).keys())
    if not themes:
        return []
    return themes


_ALLOWED_THEMES = _load_allowed_themes()


# ---------------------------------------------------------------------------
# Build schema for JSON output
# ---------------------------------------------------------------------------

def _theme_schema() -> Dict[str, Any]:
    """
    Pydantic-like schema: Tell the LLM it must output JSON with:
      { "theme": "<string-or-null>" }
    """
    return {
        "type": "object",
        "properties": {
            "theme": {
                "type": "string",
                "nullable": True
            }
        },
        "required": ["theme"]
    }


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _build_prompt(slice_text: str) -> str:
    allowed_str = ", ".join(_ALLOWED_THEMES) if _ALLOWED_THEMES else ""
    return f"""
You are a theme classifier for a financial, geopolitical and macro‑economic documentary channel.

INPUT SLICE:
\"\"\"{slice_text}\"\"\"

TASK:
Determine the single best-fitting narrative theme from the list below.
If none apply, return null.

ALLOWED THEMES:
[{allowed_str}]

OUTPUT:
Return ONLY valid JSON:
{{"theme": "<one theme or null>"}}
"""


# ---------------------------------------------------------------------------
# Main inference function
# ---------------------------------------------------------------------------

def infer_theme_llm(slice_text: str) -> Optional[str]:
    """
    Use the LLM to infer a slice's theme.
    Returns:
        theme (str) or None
    """
    if not slice_text or not slice_text.strip():
        return None

    prompt = _build_prompt(slice_text)
    schema = _theme_schema()

    try:
        resp = run_gpt_json(prompt, schema=schema)
        theme = resp.get("theme")
        if theme and isinstance(theme, str):
            if theme in _ALLOWED_THEMES:
                return theme
        return None
    except Exception:
        return None
