

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
narrative_planner.py — script-level narrative understanding for visuals.

This module analyzes the full script and its slices to produce a
high-level "narrative plan" that can guide scene_planner.py and the
SceneEngine toward more coherent, chapter-aware visuals.

It is intentionally heuristic and lightweight so it can run without
external services, but the data structures are designed so you can
later swap in an LLM-backed planner that fills the same schema.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import re

try:
    from slices import SliceDef
except Exception:  # pragma: no cover
    SliceDef = Any  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ChapterSpec:
    """
    A logical chapter or section in the script.

    slice_start / slice_end are inclusive indices into the slice list.
    """
    id: str
    title: str
    slice_start: int
    slice_end: int
    theme: str
    dominant_entities: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}


@dataclass
class EntitySpec:
    """
    A named entity that appears throughout the video.

    This is intentionally simple: just enough to drive consistent
    visual styling via entity_styles.json or prompt hints.
    """
    name: str
    role: str            # "empire", "country", "person", "company", "asset_class", ...
    tags: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}


@dataclass
class NarrativePlan:
    """
    Full narrative plan for a script.

    chapters: high-level segments of the story
    entities: cross-cutting actors referenced throughout
    global_themes: coarse labels summarizing the whole story
    """
    chapters: List[ChapterSpec]
    entities: Dict[str, EntitySpec]
    global_themes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chapters": [c.to_dict() for c in self.chapters],
            "entities": {k: v.to_dict() for k, v in self.entities.items()},
            "global_themes": list(self.global_themes),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lc(text: str) -> str:
    return (text or "").lower()


def _contains_any(text_lc: str, keywords: List[str]) -> bool:
    return any(k in text_lc for k in keywords)


# ---------------------------------------------------------------------------
# Entity extraction (very simple heuristic)
# ---------------------------------------------------------------------------


def extract_entities(script_text: str) -> Dict[str, EntitySpec]:
    """
    Extremely lightweight entity extraction tuned for finance + history.

    It looks for known patterns and keywords; you can extend this with
    regexes or a small dictionary as needed.
    """
    t = _lc(script_text)
    entities: Dict[str, EntitySpec] = {}

    def add_entity(name: str, role: str, tags: Optional[List[str]] = None) -> None:
        key = name.lower()
        if key not in entities:
            entities[key] = EntitySpec(name=name, role=role, tags=tags)

    # Empires / countries
    if _contains_any(t, ["roman empire", "rome"]):
        add_entity("Roman Empire", "empire", ["ancient", "europe"])
    if _contains_any(t, ["ottoman empire", "istanbul", "constantinople"]):
        add_entity("Ottoman Empire", "empire", ["mediterranean", "middle_east"])
    if _contains_any(t, ["british empire", "britain", "uk ", "london"]):
        add_entity("British Empire", "empire", ["europe", "colonial"])
    if _contains_any(t, ["soviet union", "ussr", "moscow"]):
        add_entity("Soviet Union", "empire", ["20th_century", "eurasia"])
    if _contains_any(t, ["united states", "u.s.", "us ", "america"]):
        add_entity("United States", "country", ["modern_superpower"])

    # Asset classes
    if _contains_any(t, ["gold", "gold bullion", "precious metals"]):
        add_entity("Gold", "asset_class", ["store_of_value"])
    if _contains_any(t, ["bonds", "treasuries", "sovereign debt"]):
        add_entity("Bonds", "asset_class", ["fixed_income"])
    if _contains_any(t, ["stocks", "equities", "equity market"]):
        add_entity("Stocks", "asset_class", ["equity"])

    # Generic fallback: nothing special
    return entities


# ---------------------------------------------------------------------------
# Chapter segmentation
# ---------------------------------------------------------------------------


def segment_chapters(script_text: str, slices: List[SliceDef]) -> List[ChapterSpec]:
    """
    Break the script into coarse "chapters" based on pivots in the text.

    The goal is not to be perfect, but to allow the visual system to
    vary composition and metaphor chapter-by-chapter rather than only
    slice-by-slice.
    """
    chapters: List[ChapterSpec] = []

    n = len(slices)
    if n == 0:
        return chapters

    # Heuristic: look at slice texts and mark pivots where we clearly
    # change from one main subject to another (empires → modern US, etc.)
    subjects: List[str] = []
    for sl in slices:
        tl = _lc(getattr(sl, "text", "") or "")
        if _contains_any(tl, ["roman", "rome"]):
            subjects.append("rome")
        elif _contains_any(tl, ["ottoman", "istanbul"]):
            subjects.append("ottoman")
        elif _contains_any(tl, ["british", "london"]):
            subjects.append("british")
        elif _contains_any(tl, ["soviet union", "ussr", "moscow"]):
            subjects.append("soviet")
        elif _contains_any(tl, ["united states", "u.s.", "us ", "america"]):
            subjects.append("us")
        elif _contains_any(tl, ["short sellers", "trader", "hedge fund"]):
            subjects.append("traders")
        elif _contains_any(tl, ["tax", "income tax", "tax haven"]):
            subjects.append("tax")
        else:
            subjects.append("generic")

    # Group contiguous runs of the same subject into chapters
    current_subject = subjects[0]
    start_idx = 0
    chapter_index = 0

    def make_title(theme: str) -> str:
        mapping = {
            "rome": "Roman chapter",
            "ottoman": "Ottoman chapter",
            "british": "British chapter",
            "soviet": "Soviet chapter",
            "us": "United States chapter",
            "traders": "Traders and markets",
            "tax": "Taxes and migration",
            "generic": "Context and transitions",
        }
        return mapping.get(theme, theme.title())

    for i in range(1, n):
        subj = subjects[i]
        if subj != current_subject:
            # end current chapter
            chapters.append(
                ChapterSpec(
                    id=f"ch{chapter_index}",
                    title=make_title(current_subject),
                    slice_start=start_idx,
                    slice_end=i - 1,
                    theme=current_subject,
                )
            )
            chapter_index += 1
            current_subject = subj
            start_idx = i

    # last chapter
    chapters.append(
        ChapterSpec(
            id=f"ch{chapter_index}",
            title=make_title(current_subject),
            slice_start=start_idx,
            slice_end=n - 1,
            theme=current_subject,
        )
    )

    return chapters


# ---------------------------------------------------------------------------
# Global themes
# ---------------------------------------------------------------------------


def infer_global_themes(script_text: str) -> List[str]:
    """
    Coarse labels summarizing the whole story.
    """
    t = _lc(script_text)
    themes: List[str] = []

    if _contains_any(t, ["empire", "empires", "superpower", "hegemon"]):
        themes.append("empire_cycle")
    if _contains_any(t, ["war", "battle", "conflict", "invasion"]):
        themes.append("war_and_conflict")
    if _contains_any(t, ["currency", "reserve currency", "global reserve"]):
        themes.append("currency_hegemony")
    if _contains_any(t, ["short sellers", "hedge fund", "trader"]):
        themes.append("trading_behaviour")
    if _contains_any(t, ["who loses", "loses out", "gets hurt most"]):
        themes.append("distributional_impact")
    if _contains_any(t, ["tax", "income tax", "tax haven", "moved from", "relocated to"]):
        themes.append("tax_migration")
    if not themes:
        themes.append("generic_economic_story")

    return themes


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_narrative_plan(script_text: str, slices: List[SliceDef]) -> NarrativePlan:
    """
    Build a NarrativePlan from the full script and its slices.
    """
    entities = extract_entities(script_text)
    chapters = segment_chapters(script_text, slices)
    themes = infer_global_themes(script_text)

    # Optionally annotate chapters with rough entities
    for ch in chapters:
        ch_entities: List[str] = []
        for key, ent in entities.items():
            name_lc = key
            # naive: if entity name or role is mentioned in any slice of chapter, attach
            for i in range(ch.slice_start, ch.slice_end + 1):
                tl = _lc(getattr(slices[i], "text", "") or "")
                if ent.name.lower() in tl:
                    ch_entities.append(ent.name)
                    break
        ch.dominant_entities = sorted(set(ch_entities)) or None

    return NarrativePlan(
        chapters=chapters,
        entities=entities,
        global_themes=themes,
    )


def save_narrative_plan(plan: NarrativePlan, job_root: Path, job_id: str) -> Path:
    """
    Save a narrative plan to disk.

    Filename convention:
        <job_id>_narrative_plan.json
    """
    out = job_root / f"{job_id}_narrative_plan.json"
    out.write_text(
        json.dumps(plan.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out