

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scene_planner.py — high-level scene planning for visual automation.

This module produces a structured "scene plan" for each audio slice,
so that SceneEngine and prompt_style.json can generate prompts in a
more deterministic and semantically accurate way.

It is deliberately rule-based and lightweight so it can run without
external calls, but the structure is designed so that you can later
swap in an LLM-backed planner if desired (by filling the same schema).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import re
from collections import defaultdict

# Config loader and LLM theme inference
from config_loader import load_metaphor_bank, load_pacing_policy, load_entity_styles
from llm_theme_inference import infer_theme_llm
from llm_scene_spec_refiner import refine_scene_spec_llm

try:
    # SliceDef is defined in slices.py and already used in scene_gen.py
    from slices import SliceDef
except Exception:  # pragma: no cover - keep loose for import order
    SliceDef = Any  # type: ignore[misc]




# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class SceneSpec:
    """
    Structured description of what a scene should depict.

    Fields are intentionally generic so they can be used across niches
    (finance, history, geopolitics, etc.) and mapped onto templates in
    prompt_style.json or SceneEngine.
    """
    index: int
    scene_type: str
    era: Optional[str] = None
    chapter: Optional[str] = None
    subject: Optional[str] = None
    focus: Optional[str] = None
    metaphor: Optional[str] = None
    chart_usage: Optional[str] = None   # none / device / wall / map_overlay / coin_bars
    chart_style: Optional[str] = None   # candles / smooth_line / pie / gauge
    camera: Optional[str] = None        # wide / overhead / medium / closeup
    scale: Optional[str] = None         # macro / micro / household / street
    requires: Optional[List[str]] = None
    nice_to_have: Optional[List[str]] = None
    avoid: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Drop None fields to keep JSON cleaner
        return {k: v for k, v in d.items() if v is not None}


# ---------------------------------------------------------------------------
# Heuristic rules
# ---------------------------------------------------------------------------


def _lc(text: str) -> str:
    return (text or "").lower()


def _contains_any(text_lc: str, keywords: List[str]) -> bool:
    return any(k in text_lc for k in keywords)




def infer_scene_spec(idx: int, total: int, slice_text: str) -> SceneSpec:
    """
    Infer a structured SceneSpec from a single slice of text.

    This is a deterministic, keyword-and-pattern based planner that
    gives SceneEngine stronger guidance than plain text. It is NOT
    meant to be perfect, but it greatly reduces randomness and is
    designed so you can later replace this logic with an LLM that
    produces the same schema.
    """
    t = (slice_text or "").strip()
    tl = _lc(t)

    # Defaults
    scene_type = "generic"
    era: Optional[str] = None
    subject: Optional[str] = None
    focus: Optional[str] = None
    metaphor: Optional[str] = None
    chart_usage: Optional[str] = None
    chart_style: Optional[str] = None
    camera: Optional[str] = None
    scale: Optional[str] = None
    requires: List[str] = []
    nice_to_have: List[str] = []
    avoid: List[str] = []

    # --- Era detection ----------------------------------------------------
    if _contains_any(tl, ["roman", "rome", "ancient", "imperial rome"]):
        era = "ancient"
    elif _contains_any(tl, ["medieval", "middle ages"]):
        era = "medieval"
    elif _contains_any(tl, ["renaissance"]):
        era = "renaissance"
    elif _contains_any(tl, ["18th century", "1700s", "1700s"]):
        era = "1700s"
    elif _contains_any(tl, ["19th century", "1800s", "industrial revolution"]):
        era = "1800s"
    elif _contains_any(tl, ["20th century", "1900s", "cold war", "great depression"]):
        era = "1900s"

    # --- High-level subject / theme --------------------------------------
    if _contains_any(tl, ["empire", "empires", "imperial", "colonial", "superpower"]):
        subject = "empire"
    elif _contains_any(tl, ["stock", "stocks", "market", "equity", "shares"]):
        subject = "stocks"
    elif _contains_any(tl, ["currency", "currencies", "exchange rate", "fx"]):
        subject = "currency"
    elif _contains_any(tl, ["war", "battle", "army", "invasion", "front line"]):
        subject = "war"
    elif _contains_any(tl, ["tax", "income tax", "state tax", "tax haven"]):
        subject = "tax"
    elif _contains_any(tl, ["holy city", "holy cities", "pilgrimage", "sacred"]):
        subject = "holy_cities"

    # --- Scene type & metaphor -------------------------------------------
    # Empire / territory
    if subject == "empire":
        if _contains_any(tl, ["map", "territory", "territories", "borders", "frontiers", "lost control", "lost their grip"]):
            scene_type = "empire_map"
            metaphor = "territory_chipping"
            requires.append("map")
            avoid.extend(["chart_device", "modern_office"])
            scale = "macro"
            camera = "overhead"
        elif _contains_any(tl, ["pattern", "cycle", "stage", "trajectory"]):
            scene_type = "empire_clocks"
            metaphor = "time_cycle"
            requires.append("clock")
            avoid.append("chart_device")
        else:
            scene_type = "empire_capital"
            metaphor = "capital_city"
            avoid.append("chart_device")

    # War / battle
    elif subject == "war":
        if _contains_any(tl, ["battlefield", "front line", "campaign"]):
            scene_type = "battlefield"
            metaphor = "front_line"
            requires.append("armies")
            avoid.append("chart_device")
        elif _contains_any(tl, ["air force", "fighter", "bomber", "air strike", "airstrike", "jets"]):
            scene_type = "military_deployment"
            metaphor = "air_power"
            requires.append("warplanes")
        else:
            scene_type = "war_map"
            metaphor = "military_map"
            requires.append("map")
            avoid.append("chart_device")

    # Taxes / migration
    elif subject == "tax":
        if _contains_any(tl, ["moved from", "relocated", "moved to", "left a high tax state", "no income tax state"]):
            scene_type = "tax_map"
            metaphor = "tax_migration"
            requires.append("map")
            chart_usage = "none"
        else:
            scene_type = "tax_document"
            metaphor = "tax_forms"

    # Holy cities
    elif subject == "holy_cities":
        scene_type = "holy_cities"
        metaphor = "pilgrimage"
        requires.append("holy_architecture")
        avoid.append("chart_device")

    # Currency
    elif subject == "currency":
        if _contains_any(tl, ["exchange", "exchange rate", "swap", "trade currencies"]):
            scene_type = "currency_exchange"
            metaphor = "hands_exchange"
            requires.append("hands_money")
        elif _contains_any(tl, ["global reserve", "reserve currency", "hegemonic currency"]):
            scene_type = "currency_throne"
            metaphor = "currency_throne"
            requires.append("symbols_money")
        else:
            scene_type = "currency_generic"
            metaphor = "coins_and_notes"

    # Stocks / trading / finance micro
    elif subject == "stocks":
        # Short sellers, wins/losses
        if _contains_any(tl, ["short sellers win", "shorts win", "bet against and won"]):
            scene_type = "short_sellers_win"
            metaphor = "celebration"
            chart_usage = "wall"
            chart_style = "candles_down"
            requires.extend(["trading_hall", "chart"])
        elif _contains_any(tl, ["short sellers lose", "shorts got squeezed", "short squeeze", "bet against and lost"]):
            scene_type = "short_sellers_lose"
            metaphor = "squeeze"
            chart_usage = "wall"
            chart_style = "candles_up"
            requires.extend(["trading_hall", "chart"])
        # Timing
        elif _contains_any(tl, ["timing is everything", "got the timing right", "missed the timing"]):
            scene_type = "timing_clock"
            metaphor = "clock_chart"
            chart_usage = "integrated"
            chart_style = "smooth_line"
            requires.append("clock")
        # Board game / game of markets
        elif _contains_any(tl, ["game of", "board game", "playing the market"]):
            scene_type = "market_boardgame"
            metaphor = "boardgame"
            chart_usage = "none"
        # Who loses
        elif _contains_any(tl, ["who loses", "loses out", "gets hurt most", "pays the price"]):
            scene_type = "who_loses_scale"
            metaphor = "scale"
            chart_usage = "none"
        # Money printers
        elif _contains_any(tl, ["money printer", "printing money", "flooding the system", "liquidity injection"]):
            scene_type = "money_printer"
            metaphor = "printer"
            chart_usage = "none"
        # Gold allocation
        elif _contains_any(tl, ["allocate to gold", "gold allocation", "portion of portfolio in gold", "hedge with gold"]):
            scene_type = "gold_allocation"
            metaphor = "gold_pie"
            chart_usage = "none"
        # Rocket stocks
        elif _contains_any(tl, ["stock took off", "rocket", "moonshot", "skyrocketed", "shot up"]):
            scene_type = "rocket_gain"
            metaphor = "rocket"
            chart_usage = "none"
        else:
            # Generic finance / trading context
            if _contains_any(tl, ["trader", "trading floor", "hedge fund", "fund manager"]):
                scene_type = "trading_hall"
                chart_usage = "wall"
                chart_style = "candles"
                requires.append("chart")
            elif _contains_any(tl, ["retail investor", "ordinary investor", "small investor", "home trader"]):
                scene_type = "retail_holding"
                chart_usage = "device_or_wall"
                chart_style = "smooth_line"
                requires.append("chart")
                scale = "household"
            else:
                scene_type = "stock_chart"
                chart_usage = "wall"
                chart_style = "smooth_line"
                requires.append("chart")

    # Information networks (macro)
    if _contains_any(tl, ["information spreads", "spread instantly", "network of", "web of", "information network", "signals traveled", "news traveled"]):
        # Only override if we don't already have a strong type
        if scene_type in ("generic", "empire_capital"):
            scene_type = "info_network"
            metaphor = "network_nodes"
            chart_usage = "none"

    # U.S. / superpower context – hint modern era if unknown
    if era is None and _contains_any(tl, ["united states", "america", "u.s.", "us "]):
        era = "modern"

    # Camera / scale defaults
    if camera is None:
        if scene_type in ("empire_map", "war_map", "tax_map", "info_network"):
            camera = "overhead"
        elif scene_type in ("battlefield", "holy_cities"):
            camera = "wide"
        elif scene_type in ("retail_holding", "who_loses_scale"):
            camera = "medium"
        else:
            camera = "wide"

    if scale is None:
        if scene_type in ("empire_map", "war_map", "info_network"):
            scale = "macro"
        elif scene_type in ("retail_holding", "who_loses_scale"):
            scale = "household"
        else:
            scale = "macro"

    # Drop empty lists
    requires = requires or None
    nice_to_have = nice_to_have or None
    avoid = avoid or None

    return SceneSpec(
        index=idx,
        scene_type=scene_type,
        era=era,
        subject=subject,
        focus=focus,
        metaphor=metaphor,
        chart_usage=chart_usage,
        chart_style=chart_style,
        camera=camera,
        scale=scale,
        requires=requires,
        nice_to_have=nice_to_have,
        avoid=avoid,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_scene_plan(script_text: str, slices: List[SliceDef]) -> Dict[int, Dict[str, Any]]:
    """
    Build a scene plan for all slices in a job.

    Returns:
        A dict mapping slice index → spec dict. This is designed to be
        serialized as JSON and consumed by scene_gen.py + SceneEngine.
    """
    total = len(slices)
    plan: Dict[int, Dict[str, Any]] = {}

    metaphor_bank = load_metaphor_bank()
    pacing = load_pacing_policy()
    entity_styles = load_entity_styles()

    themes_cfg = metaphor_bank.get("themes", {})
    pacing_defaults = pacing.get("defaults", {})
    max_per_video = pacing_defaults.get("max_per_video", {})
    spacing_cfg = pacing_defaults.get("spacing", {})
    no_immediate = set(spacing_cfg.get("no_immediate_repeat", []))
    min_gap = spacing_cfg.get("min_gap", {})
    families_cfg = pacing.get("families", {})
    family_spacing = pacing.get("family_spacing", {}).get("family_min_gap", {})
    high_impact_caps = pacing.get("high_impact_caps", {})
    intensity_score = pacing.get("intensity_score", {})
    chapter_score_limits = pacing.get("chapter_score_limits", {})

    used_per_type: Dict[str, int] = defaultdict(int)
    last_used_idx: Dict[str, int] = {}
    family_last_used_idx: Dict[str, int] = defaultdict(lambda: -9999)
    high_impact_used: Dict[str, int] = defaultdict(int)
    chapter_intensity: Dict[str, int] = defaultdict(int)  # simple single-chapter tracking for now

    # Simple helper: map scene_type to family
    scene_type_to_family: Dict[str, str] = {}
    for fam, members in families_cfg.items():
        for st in members:
            scene_type_to_family[st] = fam

    for sl in slices:
        text = getattr(sl, "text", "") or ""
        tl = _lc(text)

        spec = infer_scene_spec(sl.index, total, text)
        spec_dict = spec.to_dict()

        # Theme inference for this slice: LLM first, optional heuristic fallback
        theme = infer_theme_llm(text)
        if not theme:
            # Lightweight heuristic fallback if LLM returns nothing
            tl = _lc(text)
            if _contains_any(tl, ["empire", "empires", "superpower", "hegemon"]):
                theme = "empire_cycle"
            elif _contains_any(tl, ["short sellers", "trader", "hedge fund", "trading floor"]):
                theme = "trading_behaviour"
        if theme:
            spec_dict["theme"] = theme

        # Refine the baseline spec using the LLM, if possible
        spec_dict = refine_scene_spec_llm(text, spec_dict, theme=theme)

        # Adjust scene_type using metaphor_bank theme mapping when available
        desired_scene_type = spec_dict.get("scene_type", "generic")
        candidate_scene_types: List[str] = []
        if theme and theme in themes_cfg:
            candidate_scene_types = themes_cfg[theme].get("scene_type_priority", []) or []
        if not candidate_scene_types:
            candidate_scene_types = [desired_scene_type]

        # Respect pacing_policy when choosing final scene_type
        idx = sl.index
        chosen_type = desired_scene_type
        for st in candidate_scene_types:
            # Max per video
            if st in max_per_video and used_per_type[st] >= max_per_video[st]:
                continue
            # No immediate repeat
            if st in no_immediate and last_used_idx.get(st, -9999) == idx - 1:
                continue
            # Min gap per type
            if st in min_gap and idx - last_used_idx.get(st, -9999) <= min_gap[st]:
                continue
            # Family spacing
            fam = scene_type_to_family.get(st)
            if fam and fam in family_spacing:
                if idx - family_last_used_idx[fam] <= family_spacing[fam]:
                    continue
            # High impact caps
            if st in high_impact_caps and high_impact_used[st] >= high_impact_caps[st]:
                continue
            # Intensity per chapter: use default limit for now
            intensity = intensity_score.get(st, 0)
            limit = chapter_score_limits.get("default", 9999)
            if chapter_intensity["default"] + intensity > limit:
                continue

            # If we reach here, st is acceptable
            chosen_type = st
            break

        spec_dict["scene_type"] = chosen_type

        # Update pacing counters
        used_per_type[chosen_type] += 1
        last_used_idx[chosen_type] = idx
        fam = scene_type_to_family.get(chosen_type)
        if fam:
            family_last_used_idx[fam] = idx
        if chosen_type in high_impact_caps:
            high_impact_used[chosen_type] += 1
        chapter_intensity["default"] += intensity_score.get(chosen_type, 0)

        # Attach entity style hints if any entity name appears in this slice
        es_entities = entity_styles.get("entities", {})
        primary_entity_key: Optional[str] = None
        for key, estyle in es_entities.items():
            disp = _lc(estyle.get("display_name", ""))
            if disp and disp in tl:
                primary_entity_key = key
                break
        if primary_entity_key:
            estyle = es_entities[primary_entity_key]
            spec_dict["primary_entity"] = primary_entity_key
            spec_dict["entity_display_name"] = estyle.get("display_name")
            spec_dict["entity_family"] = estyle.get("family")
            spec_dict["entity_role"] = estyle.get("role")
            if "palette" in estyle:
                spec_dict["entity_palette"] = estyle["palette"]
            if "architecture" in estyle:
                spec_dict["entity_architecture"] = estyle["architecture"]
            if "symbols" in estyle:
                spec_dict["entity_symbols"] = estyle["symbols"]
            if "environments" in estyle:
                spec_dict["entity_environments"] = estyle["environments"]
            # Merge avoid hints into spec_dict.avoid if present
            avoid_list = spec_dict.get("avoid") or []
            if isinstance(avoid_list, list) and "avoid" in estyle:
                avoid_list = list(set(avoid_list + estyle["avoid"]))
                spec_dict["avoid"] = avoid_list

        plan[sl.index] = spec_dict

    return plan


def save_scene_plan(plan: Dict[int, Dict[str, Any]], job_root: Path, job_id: str) -> Path:
    """
    Save a scene plan dict to disk as JSON.

    Filename convention:
        <job_id>_scene_plan.json
    """
    out = job_root / f"{job_id}_scene_plan.json"
    out.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    return out