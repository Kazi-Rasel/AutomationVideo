#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scene_engine.py — Rule-based scene selector + prompt builder.

Uses:
  - prompt_style.json  (channel-specific rules + templates)
  - Optional concept kwargs (amount, year_text, etc.)

Goal:
  For each slice:
    raw text  → best matching rule  → template  → final prompt text
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
import os
import re
import logging
from config_loader import load_prompt_style, get_default_channel, get_channel_config_dir

logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    fmt = logging.Formatter("[SCENE_ENGINE] %(levelname)s: %(message)s")
    h.setFormatter(fmt)
    logger.addHandler(h)
logger.setLevel(logging.INFO)


@dataclass
class SceneRule:
    id: str
    priority: int
    any_keywords: List[str]
    template: str


class SceneEngine:
    """
    Stateless rule engine over prompt_style.json.

    - Loads:
        global.prompt_prefix / prompt_suffix / negative_prompt
        rules[]: {id, priority, any_keywords, template}
        templates: {template_id: text with {slice_summary} etc.}
    - Given text + slice_summary + concept kwargs:
        → picks best rule
        → renders template
        → prepends/append global prefix/suffix
    """

    def __init__(self, style_path: Path, config: Optional[Dict[str, Any]] = None):
        """
        Initialize SceneEngine with a style path and optional preloaded config.

        If config is provided, it is used directly. Otherwise, the JSON
        is read from style_path.
        """
        self.style_path = style_path
        if config is not None:
            self.config = config
        else:
            if not style_path.exists():
                raise FileNotFoundError(f"prompt_style.json not found at {style_path}")
            self.config = json.loads(style_path.read_text(encoding="utf-8"))

        self.global_cfg: Dict[str, Any] = self.config.get("global", {})
        self.templates: Dict[str, str] = self.config.get("templates", {})
        self.fallback_template: str = self.config.get("fallback_template", "")

        # Optional diversity control from global config
        self.max_same_template_streak: int = int(self.global_cfg.get("max_same_template_streak", 0) or 0)
        self._recent_templates: List[str] = []

        rules_cfg: List[Dict[str, Any]] = self.config.get("rules", [])
        self.rules: List[SceneRule] = [
            SceneRule(
                id=r.get("id", f"RULE_{i}"),
                priority=int(r.get("priority", 999)),
                any_keywords=[k.lower() for k in r.get("any_keywords", [])],
                template=r.get("template", self.fallback_template),
            )
            for i, r in enumerate(rules_cfg)
        ]

        # Sort rules once by priority; we still re-score by hits each call.
        self.rules.sort(key=lambda r: r.priority)

        logger.info(
            "Loaded style from %s: %d rules, %d templates, fallback=%s",
            style_path,
            len(self.rules),
            len(self.templates),
            self.fallback_template,
        )

    @classmethod
    def from_env(cls, default_root: Optional[Path] = None, channel: Optional[str] = None) -> "SceneEngine":
        """
        Construct a SceneEngine using channel-aware config loading.

        Priority:
          - If VA_PROMPT_STYLE_JSON is set, config_loader will use it.
          - Otherwise, load prompt_style.json from the channel's Config dir.
        """
        if channel is None:
            channel = get_default_channel()
        cfg = load_prompt_style(channel)
        # Determine a representative style_path for logging/meta; fall back to default_root
        try:
            cfg_dir = get_channel_config_dir(channel)
            style_path = cfg_dir / "prompt_style.json"
        except Exception:
            if default_root is not None:
                style_path = default_root / "prompt_style.json"
            else:
                style_path = Path("prompt_style.json")
        return cls(style_path, config=cfg or None)

    def _template_for_scene_type(self, scene_type: Optional[str]) -> Optional[str]:
        """
        Map a high-level scene_type (from scene_planner) to a template id
        defined in prompt_style.json, if possible.
        """
        if not scene_type:
            return None
        st = scene_type.lower()
        mapping = {
            "empire_map": "EMPIRE_MAP_SCENE",
            "battlefield": "BATTLEFIELD_SCENE",
            "military_deployment": "MILITARY_DEPLOYMENT_SCENE",
            "info_network": "INFO_NETWORK_SCENE",
            "holy_cities": "HOLY_CITIES_SCENE",
            "currency_exchange": "CURRENCY_EXCHANGE_SCENE",
            "tax_map": "TAX_MAP_SCENE",
            "short_sellers_win": "SHORT_WIN_SCENE",
            "short_sellers_lose": "SHORT_LOSE_SCENE",
            "who_loses_scale": "LOSERS_SCALE_SCENE",
            "money_printer": "MONEY_PRINTER_SCENE",
            "market_boardgame": "BOARDGAME_MARKET_SCENE",
            "gold_allocation": "GOLD_PIE_SCENE",
            "rocket_gain": "ROCKET_STOCK_SCENE",
            "timing_clock": "TIMING_CLOCK_SCENE",
            "retail_holding": "RETAIL_HOLDING_SCENE",
            "trading_hall": "HEDGE_FUND_SCENE",
            # Extended mappings for richer scene types (only if templates exist)
            "currency_throne": "CURRENCY_THRONE_SCENE",
            "currency_generic": "CURRENCY_GENERIC_SCENE",
            "housing_market": "HOUSING_MARKET_SCENE",
            "job_market": "JOB_MARKET_SCENE",
            "factory_floor": "FACTORY_FLOOR_SCENE",
            "energy_infrastructure": "ENERGY_INFRASTRUCTURE_SCENE",
            "shipping_lanes": "SHIPPING_LANES_SCENE",
            "supply_chain_network": "SUPPLY_CHAIN_SCENE",
            "tech_platforms": "TECH_PLATFORMS_SCENE",
            "data_center": "DATA_CENTER_SCENE",
            "ai_brain": "AI_BRAIN_SCENE",
            "macro_dashboard": "MACRO_DASHBOARD_SCENE",
            "recession_city": "RECESSION_CITY_SCENE",
            "crisis_timeline": "CRISIS_TIMELINE_SCENE",
            "recovery_growth": "RECOVERY_GROWTH_SCENE",
        }
        tmpl = mapping.get(st)
        if tmpl and tmpl in self.templates:
            return tmpl
        return None

    # ------------------------------------------------------------------ #
    # Core matching
    # ------------------------------------------------------------------ #

    def _score_rule(self, rule: SceneRule, text_lc: str) -> Tuple[int, int]:
        """
        Return (hits, priority_score) for a rule.

        - hits: number of keyword matches in text
        - priority_score: negative of priority (lower priority is better)
        """
        hits = 0
        for kw in rule.any_keywords:
            if kw and kw in text_lc:
                hits += 1
        if hits == 0:
            return (0, -999999)

        # Smaller priority is better → higher score
        return (hits, -rule.priority)

    def choose_rule(self, slice_text: str, avoid_templates: Optional[set[str]] = None) -> SceneRule:
        """
        Pick the best rule for the given text.

        Strategy:
          - Lowercase text
          - For each rule: count keyword hits
          - Choose rule with:
                max hits, then lowest priority
          - If no rule hits: use fallback template
        """
        text_lc = (slice_text or "").lower()
        avoid = avoid_templates or set()
        best: Optional[SceneRule] = None
        best_hits = 0
        best_pri_score = -999999

        for rule in self.rules:
            if rule.template in avoid:
                continue
            hits, pri_score = self._score_rule(rule, text_lc)
            if hits == 0:
                continue
            if hits > best_hits or (hits == best_hits and pri_score > best_pri_score):
                best = rule
                best_hits = hits
                best_pri_score = pri_score

        if best is None:
            # Synthetic fallback rule
            fallback = SceneRule(
                id="FALLBACK_GENERIC",
                priority=9999,
                any_keywords=[],
                template=self.fallback_template,
            )
            logger.debug("No rule matched text=%r → using fallback template=%s",
                         slice_text, self.fallback_template)
            return fallback

        logger.debug(
            "Matched rule=%s hits=%d priority=%d for text=%r",
            best.id, best_hits, best.priority, slice_text[:140],
        )
        return best

    # ------------------------------------------------------------------ #
    # Template rendering
    # ------------------------------------------------------------------ #

    def render_template(
        self,
        template_id: str,
        slice_summary: str,
        **concept: Any,
    ) -> str:
        """
        Render the chosen template with the given context.

        Notes:
          - str.format ignores extra kwargs, so concept can contain more
            fields than the template actually uses.
          - Ensures we always pass `slice_summary`.
        """
        tmpl = self.templates.get(template_id)
        if not tmpl:
            # last resort: fallback generic scene
            tmpl = self.templates.get(self.fallback_template, "")
            logger.warning(
                "Template %s missing, using fallback template %s",
                template_id, self.fallback_template,
            )

        ctx = dict(concept)
        ctx.setdefault("slice_summary", slice_summary)

        try:
            return tmpl.format(**ctx)
        except Exception as e:
            logger.error("Template format error for %s: %s", template_id, e)
            # If formatting fails, fall back to raw summary.
            return slice_summary

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def build_prompt(
        self,
        slice_text: str,
        slice_summary: str,
        **concept: Any,
    ) -> Tuple[str, str, Dict[str, Any]]:
        """
        Main entrypoint.

        Returns:
          prompt_text, negative_prompt, debug_meta
        """
        # Try to honor scene_type from concept first, if provided
        scene_type = concept.get("scene_type")
        rule: Optional[SceneRule] = None
        tmpl_from_type = self._template_for_scene_type(scene_type)
        if tmpl_from_type:
            # Try to find a rule that uses this template so rule_id stays meaningful
            for r in self.rules:
                if r.template == tmpl_from_type:
                    rule = r
                    break
            if rule is None:
                # Synthetic rule based purely on scene_type → template mapping
                rule = SceneRule(
                    id=f"SCENETYPE_{scene_type}",
                    priority=0,
                    any_keywords=[],
                    template=tmpl_from_type,
                )
        else:
            rule = self.choose_rule(slice_text)
        # Optional diversity guard: avoid using the same template too many times in a row
        if self.max_same_template_streak > 0 and rule is not None:
            streak = 0
            for tmpl in reversed(self._recent_templates):
                if tmpl == rule.template:
                    streak += 1
                else:
                    break
            if streak >= self.max_same_template_streak:
                alt_rule = self.choose_rule(slice_text, avoid_templates={rule.template})
                rule = alt_rule
        # Update recent template history
        self._recent_templates.append(rule.template)
        if len(self._recent_templates) > 32:
            self._recent_templates = self._recent_templates[-32:]

        # Inject high-level planner hints into concept for template formatting
        concept = dict(concept)  # shallow copy
        # Theme + metaphor
        concept.setdefault("theme", concept.get("theme"))
        concept.setdefault("metaphor", concept.get("metaphor"))
        # Entity style hints
        concept.setdefault("entity_palette", concept.get("entity_palette"))
        concept.setdefault("entity_architecture", concept.get("entity_architecture"))
        concept.setdefault("entity_symbols", concept.get("entity_symbols"))
        concept.setdefault("entity_environments", concept.get("entity_environments"))
        concept.setdefault("primary_entity", concept.get("primary_entity"))
        concept.setdefault("entity_family", concept.get("entity_family"))
        concept.setdefault("entity_role", concept.get("entity_role"))
        # Chart usage + style (from scene planner)
        concept.setdefault("chart_usage", concept.get("chart_usage"))
        concept.setdefault("chart_style", concept.get("chart_style"))

        core = self.render_template(
            rule.template,
            slice_summary=slice_summary,
            **concept,
        )

        prefix = self.global_cfg.get("prompt_prefix", "") or ""
        suffix = self.global_cfg.get("prompt_suffix", "") or ""
        negative = self.global_cfg.get("negative_prompt", "") or ""

        # Add entity-specific avoid hints to negative prompt
        avoid_list = concept.get("avoid")
        if isinstance(avoid_list, list) and avoid_list:
            # Convert avoid keywords into a safe negative string
            avoid_str = ", ".join([str(a) for a in avoid_list])
            negative = negative + " " + f"avoid: ({avoid_str})"

        parts = [prefix.strip(), core.strip(), suffix.strip()]
        prompt = " ".join(p for p in parts if p)

        meta = {
            "rule_id": rule.id,
            "template": rule.template,
            "style_path": str(self.style_path),
        }

        return prompt, negative, meta