#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
concept_extractor.py — Extracts structured hints from a slice of text.

Safe to over-supply concept keys; templates will only use what they define.
"""

from __future__ import annotations
from typing import Dict, Any
import re


def _shorten(text: str, max_len: int = 180) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if len(text) <= max_len:
        return text
    # Cut on word boundary if possible
    cut = text[:max_len].rsplit(" ", 1)[0]
    return cut + "…"


def extract_concept(slice_text: str) -> Dict[str, Any]:
    """
    Very lightweight NLP over the slice text.

    Returns:
      {
        "slice_summary": "...",   # always present
        "amount": "50 billion",   # if found
        "year_text": "2007",      # if found
        "has_war": bool,
        "has_map": bool,
        ...
      }

    You can extend this gradually. SceneEngine ignores unknown keys.
    """
    t = (slice_text or "").strip()
    t_lc = t.lower()
    concept: Dict[str, Any] = {}
    concept["slice_summary"] = _shorten(t)

    # Amount: "$50 billion", "2 trillion", "800b", etc.
    m1 = re.search(r"\$?\s*([\d\.]+\s*(?:billion|trillion))", t_lc)
    m2 = re.search(r"\$?\s*([\d]{2,3,4,}\s*(?:bn|trn))", t_lc)
    if m1:
        concept["amount"] = m1.group(1).upper()
    elif m2:
        concept["amount"] = m2.group(1).upper()

    # Year / era
    y = re.search(r"\b(1[89]\d{2}|20\d{2})\b", t)
    if y:
        concept["year_text"] = y.group(1)

    # Simple boolean flags the rule system might care about later
    concept["has_war"] = any(k in t_lc for k in ["war", "battle", "army", "military"])
    concept["has_map"] = any(k in t_lc for k in ["map", "border", "frontier", "territory"])
    concept["has_tax"] = "tax" in t_lc

    return concept