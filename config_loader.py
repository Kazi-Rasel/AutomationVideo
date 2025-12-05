#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config_loader.py — shared configuration loading utilities for SysVisuals.

This module centralizes loading of per-channel configuration JSONs such as:
  - metaphor_bank.json
  - pacing_policy.json
  - entity_style.json
  - prompt_style.json (optional)

It is designed so that global engines (scene_planner, scene_engine, etc.)
can remain mostly logic-only and delegate all filesystem / path decisions
to this module.

By default, it assumes the standard VideoAutomation layout:

  /Users/<user>/VideoAutomation/
      System/SysVisuals/...
      Channels/<ChannelName>/Config/

but it also supports environment overrides for maximum flexibility.

Environment overrides:
  - VA_ROOT                     → root of the VideoAutomation tree (optional)
  - VA_CHANNEL                  → default channel name (e.g. "CapitalChronicles")
  - VA_CHANNEL_CONFIG_DIR       → explicit directory for channel config
  - VA_METAPHOR_BANK_JSON       → explicit path to metaphor_bank.json
  - VA_PACING_POLICY_JSON       → explicit path to pacing_policy.json
  - VA_ENTITY_STYLE_JSON        → explicit path to entity_style.json
  - VA_PROMPT_STYLE_JSON        → explicit path to prompt_style.json
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional

import json
import os


# ---------------------------------------------------------------------------
# Global caches
# ---------------------------------------------------------------------------

_METAPHOR_BANK_CACHE: Dict[str, Dict[str, Any]] = {}
_PACING_POLICY_CACHE: Dict[str, Dict[str, Any]] = {}
_ENTITY_STYLES_CACHE: Dict[str, Dict[str, Any]] = {}
_PROMPT_STYLE_CACHE: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    """Read JSON from path, returning None on any error."""
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def get_va_root() -> Path:
    """
    Determine the root of the VideoAutomation tree.

    Priority:
      1) VA_ROOT env var
      2) Walk up from this file assuming standard layout
    """
    env_root = os.environ.get("VA_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    # Fallback: /System/SysVisuals/Shared/config_loader.py -> /VideoAutomation
    base = Path(__file__).resolve()
    # Shared -> SysVisuals -> System -> VideoAutomation
    return base.parents[3]


def get_default_channel() -> str:
    """
    Determine the default channel name.

    Priority:
      1) VA_CHANNEL env var
      2) "CapitalChronicles" as a sane default for your main channel
    """
    return os.environ.get("VA_CHANNEL", "CapitalChronicles")


def get_channel_config_dir(channel: Optional[str] = None) -> Path:
    """
    Compute the configuration directory for a given channel.

    Priority:
      1) VA_CHANNEL_CONFIG_DIR env var (if set)
      2) <VA_ROOT>/Channels/<channel>/Config
    """
    env_dir = os.environ.get("VA_CHANNEL_CONFIG_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()

    if channel is None:
        channel = get_default_channel()

    va_root = get_va_root()
    return va_root / "Channels" / channel / "Config"


def _resolve_config_path(
    filename: str,
    env_var: str,
    channel: Optional[str] = None,
) -> Path:
    """
    Determine the path to a specific config JSON, with env override.

    If env_var is set, that absolute or relative path is used.
    Otherwise, we look in the channel's Config directory.
    """
    override = os.environ.get(env_var)
    if override:
        return Path(override).expanduser().resolve()

    cfg_dir = get_channel_config_dir(channel)
    return cfg_dir / filename


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def load_metaphor_bank(channel: Optional[str] = None) -> Dict[str, Any]:
    """
    Load metaphor_bank.json for a given channel, with caching.

    Returns an empty dict if not found or invalid.
    """
    if channel is None:
        channel = get_default_channel()
    if channel in _METAPHOR_BANK_CACHE:
        return _METAPHOR_BANK_CACHE[channel]

    path = _resolve_config_path("metaphor_bank.json", "VA_METAPHOR_BANK_JSON", channel)
    data = _safe_read_json(path) or {}
    _METAPHOR_BANK_CACHE[channel] = data
    return data


def load_pacing_policy(channel: Optional[str] = None) -> Dict[str, Any]:
    """
    Load pacing_policy.json for a given channel, with caching.

    Returns an empty dict if not found or invalid.
    """
    if channel is None:
        channel = get_default_channel()
    if channel in _PACING_POLICY_CACHE:
        return _PACING_POLICY_CACHE[channel]

    path = _resolve_config_path("pacing_policy.json", "VA_PACING_POLICY_JSON", channel)
    data = _safe_read_json(path) or {}
    _PACING_POLICY_CACHE[channel] = data
    return data


def load_entity_styles(channel: Optional[str] = None) -> Dict[str, Any]:
    """
    Load entity_style.json for a given channel, with caching.

    Returns an empty dict if not found or invalid.
    """
    if channel is None:
        channel = get_default_channel()
    if channel in _ENTITY_STYLES_CACHE:
        return _ENTITY_STYLES_CACHE[channel]

    path = _resolve_config_path("entity_style.json", "VA_ENTITY_STYLE_JSON", channel)
    data = _safe_read_json(path) or {}
    _ENTITY_STYLES_CACHE[channel] = data
    return data


def load_prompt_style(channel: Optional[str] = None) -> Dict[str, Any]:
    """
    Load prompt_style.json for a given channel, with caching.

    Returns an empty dict if not found or invalid.
    """
    if channel is None:
        channel = get_default_channel()
    if channel in _PROMPT_STYLE_CACHE:
        return _PROMPT_STYLE_CACHE[channel]

    path = _resolve_config_path("prompt_style.json", "VA_PROMPT_STYLE_JSON", channel)
    data = _safe_read_json(path) or {}
    _PROMPT_STYLE_CACHE[channel] = data
    return data
