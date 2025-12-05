#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified AI Image Engine Interface & Selector

This file contains:
  1. A provider-agnostic interface (AIImageEngine)
  2. A selector that chooses which implementation to use

To switch AI providers:
  - Replace the import at the bottom:
        from fal_ai_engine import FalAIEngine as AIImageEngine
  - Or use:
        from google_ai_engine import GoogleAIEngine as AIImageEngine
        from flux_ai_engine import FluxAIEngine as AIImageEngine
        ...
Nothing else in the entire system needs to change.
"""

from pathlib import Path
import json


class AIImageEngine:
    """
    Generic interface all AI engines must implement.

    Optional semantic fields (for future engines):
        - negative: Optional[str] = None
            Negative prompt, describing what to avoid in the image.
        - topic: Optional[str] = None
            The main subject or theme.
        - role: Optional[str] = None
            The intended role or character for the image.
        - concept_id: Optional[str] = None
            A provider-specific or dataset concept identifier.

    These fields are optional and harmless for engines that do not use them.
    """

    def generate(
        self,
        prompt,
        out_path,
        seed=None,
        *,
        negative=None,
        topic=None,
        role=None,
        concept_id=None,
        **kwargs
    ):
        """
        Generate an image given a prompt.
        Optional semantic fields:
            negative: negative prompt (what to avoid)
            topic: main subject or theme
            role: intended role/character
            concept_id: provider/dataset concept identifier
        These are ignored by engines that do not use them.
        """
        _ = (negative, topic, role, concept_id)
        raise NotImplementedError("generate() not implemented.")

    def generate_batch(
        self,
        prompts,
        out_dir,
        seed_base=None,
        *,
        negative=None,
        topic=None,
        role=None,
        concept_id=None,
        **kwargs
    ):
        """
        Generate a batch of images given a list of prompts.
        Optional semantic fields:
            negative: negative prompt (what to avoid)
            topic: main subject or theme
            role: intended role/character
            concept_id: provider/dataset concept identifier
        These are ignored by engines that do not use them.
        """
        _ = (negative, topic, role, concept_id)
        raise NotImplementedError("generate_batch() not implemented.")


VA_ROOT = Path(__file__).resolve().parents[4]


def _sync_pipeline_image_engine(name: str) -> None:
    """Auto-sync pipeline.json image_engine with the active provider name.

    This keeps Config/pipeline.json in agreement with the engine selected
    below, without requiring manual edits.
    """
    cfg_path = VA_ROOT / "System" / "SysVisuals" / "Config" / "pipeline.json"
    try:
        if not cfg_path.exists():
            return
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        current = str(data.get("image_engine", "")).strip()
        if current != name:
            data["image_engine"] = name
            cfg_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        # Never fail pipeline startup because of config sync issues
        return


# ---------------------------------------------------------------------
# ACTIVE ENGINE SELECTION (comment/uncomment ONE line)
# ---------------------------------------------------------------------

# Fal AI Engine
# ENGINE_NAME = "fal"; from fal_ai_engine import FalAIEngine as AIImageEngine

# Google Imagen 4 Engine
ENGINE_NAME = "google"; from imagen4_ai_engine import Imagen4AIEngine as AIImageEngine

# Flux Engine
# ENGINE_NAME = "flux"; from flux_ai_engine import FluxAIEngine as AIImageEngine

# SDXL Engine
# ENGINE_NAME = "sdxl"; from sdxl_ai_engine import SDXLAIEngine as AIImageEngine

# ---------------------------------------------------------------------
# ACTIVATE THE ONE LINE YOU WANT
# ---------------------------------------------------------------------
_sync_pipeline_image_engine(ENGINE_NAME)
