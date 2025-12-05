#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""clip_utils.py

Shared CLIP utilities for the VideoAutomation project.

This module provides a provider-agnostic API around an image/text encoder.
It is used by both the online AI image engines (Fal/Imagen/Flux/etc.) and
by offline tools (manual VisualLibrary import).

Responsibilities:
  - Manage a single global CLIP model instance (lazy-loaded)
  - Encode text -> embedding (list[float])
  - Encode images -> embedding (list[float])
  - Compute cosine similarity between embeddings
  - Provide a CategoryIndex abstraction over VisualLibrary/categories.json

No provider-specific code lives here; this is generic infrastructure.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import torch

try:  # open_clip is provided in the venv per your existing dependencies
    import open_clip
except Exception as e:  # pragma: no cover - we don't want to explode if missing
    open_clip = None

# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------

# NOTE: we use a global lazy-initialised model to avoid reloading per call.
# This is safe in a long-running process (e.g. auto_on_job) and keeps
# CPU/GPU memory usage under control.

_MODEL = None
_PREPROCESS = None
_DEVICE = None
_MODEL_NAME = None


def _detect_device() -> str:
    """Choose best available device: cuda, mps, or cpu.

    This mirrors typical PyTorch patterns:
      - If CUDA is available, prefer 'cuda'.
      - Else if Apple's Metal backend is available, use 'mps'.
      - Else fall back to 'cpu'.
    """
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_model() -> None:
    """Lazy-load the CLIP model and preprocess pipeline.

    Uses the same open_clip version you already installed in the venv.
    The exact model name / pretrained weights can be adjusted here later.
    """
    global _MODEL, _PREPROCESS, _DEVICE
    if _MODEL is not None:
        return

    if open_clip is None:
        raise RuntimeError("open_clip is not available. Ensure 'open-clip-torch' is installed.")

    _DEVICE = _detect_device()
    model_name = os.environ.get("CLIP_MODEL_NAME", "ViT-B-32")
    pretrained = os.environ.get("CLIP_PRETRAINED", "laion2b_s34b_b79k")

    global _MODEL_NAME
    _MODEL_NAME = model_name

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=_DEVICE,
    )
    model.eval()
    _MODEL = model
    _PREPROCESS = preprocess


def encode_text(text: str) -> Optional[List[float]]:
    """Encode a text string into a normalised CLIP embedding.

    Returns a list[float] or None if encoding failed.
    """
    if not text:
        return None

    _load_model()
    assert _MODEL is not None and _PREPROCESS is not None and _DEVICE is not None

    with torch.no_grad():
        # open_clip uses its own tokenizer
        tokenizer = open_clip.get_tokenizer(_MODEL_NAME)
        tokens = tokenizer([text]).to(_DEVICE)
        feats = _MODEL.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats[0].cpu().tolist()


def encode_image(path: Path) -> Optional[List[float]]:
    """Encode an image file into a normalised CLIP embedding.

    Returns a list[float] or None on failure.
    """
    if not path.is_file():
        return None

    _load_model()
    assert _MODEL is not None and _PREPROCESS is not None and _DEVICE is not None

    from PIL import Image  # local import to avoid hard dependency if unused

    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return None

    with torch.no_grad():
        img_t = _PREPROCESS(img).unsqueeze(0).to(_DEVICE)
        feats = _MODEL.encode_image(img_t)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats[0].cpu().tolist()


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute cosine similarity between two 1-D vectors.

    Returns a float in [-1, 1]. If shapes mismatch or zero norms occur,
    this returns 0.0.
    """
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        return 0.0

    # simple explicit dot product to avoid extra tensor allocations
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dx = float(x)
        dy = float(y)
        dot += dx * dy
        norm_a += dx * dx
        norm_b += dy * dy

    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0

    return dot / math.sqrt(norm_a * norm_b)


# ---------------------------------------------------------------------------
# CategoryIndex abstraction
# ---------------------------------------------------------------------------

@dataclass
class Category:
    id: str
    prompt: str
    embedding: List[float]


class CategoryIndex:
    """Helper around VisualLibrary/categories.json.

    Allows querying top-k labels for a given text or image embedding.
    """

    def __init__(self, categories: List[Category]):
        self._categories = categories

    @classmethod
    def from_config(cls, categories_path: Path) -> "CategoryIndex":
        """Load the category config file and pre-compute embeddings.

        categories.json format:
          [
            {"id": "MARKET_CRASH", "prompt": "stock market crash red candlesticks panic investors recession"},
            ...
          ]
        """
        if not categories_path.is_file():
            raise FileNotFoundError(f"categories.json not found at {categories_path}")

        with categories_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        cats: List[Category] = []
        for item in raw:
            cid = str(item.get("id", "")).strip()
            prompt = str(item.get("prompt", "")).strip()
            if not cid or not prompt:
                continue
            emb = encode_text(prompt)
            if not emb:
                continue
            cats.append(Category(id=cid, embedding=emb, prompt=prompt))

        return cls(cats)

    def labels_for_text_vec(
        self,
        text_vec: Sequence[float],
        top_k: int = 8,
        min_sim: float = 0.25,
    ) -> List[tuple[str, float]]:
        """Return a list of (label_id, similarity) pairs.

        - top_k: maximum number of labels to return
        - min_sim: minimum cosine similarity threshold
        """
        if not self._categories:
            return []
        if not text_vec:
            return []

        sims = []
        for cat in self._categories:
            s = cosine_similarity(text_vec, cat.embedding)
            if s >= min_sim:
                sims.append((cat.id, s))

        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:top_k]

    def best_label_for_image_vec(self, img_vec: Sequence[float]) -> Optional[str]:
        """Return the best label ID for an image embedding, or None.

        Currently we reuse the same label prompts and compare the image
        embedding to category text embeddings.
        """
        if not self._categories:
            return None
        if not img_vec:
            return None

        best_label: Optional[str] = None
        best_sim = 0.0
        for cat in self._categories:
            s = cosine_similarity(img_vec, cat.embedding)
            if s > best_sim:
                best_sim = s
                best_label = cat.id

        # Optionally apply a minimum similarity threshold here if desired
        return best_label
