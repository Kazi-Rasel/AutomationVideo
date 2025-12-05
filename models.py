#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class PromptSlice:
    t0: float = 0.0
    t1: float = 0.0
    text: str = ""

@dataclass
class Frame:
    # path to the image inside the job folder (e.g., "Scenes/scene_001.jpg")
    file: str

    # optional metadata present in your manifest
    text: str = ""
    t0: Optional[float] = None
    t1: Optional[float] = None
    index: Optional[int] = None  # ← accept 'index' from manifest

    def path(self) -> Path:
        return Path(self.file)

    def dur(self) -> Optional[float]:
        if self.t0 is not None and self.t1 is not None:
            return max(0.0, float(self.t1) - float(self.t0))
        return None