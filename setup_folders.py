#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup & Migration for VideoAutomation

- Creates the full folder layout under ~/VideoAutomation
- Idempotent: safe to run multiple times
- Options:
    --init            create root tree only (skip channel baselines)
    --dry-run         log actions without changing files
    --no-symlinks     (deprecated; no effect, kept for compatibility)
"""

import os, sys, json, argparse, datetime
from pathlib import Path

HOME = Path.home()
VA_ROOT = HOME / "VideoAutomation"

# New tree (authoritative)
CHANNELS = VA_ROOT / "Channels"
SYSTEM   = VA_ROOT / "System"
SYSTTS   = SYSTEM / "SysTTS"
SYSVIS   = SYSTEM / "SysVisuals"
VENVDIR  = SYSTEM / "venv"

# SysTTS contents
TTS_ENGINE  = SYSTTS / "EngineTTS"
TTS_CHUNKS  = SYSTTS / "InitialAudio"
SUBTITLES_DIR = SYSTTS / "Subtitles"  # NEW: dedicated subtitles/timing root

# SysVisuals contents
CONF_DIR    = SYSVIS / "Config"
ENG_DIR     = SYSVIS / "Engines"
ORCH_DIR    = SYSVIS / "Orchestrators"
TOOLS_DIR   = SYSVIS / "Tools"
SHARED_DIR  = SYSVIS / "Shared"
VLIB_DIR    = SYSVIS / "VisualLibrary"

def log(msg):
    print(msg, flush=True)

def safe_mkdir(p: Path, dry: bool):
    if dry:
        log(f"[dry] mkdir -p {p}")
        return
    p.mkdir(parents=True, exist_ok=True)

def create_tree(dry: bool):
    # root
    safe_mkdir(VA_ROOT, dry)
    safe_mkdir(CHANNELS, dry)

    # top-level files/folders
    safe_mkdir(VA_ROOT / "PutScript", dry)

    # System
    for p in [SYSTEM, SYSTTS, SYSVIS, VENVDIR]:
        safe_mkdir(p, dry)

    # SysTTS
    for p in [TTS_ENGINE, TTS_CHUNKS, SUBTITLES_DIR]:
        safe_mkdir(p, dry)
    for p in [SYSTTS / "recreate", SYSTTS / "using"]:
        safe_mkdir(p, dry)

    # SysVisuals
    for p in [CONF_DIR, ENG_DIR, ORCH_DIR, TOOLS_DIR, SHARED_DIR, VLIB_DIR]:
        safe_mkdir(p, dry)

    # Engines namespaces
    for p in [
        ENG_DIR / "image",
        ENG_DIR / "video",
        ENG_DIR / "importer",
    ]:
        safe_mkdir(p, dry)


def create_channel_baseline(dry: bool, notes: list):
    """
    Ensure each configured channel has baseline subfolders.
    Uses channel_config.json if present; otherwise creates none (user can run later).
    """
    cfg_path = VA_ROOT / "ChnlCnfg" / "channel_config.json"
    if not cfg_path.exists():
        return
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return
    channels = set(cfg.values()) - {"UNTAGGED"}
    channels.add(cfg.get("UNTAGGED", "NoTagVdo"))
    for ch in channels:
        chroot = CHANNELS / ch
        # Input
        for sub in ["Input/Script", "Input/Audio", "Input/Jobs"]:
            safe_mkdir(chroot / sub, dry)
        # Build
        for sub in ["Build/Prompts", "Build/Ready", "Build/Logs", "Build/Candidates", "Build/Proof"]:
            safe_mkdir(chroot / sub, dry)
        # Resolve / Upload
        for sub in ["Resolve/AutoTimeline", "Resolve/Exports", "UploadQueue"]:
            safe_mkdir(chroot / sub, dry)
        # FinalVideo
        safe_mkdir(chroot / "FinalVideo", dry)
        # Channel-level Config (for things like prompt_style.json)
        cfg_dir = chroot / "Config"
        safe_mkdir(cfg_dir, dry)
        style_path = cfg_dir / "prompt_style.json"
        default_style_path = CONF_DIR / "default_prompt_style.json"

        if not dry and not style_path.exists():
            try:
                if default_style_path.exists():
                    # Copy full structured template
                    style_path.write_text(
                        default_style_path.read_text(encoding="utf-8"),
                        encoding="utf-8",
                    )
                    notes.append(f"created default prompt_style.json for {ch} from default_prompt_style.json")
                else:
                    # Fallback minimal structure (safety only)
                    minimal = {
                        "global": {
                            "prompt_prefix": "",
                            "prompt_suffix": "",
                            "negative_prompt": "",
                        },
                        "rules": [],
                        "fallback_template": "GENERIC_SCENE",
                        "templates": {
                            "GENERIC_SCENE": "{slice_summary}",
                        },
                    }
                    style_path.write_text(json.dumps(minimal, indent=2), encoding="utf-8")
                    notes.append(f"created minimal fallback prompt_style.json for {ch}")
            except Exception:
                # fail silently if we can't write, but don't break setup
                pass

        # Auto-generate a per-channel policy stub if not present
        policy_path = cfg_dir / "channel_policy.py"
        if not dry and not policy_path.exists():
            try:
                stub = (
                    f"# Auto-generated channel policy stub for {ch}\n"
                    "# ---------------------------------------------------------------------------\n"
                    "# This file contains OPTIONAL per-channel customization hooks for the visual\n"
                    "# pipeline. The global engine (prompt_style.py, scene_orchestrator.py,\n"
                    "# image_pipeline.py) remains completely generic. It will call these functions\n"
                    "# ONLY if they exist.\n"
                    "#\n"
                    "# This means: you can safely edit this file to tune a channel's behavior\n"
                    "# without touching ANY global code. Every channel can have its own rules.\n"
                    "#\n"
                    "# These hooks let you influence how the LLM writes prompts, how strict CLIP\n"
                    "# should be for reuse, and what extra hints should be added to a scene.\n"
                    "#\n"
                    "# IMPORTANT: All functions below are OPTIONAL. If you don't modify them, the\n"
                    "# system behaves with default global logic. Nothing breaks.\n"
                    "# ---------------------------------------------------------------------------\n\n"
                    "from typing import Dict, Any\n\n"
                    "def refine_llm_scene_context(ctx: Dict[str, Any]) -> Dict[str, Any]:\n"
                    "    \"\"\"\n"
                    "    OPTIONAL: Modify the LLM context *before* the prompt is created.\n"
                    "\n"
                    "    The incoming dict `ctx` may contain:\n"
                    "        - job_profile: The global LLM semantic analysis of the whole script.\n"
                    "        - channel_style: Visual style block from prompt_style.json.\n"
                    "        - channel_llm: LLM instructions block from prompt_style.json.\n"
                    "        - topic: Auto-routed category for this segment (from categories.json).\n"
                    "        - role: Preliminary role guess for this segment.\n"
                    "        - text: Raw script text for this segment.\n"
                    "\n"
                    "    You may modify ANY of these to further shape LLM behavior.\n"
                    "    For example, for finance-focused channels you might:\n"
                    "        - enforce more aggressive topic labels during crashes\n"
                    "        - inject additional stylistic hints\n"
                    "        - override topic if the LLM misclassified something\n"
                    "\n"
                    "    Return the modified `ctx` dict.\n"
                    "    \"\"\"\n"
                    "    return ctx\n\n"
                    "def refine_scene_prompt_text(scene: str, topic: str, role: str) -> str:\n"
                    "    \"\"\"\n"
                    "    OPTIONAL: Post-process the LLM-generated `scene` text *before* sending it\n"
                    "    to the image engine.\n"
                    "\n"
                    "    Good use cases:\n"
                    "        - Ensure FINANCE scenes always contain a visible STOCK CHART mention.\n"
                    "        - Ensure WAR scenes always mention troops, battlefield geometry, etc.\n"
                    "        - Prevent portrait-style outputs on channels that dislike them.\n"
                    "\n"
                    "    Example modification:\n"
                    "        if topic == 'STOCK_MARKET' and 'chart' not in scene.lower():\n"
                    "            scene += ' A screen clearly shows a real stock chart with visible movement.'\n"
                    "\n"
                    "    Return the modified scene text.\n"
                    "    \"\"\"\n"
                    "    return scene\n\n"
                    "def topic_clip_threshold(topic: str) -> float:\n"
                    "    \"\"\"\n"
                    "    OPTIONAL: Adjust the CLIP similarity threshold used for reuse decisions\n"
                    "    *only for this channel*.\n"
                    "\n"
                    "    Higher values → stricter matching → fewer wrong reuses.\n"
                    "    Lower values  → looser matching → more aggressive reuse.\n"
                    "\n"
                    "    Typical usage:\n"
                    "        if topic in ('FINANCE_CHARTS', 'STOCK_MARKET'):\n"
                    "            return 0.32     # demand very strong alignment\n"
                    "        if topic.startswith('HISTORY_'):\n"
                    "            return 0.28     # slightly stricter for history visuals\n"
                    "        return 0.25         # global default\n"
                    "\n"
                    "    The global system will call this ONLY IF it exists.\n"
                    "    \"\"\"\n"
                    "    return 0.25\n"
                )
                policy_path.write_text(stub, encoding="utf-8")
                notes.append(f"created channel_policy.py stub for {ch}")
            except Exception:
                # fail silently if we can't write, but don't break setup
                pass
    notes.append("Channel baselines ensured")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", action="store_true", help="create root tree only (skip channel baselines)")
    ap.add_argument("--dry-run", action="store_true", help="log actions without changing files")
    ap.add_argument("--no-symlinks", action="store_true", help="deprecated; no effect (kept for compatibility)")
    args = ap.parse_args()

    dry = args.dry_run
    init_only = args.init

    notes = []
    log("▶ VideoAutomation setup")
    log(f"Root: {VA_ROOT}")

    # 1) Create tree
    create_tree(dry)

    # 2) Create channel baselines (unless --init)
    if not init_only:
        create_channel_baseline(dry, notes)

    # 3) Setup finished
    log("✅ Setup finished")

if __name__ == "__main__":
    main()