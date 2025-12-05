#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, re, datetime, json
from pathlib import Path
from typing import Optional, Tuple

HOME = Path.home()
VA_ROOT = Path(os.environ.get("VA_ROOT", str(HOME / "VideoAutomation")))

CHANNELS_DIR = VA_ROOT / "Channels"
SYSTEM_DIR   = VA_ROOT / "System"
SYSTTS_DIR   = SYSTEM_DIR / "SysTTS"
SYSVIS_DIR   = SYSTEM_DIR / "SysVisuals"

CONFIG_DIR   = SYSVIS_DIR / "Config"
TOOLS_DIR    = SYSVIS_DIR / "Tools"
SHARED_DIR   = SYSVIS_DIR / "Shared"
VLIB_DIR     = SYSVIS_DIR / "VisualLibrary"

CHANNEL_CFG   = VA_ROOT / "ChnlCnfg" / "channel_config.json"

def today_str() -> str:
    return datetime.date.today().isoformat()

_TAG_RE = re.compile(r'[_\-]([A-Z]{2})(?:[_\.]|$)')

def load_channel_map() -> Tuple[dict, str]:
    if CHANNEL_CFG.exists():
        data = json.loads(CHANNEL_CFG.read_text(encoding="utf-8"))
        umap = dict(data)
        untagged = umap.pop("UNTAGGED", "NoTagVdo")
        return umap, untagged
    return {}, "NoTagVdo"

def filename_to_channel(name: str) -> str:
    cmap, untag = load_channel_map(); m = _TAG_RE.search(name)
    return cmap[m.group(1)] if (m and m.group(1) in cmap) else untag

def channel_root(channel: str) -> Path:
    return CHANNELS_DIR / channel

def ensure_channel_basics(channel: str) -> Path:
    cr = channel_root(channel)
    for sub in (
        "Input/Script",
        "Input/Audio",
        "Input/Jobs",
        "Build/Prompts",
        "Build/Ready",
        "Build/Logs",
        "Build/Candidates",
        "Build/Proof",
        "Resolve/AutoTimeline",
        "Resolve/Exports",
        "UploadQueue",
        "FinalVideo",
    ):
        (cr / sub).mkdir(parents=True, exist_ok=True)
    return cr

def input_script_dir(channel: str, day: Optional[str] = None) -> Path:
    return ensure_channel_basics(channel) / "Input" / "Script" / (day or today_str())

def input_audio_dir(channel: str, day: Optional[str] = None) -> Path:
    return ensure_channel_basics(channel) / "Input" / "Audio" / (day or today_str())

def input_jobs_dir(channel: str, day: Optional[str] = None) -> Path:
    return ensure_channel_basics(channel) / "Input" / "Jobs" / (day or today_str())

def build_prompts_dir(channel: str, day: Optional[str] = None) -> Path:
    return ensure_channel_basics(channel) / "Build" / "Prompts" / (day or today_str())

def build_ready_day_dir(channel: str, day: Optional[str] = None) -> Path:
    return ensure_channel_basics(channel) / "Build" / "Ready" / (day or today_str())

def build_ready_job_root(channel: str, job_id: str, day: Optional[str] = None) -> Path:
    return build_ready_day_dir(channel, day) / job_id

def job_scenes_dir(channel: str, day: str, job_id: str) -> Path:
    return build_ready_job_root(channel, job_id, day) / "Scenes"

def job_manifest_path(channel: str, day: str, job_id: str) -> Path:
    return build_ready_job_root(channel, job_id, day) / "manifest.json"

def job_preview_path(channel: str, day: str, job_id: str) -> Path:
    return build_ready_job_root(channel, job_id, day) / f"{job_id}__mv_preview.mp4"

def build_logs_dir(channel: str, day: Optional[str] = None) -> Path:
    return ensure_channel_basics(channel) / "Build" / "Logs" / (day or today_str())

def resolve_exports_dir(channel: str, day: Optional[str] = None) -> Path:
    return ensure_channel_basics(channel) / "Resolve" / "Exports" / (day or today_str())

def final_video_dir(channel: str, day: Optional[str] = None) -> Path:
    return ensure_channel_basics(channel) / "FinalVideo" / (day or today_str())

def normalize_job_id(stem: str) -> str:
    s = re.sub(r'\s+', '_', stem.strip())
    s = re.sub(r'[\\/:*?"<>|]', '_', s)
    return s

def script_dest(channel: str, day: str, base_stem: str) -> Path:
    return input_script_dir(channel, day) / f"{base_stem}.txt"

def audio_dest(channel: str, day: str, job_id: str) -> Path:
    return input_audio_dir(channel, day) / f"{job_id}.wav"

def job_ticket_path(channel: str, day: str, job_id: str) -> Path:
    return input_jobs_dir(channel, day) / f"{job_id}.job.json"

def prompts_json_path(channel: str, day: str, job_id: str) -> Path:
    return build_prompts_dir(channel, day) / f"{job_id}_final.prompts.json"