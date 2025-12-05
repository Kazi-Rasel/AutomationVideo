#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, time, json, shutil
import datetime
from pathlib import Path
from typing import Iterable

# ========= Project root =========
HOME    = Path.home()
VA_ROOT = Path(os.environ.get("VA_ROOT", str(HOME / "VideoAutomation")))
CHANNELS_DIR = VA_ROOT / "Channels"
SYSVIS_DIR   = VA_ROOT / "System" / "SysVisuals"
SYSTTS_DIR   = VA_ROOT / "System" / "SysTTS"
SUBTITLES_DIR = SYSTTS_DIR / "Subtitles"   # where *_karaoke.json live

CHANNEL_CFG   = VA_ROOT / "ChnlCnfg" / "channel_config.json"

SYS     = VA_ROOT / "System"
LOG_DIR = SYS / "Logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "weekly_cleanup.log"


def _log(msg: str) -> None:
    """Log weekly cleanup events to terminal and central log file."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[CLEANUP {timestamp}] {msg}"
    # print to terminal
    print(line, flush=True)
    # append to central log file
    try:
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

def load_channels_list() -> list[str]:
    """
    Load channel names from project config:
      - ChnlCnfg/channel_config.json (project root)
      - If missing/invalid, fall back to only UNTAGGED -> "NoTagVdo".
    Returns a list of channel folder names (e.g. "CapitalChronicles").
    """
    data = None
    try:
        if CHANNEL_CFG.exists():
            data = json.loads(CHANNEL_CFG.read_text(encoding="utf-8"))
    except Exception:
        data = None

    if not isinstance(data, dict):
        # fallback defaults: only provide UNTAGGED if no config exists
        data = {"UNTAGGED": "NoTagVdo"}

    chans = [v for k, v in data.items() if k != "UNTAGGED"]
    chans.append(data.get("UNTAGGED", "NoTagVdo"))

    seen, out = set(), []
    for c in chans:
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out

# ========= policy =========
# All cleanup actions must never touch anything newer than 7 days.
KEEP_LAST    = 12   # keep newest K completed projects per channel
TTL_DAYS     = 7    # archive/delete completed projects older than this many days
ARCHIVE      = True # True = move to Build/Archive/<pid>; False = delete
LOG_TTL_DAYS = 7    # delete Build/Logs files older than this many days

EXPORT_EXTS = (".mp4", ".mov", ".mkv", ".m4v", ".avi", ".wmv", ".webm")

def safe_move_tree(src: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():
        try:
            shutil.move(str(p), str(dst / p.name))
        except Exception:
            pass

# --------- helpers for current & legacy layouts ---------
def job_dirs(ready: Path) -> Iterable[Path]:
    """Yield job folders (current layout)."""
    if not ready.exists():
        return []
    return [p for p in ready.iterdir() if p.is_dir()]

def has_export(exp_dir: Path, pid: str) -> bool:
    if not exp_dir.exists():
        return False
    for f in exp_dir.iterdir():
        if f.is_file() and f.suffix.lower() in EXPORT_EXTS and f.name.startswith(pid):
            return True
    return False

def mtime_tree(p: Path) -> float:
    """Latest mtime from all files under p (folder) or just file mtime."""
    try:
        if p.is_dir():
            mt = 0.0
            for q in p.rglob("*"):
                try:
                    mt = max(mt, q.stat().st_mtime)
                except Exception:
                    pass
            return mt or p.stat().st_mtime
        return p.stat().st_mtime
    except Exception:
        return time.time()

# --------- cleaning routines ---------
def purge_try_frames(chroot: Path):
    """Remove scene_*.try*.jpg retry frames in all job folders."""
    ready = chroot / "Build" / "Ready"
    if not ready.exists():
        return
    # current layout
    for d in job_dirs(ready):
        for p in d.glob("scene_*.try*.jpg"):
            try:
                p.unlink()
                # print(f"🧹 removed {p}")
            except Exception:
                pass
    # legacy safety
    for p in ready.glob("*.try*.jpg"):
        try:
            p.unlink()
        except Exception:
            pass

def cleanup_logs(chroot: Path, now: float):
    logs = chroot / "Build" / "Logs"
    if not logs.exists():
        return
    ttl = LOG_TTL_DAYS * 86400.0
    for p in logs.glob("*"):
        try:
            if p.is_file() and (now - p.stat().st_mtime) >= ttl:
                p.unlink()
                # print(f"🧹 old log removed: {p}")
        except Exception:
            pass

def cleanup_subtitles(now: float):
    """
    Clean SysTTS/Subtitles timing JSON older than TTL_DAYS.
    Never delete timing created within the last TTL_DAYS (7 days).
    """
    if not SUBTITLES_DIR.exists():
        return
    ttl = TTL_DAYS * 86400.0
    for p in SUBTITLES_DIR.rglob("*.json"):
        try:
            age = now - p.stat().st_mtime
            if age >= ttl:
                try:
                    p.unlink()
                    # print(f"🧹 old subtitle timing removed: {p}")
                except Exception:
                    pass
        except Exception:
            pass

def archive_or_delete_dir(chroot: Path, job_dir: Path):
    """Archive/delete entire job folder (current layout)."""
    pid = job_dir.name
    if ARCHIVE:
        dst = chroot / "Build" / "Archive" / pid
        dst.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(job_dir), str(dst))
            _log(f"📦 archived  {chroot.name}/{pid} → Build/Archive/{pid}")
        except Exception:
            pass
    else:
        try:
            shutil.rmtree(job_dir, ignore_errors=True)
            _log(f"🗑️ deleted   {chroot.name}/{pid}")
        except Exception:
            pass

def main():
    now = time.time()

    # Global cleanup for subtitle timings (SysTTS/Subtitles)
    cleanup_subtitles(now)

    for ch in load_channels_list():
        chroot = CHANNELS_DIR / ch
        ready  = chroot / "Build" / "Ready"
        exports= chroot / "Resolve" / "Exports"
        jobs   = chroot / "Input" / "Jobs"
        if not ready.exists():
            continue

        # --- always do light housekeeping first
        purge_try_frames(chroot)        # remove retries
        cleanup_logs(chroot, now)       # old logs

        # --- CURRENT LAYOUT: job folders
        job_folders = [d for d in job_dirs(ready)]
        # completion (no job ticket + export exists)
        completed_dirs = []
        for d in job_folders:
            pid = d.name
            if (jobs / f"{pid}.job.json").exists():
                continue
            if has_export(exports, pid):
                completed_dirs.append(d)

        # sort newest first by mtime of folder content
        completed_dirs.sort(key=lambda p: mtime_tree(p), reverse=True)

        # keep newest K, archive/delete others older than TTL
        keep_dirs = set(completed_dirs[:KEEP_LAST])
        for d in completed_dirs[KEEP_LAST:]:
            age_days = (now - mtime_tree(d)) / 86400.0
            if age_days >= TTL_DAYS:
                archive_or_delete_dir(chroot, d)

    _log("✔ weekly cleanup done.")

if __name__ == "__main__":
    main()