#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import time, datetime, sys, os
from pathlib import Path

VA_ROOT = Path(os.environ.get("VA_ROOT", str(Path.home() / "VideoAutomation")))
SYSVIS = VA_ROOT / "System" / "SysVisuals"
SHARED = SYSVIS / "Shared"

SYS     = VA_ROOT / "System"
LOG_DIR = SYS / "Logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "resolve_engine.log"

if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))

import paths

def log(m: str) -> None:
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[RESOLVE {timestamp}] {m}"
    # print to terminal
    print(line, flush=True)
    # append to central log file
    try:
        with LOG_FILE.open('a', encoding='utf-8') as f:
            f.write(line + '\n')
    except Exception:
        pass

def import_to_resolve(channel: str, day: str, job_id: str, job_root: Path, video: Path, srt: Path | None) -> None:
    log(f"▶ import {channel}/{day}/{job_id}")
    try:
        import DaVinciResolveScript as dvr
    except Exception as e:
        log(f"⚠ Resolve scripting unavailable: {e}")
        return

    resolve = dvr.scriptapp("Resolve")
    if not resolve:
        log("⚠ Resolve handle is None"); return

    pm = resolve.GetProjectManager()
    proj = pm.GetCurrentProject() or pm.CreateProject(channel)
    log(f"[project] 🎬 {proj.GetName()}")
    mp  = proj.GetMediaPool()
    root= mp.GetRootFolder()

    bin_name = f"AutoImports-{day}"
    bins = {b.GetName(): b for b in root.GetSubFolderList()}
    bin_folder = bins.get(bin_name) or mp.AddSubFolder(root, bin_name)
    mp.SetCurrentFolder(bin_folder)

    mp.ImportMedia([str(video)])
    log(f"[pool] ✅ {video.name} → {bin_name}")
    tl = mp.CreateTimelineFromClips(job_id, bin_folder.GetClipList())
    if tl:
        proj.SetCurrentTimeline(tl)
        log(f"[timeline] ✅ created {job_id}")
        if srt and srt.exists():
            try:
                tl.ImportSubtitles(str(srt))
                log(f"[subs] ✅ {srt.name}")
            except Exception:
                time.sleep(0.3)
                try:
                    tl.ImportSubtitles(str(srt))
                    log(f"[subs] ✅ {srt.name} (retry)")
                except Exception:
                    pass

    export_dir = paths.final_video_dir(channel, day)
    export_dir.mkdir(parents=True, exist_ok=True)
    proj.SetRenderSettings({"TargetDir": str(export_dir)})
    log(f"[export] ✅ path → {export_dir}")