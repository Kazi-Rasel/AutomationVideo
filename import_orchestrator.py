#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os, sys, errno, datetime, time
from pathlib import Path
from typing import Optional, Tuple


# ---------- logging ----------
__ORCH_FILE__ = __file__
def _log(msg: str) -> None:
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[IMPORT {timestamp}] {msg}"
    # print to terminal
    print(line, flush=True)
    # append to central log file
    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(line + "\n")
    except Exception:
        pass

#
# ---------- paths/bootstrap ----------
VA_ROOT = Path(os.environ.get("VA_ROOT", str(Path.home() / "VideoAutomation")))
SYS     = VA_ROOT / "System"
SYSVIS  = SYS / "SysVisuals"
SHARED  = SYSVIS / "Shared"

# --- Central log directory ---
LOG_DIR = SYS / "Logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "import_orchestrator.log"
if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))
try:
    import paths  # Shared/paths.py
except Exception:
    paths = None  # tolerant: operate without Shared helpers

#
# LLM sentinel: if present, we should not start/continue LLM-driven jobs
LLM_SENTINEL = SYSVIS / "LLM" / "LLM_DISABLED"

def _llm_disabled() -> bool:
    """Return True if the LLM_DISABLED sentinel exists.

    When present, the import orchestrator should not start or continue
    Resolve import for this job. This mirrors watcher.py, auto_on_job.py,
    scene_orchestrator.py, and video_orchestrator.py to prevent importing
    partial/corrupt outputs when the LLM layer or pipeline is deliberately paused.
    """
    try:
        if LLM_SENTINEL.exists():
            try:
                reason = LLM_SENTINEL.read_text(encoding="utf-8", errors="ignore").strip()
            except Exception:
                reason = ""
            if not reason:
                reason = "LLM disabled by sentinel file."
            _log(f"🚫 LLM_DISABLED sentinel present at {LLM_SENTINEL}")
            _log(f"🚫 Reason: {reason}")
            return True
    except Exception as e:
        _log(f"⚠ Failed to inspect LLM sentinel: {e}")
    return False

def _channel_root(channel: str) -> Path:
    if paths and hasattr(paths, "channel_root"):
        return paths.channel_root(channel)  # type: ignore[attr-defined]
    return VA_ROOT / "Channels" / channel

def _job_root(channel: str, day: str, job_id: str) -> Path:
    if paths and hasattr(paths, "job_manifest_path"):
        return paths.job_manifest_path(channel, day, job_id).parent  # type: ignore[attr-defined]
    return _channel_root(channel) / "Build" / "Ready" / day / job_id

def _preview_mp4(channel: str, day: str, job_id: str) -> Path:
    r = _job_root(channel, day, job_id)
    a = r / f"{job_id}__vastudio.mp4"
    b = r / f"{job_id}__mv_preview.mp4"
    return a if a.exists() else b

def _split_job_root(job_root: Path) -> Optional[Tuple[str, str, str]]:
    """Parse …/Channels/<ch>/Build/Ready/<day>/<job_id> → (ch, day, job_id)."""
    try:
        parts = job_root.resolve().parts
    except Exception:
        parts = job_root.parts
    try:
        i = parts.index("Channels")
        ch = parts[i + 1]
        j = i + 2
        while j < len(parts) and parts[j] != "Build":
            j += 1
        if j >= len(parts) - 3:
            return None
        if parts[j] != "Build" or parts[j + 1] != "Ready":
            return None
        day, job = parts[j + 2], parts[j + 3]
        return ch, day, job
    except Exception:
        return None

def _as_str(x) -> Optional[str]:
    return None if x is None else str(x)

def _callable(obj, name: str) -> bool:
    return (obj is not None) and hasattr(obj, name) and callable(getattr(obj, name))

def _sanitize_project_name(name: str) -> str:
    clean = "".join(ch if (ch.isalnum() or ch in "_- ") else " " for ch in (name or ""))
    s = clean.strip()
    return s or "VideoAutomation"

# ---------- Resolve bootstrap ----------
RESOLVE_API_DIR = os.environ.get(
    "RESOLVE_SCRIPT_API",
    "/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting",
)
RESOLVE_MODS = os.path.join(RESOLVE_API_DIR, "Modules")
if RESOLVE_MODS not in sys.path:
    sys.path.append(RESOLVE_MODS)
try:
    import DaVinciResolveScript as bmd  # type: ignore
except Exception:
    bmd = None  # tolerant “no Resolve” mode

def _open_or_switch_to_project(target_name: str):
    """
    Return (resolve, pm, ms, project) with project named target_name loaded/created.
    If another project is open, attempt a save then switch to avoid UI prompts.
    """
    if not bmd:
        _log("ok (no Resolve): scripting module not present")
        return None, None, None, None
    try:
        resolve = bmd.scriptapp("Resolve")
    except Exception as e:
        _log(f"ok (no Resolve): {e}")
        return None, None, None, None
    if not resolve:
        _log("ok (no Resolve): scriptapp returned None")
        return None, None, None, None

    pm = resolve.GetProjectManager() if _callable(resolve, "GetProjectManager") else None
    ms = resolve.GetMediaStorage()  if _callable(resolve, "GetMediaStorage")  else None
    if not pm:
        _log("⚠ Resolve ProjectManager unavailable")
        return resolve, None, ms, None

    name = _sanitize_project_name(target_name)

    # If a different project is open, save it so Resolve won’t prompt when switching.
    cur = pm.GetCurrentProject() if _callable(pm, "GetCurrentProject") else None
    try:
        if cur and _callable(cur, "GetName") and cur.GetName() != name:
            cur_name = cur.GetName()
            if _callable(pm, "SaveProject"):
                if not (isinstance(cur_name, str) and cur_name.lower().startswith("untitled")):
                    pm.SaveProject()
    except Exception:
        pass

    proj = None
    # Try load
    if _callable(pm, "LoadProject"):
        try:
            loaded = pm.LoadProject(name)
            if loaded:
                proj = loaded
                _log(f"• Loaded project: {name}")
        except Exception:
            proj = None
    # Try create
    if not proj and _callable(pm, "CreateProject"):
        try:
            created = pm.CreateProject(name)
            if created:
                proj = created
                _log(f"• Created new project: {name}")
        except Exception as e:
            _log(f"⚠ CreateProject('{name}') failed: {e}")
            proj = None

    if resolve and proj:
        _log("✅ DaVinci Resolve connected.")
    return resolve, pm, ms, proj

def _ensure_timeline(project, mpool, name: str):
    """
    Return a timeline named `name`, creating it if necessary.
    Prefer MediaPool.CreateEmptyTimeline, fallback to project.CreateTimeline.
    """
    try:
        if _callable(project, "GetTimelineCount") and _callable(project, "GetTimelineByIndex"):
            n = project.GetTimelineCount() or 0
            for i in range(1, n + 1):
                tl = project.GetTimelineByIndex(i)
                if tl and _callable(tl, "GetName") and tl.GetName() == name:
                    if _callable(project, "SetCurrentTimeline"):
                        project.SetCurrentTimeline(tl)
                    return tl

        if mpool and _callable(mpool, "CreateEmptyTimeline"):
            tl = mpool.CreateEmptyTimeline(name)
            if tl:
                if _callable(project, "SetCurrentTimeline"):
                    project.SetCurrentTimeline(tl)
                _log(f"[timeline] 🎬 created {name}")
                return tl

        if _callable(project, "CreateTimeline"):
            tl = project.CreateTimeline(name)
            if tl:
                if _callable(project, "SetCurrentTimeline"):
                    project.SetCurrentTimeline(tl)
                _log(f"[timeline] 🎬 created {name} (fallback)")
                return tl
    except Exception as e:
        _log(f"⚠ ensure_timeline: {e}")
    return None

# ---------- main entry with atomic lock ----------
def run_import(*args, **kwargs) -> None:
    """
    Accepts any of:
      run_import(channel, day, job_id)
      run_import(job_root_path)
      run_import(job_root_path, channel_hint)  # channel_hint ignored
    """
    # Hard guard: if the LLM sentinel is present, do not start this import job.
    if _llm_disabled():
        _log("🛑 Aborting import due to LLM_DISABLED sentinel.")
        return
    # --- normalize inputs ---
    channel: Optional[str] = None
    day: Optional[str]     = None
    job_id: Optional[str]  = None

    if len(args) >= 3 and all(args[:3]):
        channel, day, job_id = _as_str(args[0]), _as_str(args[1]), _as_str(args[2])
    elif len(args) == 1:
        p0 = Path(args[0]) if isinstance(args[0], (str, Path)) else None
        parsed = _split_job_root(p0) if p0 else None
        if parsed:
            channel, day, job_id = parsed
    elif len(args) == 2:
        p0 = Path(args[0]) if isinstance(args[0], (str, Path)) else None
        parsed = _split_job_root(p0) if p0 else None
        if parsed:
            channel, day, job_id = parsed

    channel = channel or _as_str(kwargs.get("channel"))
    day     = day     or _as_str(kwargs.get("day"))
    job_id  = job_id  or _as_str(kwargs.get("job_id"))

    if not (channel and day and job_id):
        _log(f"skip (bad signature): channel={channel!r} day={day!r} job_id={job_id!r}")
        return

    _log(f"▶ import {channel}/{day}/{job_id}")

    job_root = _job_root(channel, day, job_id)
    mp4 = _preview_mp4(channel, day, job_id)
    if not mp4.exists():
        _log(f"skip (missing preview): {channel}/{day}/{job_id} expected {mp4.name}")
        return

    # --- ATOMIC lock to avoid duplicate imports ---
    importing = job_root / ".importing"
    imported  = job_root / ".imported"
    try:
        fd = os.open(str(importing), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except OSError as e:
        if e.errno == errno.EEXIST:
            _log(f"skip (already importing) → {channel}/{day}/{job_id}")
            return
        else:
            _log(f"⚠ lock create failed: {e}")
            return

    if imported.exists():
        _log(f"skip (already imported) → {channel}/{day}/{job_id}")
        try:
            importing.unlink(missing_ok=True)
        except Exception:
            pass
        return

    try:
        resolve, pm, ms, project = _open_or_switch_to_project(job_id)
        if not resolve or not project:
            _log(f"ℹ Resolve not available; import skipped: {job_id}")
            # mark as imported even in "no Resolve" mode to avoid endless retries
            try:
                imported.write_text(
                    datetime.datetime.now().isoformat(timespec="seconds"),
                    encoding="utf-8",
                )
                _log("ℹ Marked as imported (no Resolve)")
            except Exception as e:
                _log(f"⚠ could not write .imported marker (no Resolve): {e}")
            try:
                importing.unlink(missing_ok=True)
            except Exception:
                pass
            return

        # --- Import to Media Pool ---
        clips = None
        mpool = project.GetMediaPool() if _callable(project, "GetMediaPool") else None
        try:
            if mpool and _callable(mpool, "ImportMedia"):
                clips = mpool.ImportMedia([str(mp4)])
                if clips:
                    _log(f"[pool] ✅ {mp4.name} via MediaPool")
                else:
                    _log(f"[pool] ⚠ MediaPool import failed for {mp4.name}")
            elif ms and _callable(ms, "AddItemListToMediaPool"):
                clips = ms.AddItemListToMediaPool([str(mp4)])
                if clips:
                    _log(f"[pool] ✅ {mp4.name} via MediaStorage")
                else:
                    _log(f"[pool] ⚠ MediaStorage import failed for {mp4.name}")
            else:
                _log("⚠ Neither MediaPool.ImportMedia nor MediaStorage.AddItemListToMediaPool available")
        except Exception as e:
            _log(f"⚠ media import error: {e}")

        # --- Ensure timeline + append preview clip ---
        try:
            mpool = project.GetMediaPool() if _callable(project, "GetMediaPool") else None
            tl = _ensure_timeline(project, mpool, job_id)
            if tl and mpool and _callable(mpool, "AppendToTimeline"):
                to_add = clips if isinstance(clips, list) else ([clips] if clips else [])
                if not to_add and mpool and _callable(mpool, "GetRootFolder"):
                    root = mpool.GetRootFolder()
                    if root and _callable(root, "GetClipList"):
                        for clip in (root.GetClipList() or []):
                            if clip and _callable(clip, "GetName") and clip.GetName() == mp4.name:
                                to_add = [clip]
                                break
                if to_add:
                    ok = mpool.AppendToTimeline(to_add)
                    _log(f"[timeline] ✅ appended {mp4.name}" if ok else "[timeline] ⚠ AppendToTimeline returned False")
                else:
                    _log(f"⚠ clip not found in pool after import: {mp4.name}")
            else:
                _log(f"⚠ no timeline {job_id}")
        except Exception as e:
            _log(f"⚠ timeline append error: {e}")

        # --- Set modern YouTube export preset automatically ---
        try:
            exp_dir = (
                paths.final_video_dir(channel, day)
                if (paths and hasattr(paths, "final_video_dir"))
                else _channel_root(channel) / "FinalVideo" / day
            )
            exp_dir.mkdir(parents=True, exist_ok=True)

            # Base filename only; Resolve / container will append the extension
            outfile = f"{job_id}"

            # Clean slate: remove previous jobs to avoid confusion
            if _callable(project, "DeleteAllRenderJobs"):
                try:
                    project.DeleteAllRenderJobs()
                except Exception as e:
                    _log(f"⚠ DeleteAllRenderJobs failed: {e}")

            # --- PRIMARY PATH: try to load YTAuto preset (best-quality config) ---
            preset_ok = False
            if _callable(project, "LoadRenderPreset"):
                try:
                    preset_ok = bool(project.LoadRenderPreset("YTAuto"))
                    _log(f"[export] preset YTAuto → {preset_ok}")
                except Exception as e:
                    _log(f"⚠ LoadRenderPreset('YTAuto') failed: {e}")

            if preset_ok:
                # Let the preset control all advanced options; we only override path/name.
                yt_settings = {
                    "TargetDir": str(exp_dir),
                    "CustomName": outfile,
                    "SelectAllFrames": True,
                }

                if _callable(project, "SetRenderSettings"):
                    try:
                        ok = project.SetRenderSettings(yt_settings)
                        _log(
                            f"[export] ✅ YTAuto → {outfile}"
                            if ok
                            else "[export] ⚠ SetRenderSettings (YTAuto) returned False; using preset defaults"
                        )
                    except Exception as e:
                        _log(f"⚠ SetRenderSettings (YTAuto) error: {e}")

                # Optional: log current format/codec so we can confirm the preset
                try:
                    if _callable(project, "GetCurrentRenderFormatAndCodec"):
                        frc = project.GetCurrentRenderFormatAndCodec() or {}
                        _log(
                            f"[export] fmt/codec (YTAuto) → "
                            f"{frc.get('format')!r}/{frc.get('codec')!r}"
                        )
                except Exception:
                    pass

            else:
                # --- FALLBACK PATH: manual settings (no preset available) ---
                mp4_format_key = None
                h264_codec_key = None

                try:
                    if _callable(project, "GetRenderFormats"):
                        rf = project.GetRenderFormats() or {}
                        # Example: {'QuickTime': 'mov', 'MP4': 'mp4', 'AVI': 'avi', ...}
                        for fmt_name, ext in rf.items():
                            if str(ext).lower() == "mp4":
                                mp4_format_key = fmt_name
                                break

                    if mp4_format_key and _callable(project, "GetRenderCodecs"):
                        rc = project.GetRenderCodecs(mp4_format_key) or {}
                        # Example: {'H.264': 'H264', 'H.265': 'H265'}
                        for desc, code in rc.items():
                            if "264" in str(desc):
                                h264_codec_key = code
                                break
                        if not h264_codec_key:
                            for desc, code in rc.items():
                                if "264" in str(code):
                                    h264_codec_key = code
                                    break
                except Exception as e:
                    _log(f"⚠ render format introspection failed: {e}")

                # Fallback defaults if API didn’t give us better answers
                mp4_format_key = mp4_format_key or "MP4"
                h264_codec_key = h264_codec_key or "H264"

                # Ask Resolve to switch container+codec using its helper
                if _callable(project, "SetCurrentRenderFormatAndCodec"):
                    try:
                        ok_sc = project.SetCurrentRenderFormatAndCodec(mp4_format_key, h264_codec_key)
                        _log(f"[export] format/codec → {mp4_format_key}/{h264_codec_key} ({ok_sc})")
                    except Exception as e:
                        _log(f"⚠ SetCurrentRenderFormatAndCodec error: {e}")

                # Manual “best YouTube” settings – this is your previously working block.
                fb_settings = {
                    # basic selection (these drive the UI dropdowns)
                    "Format": mp4_format_key,      # e.g. 'MP4'
                    "Codec": h264_codec_key,       # some builds use this
                    "VideoCodec": "H.264",         # others use this human label

                    # location + filename
                    "TargetDir": str(exp_dir),
                    "CustomName": outfile,
                    "UseCustomName": True,

                    # render both
                    "ExportVideo": True,
                    "ExportAudio": True,

                    # resolution & fps
                    "FormatWidth": 1920,
                    "FormatHeight": 1080,
                    "FrameRate": 24,

                    # quality: Restrict to 20,000 Kb/s
                    "VideoQuality": 20000,  # Restrict to 20000 Kb/s (per API docs)

                    # color / levels (best-effort; some builds may ignore these)
                    "PixelAspectRatio": "square",
                    "DataLevels": "Video",
                    "ColorSpaceTag": "Rec.709",
                    "GammaTag": "Rec.709",

                    # behavior flags
                    "NetworkOptimization": False,
                    "BypassReEncode": False,
                    "BypassReencode": False,
                    "EnableHardwareEncoding": True,
                }

                # Audio keys (names differ slightly across builds)
                fb_settings.update(
                    {
                        "AudioCodec": "AAC",
                        "AudioEncoder": "AAC",
                        "AudioBitRate": 192,
                        "AudioBitrate": 192,
                        "AudioSampleRate": 48000,
                        "AudioChannels": "stereo",
                    }
                )

                if _callable(project, "SetRenderSettings"):
                    try:
                        success = project.SetRenderSettings(fb_settings)
                        if success:
                            _log(f"[export] ✅ fallback MP4/H.264 20Mbps → {outfile}")
                        else:
                            _log("[export] ⚠ Resolve rejected fallback SetRenderSettings; using defaults.")
                    except Exception as e:
                        _log(f"⚠ SetRenderSettings (fallback) error: {e}")

                # Log what Resolve actually kept so we can compare with the UI
                try:
                    rs = project.GetRenderSettings() or {}
                    _log(
                        "[export] effective → "
                        f"fmt={rs.get('Format')!r} codec={rs.get('Codec', rs.get('VideoCodec'))!r} "
                        f"vb={rs.get('VideoBitRate', rs.get('VideoBitrate', '?'))} "
                        f"q={rs.get('Quality')!r}"
                    )
                except Exception:
                    pass

        except Exception as e:
            _log(f"⚠ export preset failed: {e}")

        # --- Mark imported (single authoritative marker) ---
        try:
            imported.write_text(
                datetime.datetime.now().isoformat(timespec="seconds"),
                encoding="utf-8",
            )
            _log("✓ wrote .imported marker")
            _log(f"✅ Resolve import complete: {job_id}")
        except Exception as e:
            _log(f"⚠ could not write .imported marker: {e}")

    finally:
        try:
            importing.unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    run_import()