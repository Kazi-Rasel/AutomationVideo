#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys
import subprocess
import shlex
import datetime
import json
import shutil

# Safe Pillow import for upscaling/resizing
try:
    from PIL import Image  # type: ignore
except Exception:
    Image = None
from pathlib import Path
from typing import List, Tuple, Optional

# ---------- logging ----------
# (logging implementation moved below project paths)

# Debug logger for full ffmpeg commands, honors VIDEO_LOG_LEVEL
VIDEO_LOG_LEVEL = os.environ.get("VIDEO_LOG_LEVEL", "info").lower()
def _log_debug(msg: str) -> None:
    if VIDEO_LOG_LEVEL == "debug":
        _log(msg)

# ---------- project paths ----------

VA_ROOT = Path(os.environ.get("VA_ROOT", str(Path.home() / "VideoAutomation")))
SYS     = VA_ROOT / "System"
SYSVIS  = SYS / "SysVisuals"
SHARED  = SYSVIS / "Shared"

# --- Central log directory ---
LOG_DIR = SYS / "Logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "video_orchestrator.log"
if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))
try:
    import paths  # type: ignore
except Exception:
    paths = None

# LLM sentinel: if present, we should not start/continue LLM-driven jobs
LLM_SENTINEL = SYSVIS / "LLM" / "LLM_DISABLED"

def _llm_disabled() -> bool:
    """Return True if the LLM_DISABLED sentinel exists.

    When present, the video orchestrator should not start or continue
    composition for this job. This mirrors watcher.py, auto_on_job.py,
    and scene_orchestrator.py to prevent partial/corrupt outputs when
    the LLM layer or overall pipeline is deliberately paused.
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

# Karaoke builder (ASS)
try:
    from karaoke_builder import build_karaoke_ass  # in SysVisuals/Shared
except Exception:
    build_karaoke_ass = None

# ---------- path helpers ----------
def channel_p(channel: str) -> Path:
    return paths.channel_root(channel) if (paths and hasattr(paths, 'channel_root')) else (VA_ROOT / "Channels" / channel)

def job_root_p(channel: str, day: str, job_id: str) -> Path:
    if paths and hasattr(paths, 'build_ready_job_root'):
        return paths.build_ready_job_root(channel, job_id, day)  # type: ignore[attr-defined]
    return channel_p(channel) / "Build" / "Ready" / day / job_id

def preview_p(channel: str, day: str, job_id: str) -> Path:
    if paths and hasattr(paths, 'job_preview_path'):
        return paths.job_preview_path(channel, day, job_id)  # type: ignore[attr-defined]
    return job_root_p(channel, day, job_id) / f"{job_id}__mv_preview.mp4"

# ---------- shell helpers ----------
def _run(cmd: List[str]) -> Tuple[int, str, str]:
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    return p.returncode, out, err


def _ffprobe_duration_media(p: Path) -> Optional[float]:
    cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=nw=1:nk=1',
        str(p)
    ]
    rc, out, err = _run(cmd)
    if rc != 0:
        return None
    try:
        return max(0.0, float(out.strip()))
    except Exception:
        return None

# ---------- optional image upscaling helper ----------
def _maybe_upscale_image(img: Path, work_dir: Path) -> Path:
    """
    Normalize scene images to 4800x3600 (4:3) before motion, without external AI upscaling.

    Rationale:
      • The previous RealESRGAN NCNN Vulkan binary segfaults on modern macOS/M3 hardware.
      • To keep the pipeline stable, we always fall back to a pure Pillow-based resize.

    Pipeline:
      1) Check for a cached <stem>__upx.jpg in work_dir; reuse if it is 4800x3600.
      2) Otherwise, resize the source image to exactly 4800x3600 using Pillow (LANCZOS).
      3) Cache the result as <stem>__upx.jpg inside work_dir.

    This keeps pan/zoom math stable and avoids any dependency on fragile native binaries.
    """
    target_w, target_h = 4800, 3600

    out_img = work_dir / f"{img.stem}__upx.jpg"

    # If cached and up-to-date and already the right size, reuse
    try:
        if out_img.exists():
            src_mtime = img.stat().st_mtime
            out_mtime = out_img.stat().st_mtime
            if out_mtime >= src_mtime and Image is not None:
                try:
                    with Image.open(out_img) as im:
                        if im.size == (target_w, target_h):
                            return out_img
                except Exception:
                    pass
    except Exception:
        pass

    # No AI step: we only use Pillow to normalize to 4800x3600.
    if Image is None:
        # Pillow not available; just return the original image.
        return img

    candidate = img
    try:
        with Image.open(candidate) as im:
            if im.size != (target_w, target_h):
                _log(f"[upx] ℹ resizing {candidate.name} → {out_img.name} to {target_w}x{target_h}")
                im = im.convert("RGB")
                im = im.resize((target_w, target_h), resample=Image.LANCZOS)
                out_img.parent.mkdir(parents=True, exist_ok=True)
                im.save(out_img, format="JPEG", quality=95, subsampling=0)
                return out_img
    except Exception as e:
        _log(f"[upx] ⚠ final resize failed for {candidate.name}: {e}")
        return candidate

    # If source was already 4800x3600, just return it.
    return candidate

# ---------- film overlay helper (blur + color + vignette + scratches) ----------

def _apply_film_overlay(base_video: Path, out_video: Path) -> bool:
    overlay = SYSVIS / "film_damage_overlay.mp4"
    if not overlay.exists():
        _log(f"⚠ film overlay not found: {overlay} (copying base video instead)")
        try:
            shutil.copy2(str(base_video), str(out_video))
            return out_video.exists() and out_video.stat().st_size > 0
        except Exception as e:
            _log(f"⚠ failed to copy base video: {e}")
            return False

    # NEW: read base video duration so we can clamp output length
    base_dur = _ffprobe_duration_media(base_video) or 0.0
    if base_dur <= 0.0:
        _log(f"⚠ could not read duration for {base_video}, skipping overlay")
        try:
            shutil.copy2(str(base_video), str(out_video))
            return out_video.exists() and out_video.stat().st_size > 0
        except Exception as e:
            _log(f"⚠ failed to copy base video: {e}")
            return False

    # --- FINAL TWEAK: REMOVE GREEN TINT (GOLDEN SHIFT) ---

    # 1. Color Balance (Anti-Green Fix):
    cb_str = (
        "rs=0.06:rm=0.06:rh=0.06:"
        "bs=-0.06:bm=-0.06:bh=-0.06:"
        "gs=-0.02:gm=-0.02:gh=-0.02"
    )
    
    # 2. Exposure Fix (Kept Exact):
    exposure_fix = "curves=master='0/0.06 1/1',eq=gamma=1.05:saturation=0.95"
    
    # 3. Vignette (Kept Exact - 10%):
    vig_str = "vignette=PI/6"

    filter_complex = (
        # Luma-Only Blur -> Exposure -> Anti-Green Color -> Vignette
        f"[0:v] gblur=sigma=0.5:planes=1,{exposure_fix},"
        f"colorbalance={cb_str},{vig_str},format=gbrp[base_rgb];"
        # Overlay (Approved Subtle Strength)
        "[1:v] hue=s=0,setsar=1,curves=all='0/0 0.4/0.80 0.7/1 1/1',format=gbrp[ol_rgb];"
        # RGB Multiply Blend
        "[base_rgb][ol_rgb] blend=all_mode=multiply,format=yuv420p[v]"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", str(base_video),
        "-stream_loop", "-1",           # 🔁 loop overlay forever
        "-i", str(overlay),
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-t", f"{base_dur:.6f}",        # ⏱ clamp to base video length
        "-movflags", "+faststart",
        str(out_video),
    ]
    _log(f"[overlay] 🎬 film damage → {out_video.name} ({base_dur:.2f}s)")
    _log_debug("FFMPEG " + " ".join(shlex.quote(x) for x in cmd))
    rc, out, err = _run(cmd)
    if rc != 0 or not out_video.exists() or out_video.stat().st_size == 0:
        _log(f"⚠ film overlay ffmpeg failed\n{err}")
        try:
            shutil.copy2(str(base_video), str(out_video))
            return out_video.exists() and out_video.stat().st_size > 0
        except Exception as e:
            _log(f"⚠ failed to copy base video after overlay error: {e}")
            return False
    return True

# ---------- render: 3x pan + centered zoom-in, 24fps, no flatten ----------
def _render_scene(
    img: Path,
    out_mp4: Path,
    dur_s: float,
    zoom_step: float = 0.0,   # unused
    direction_up: bool = True
) -> bool:
    """
    Combined PAN + ZOOM-IN on a fixed 4:3 AI-upscaled canvas.

      • Every scene image is normalized to 4800x3600 (4:3) by _maybe_upscale_image().
      • Vertical pan happens inside that 4800x3600 space.
      • A 4800x2700 16:9 window is cropped with animated y_expr (Ken Burns move).
      • direction_up=True  → image moves UP over the scene
        direction_up=False → image moves DOWN over the scene
      • Centered zoom-in (slow, subtle):
        ~0.65% per second, max +8% total, from the center (all four directions)

    Pipeline:
      1) input: 4800x3600 still
      2) crop=4800:2700 with animated y_expr (pan)
      3) zoompan with z_expr + center x/y, output 1920x1080
    """
    if dur_s <= 0:
        return False

    fps    = 24.0
    frames = int(round(dur_s * fps))
    if frames < 1:
        return False
    denom = max(frames - 1, 1)

    # ----- PAN (your working move version) -----
    BASE_TRAVEL = 0.25
    REF_DUR     = 7.0
    travel = BASE_TRAVEL * (dur_s / REF_DUR)
    if travel > 1.0:
        travel = 1.0
    if travel < 0.0:
        travel = 0.0

    off_each = (1.0 - travel) / 2.0  # never seen top/bottom

    if travel <= 0.0:
        y_expr = "(in_h-2700)/2"
    else:
        if direction_up:
            # image UP → crop y moves DOWN: off_each*D → (1-off_each)*D
            y_expr = (
                f"(in_h-2700)*{off_each:.6f} + "
                f"(in_h-2700)*{travel:.6f}*n/{denom}"
            )
        else:
            # image DOWN → crop y moves UP: (1-off_each)*D → off_each*D
            y_expr = (
                f"(in_h-2700)*{(1.0 - off_each):.6f} - "
                f"(in_h-2700)*{travel:.6f}*n/{denom}"
            )

    # ----- ZOOM (your working zoom version, slowed to ~35%) -----
    ZOOM_FRAC_PER_SEC = 0.0065   # was 0.01 → ~35% slower
    ZOOM_MAX_FRAC     = 0.08     # max +8%
    zoom_total = ZOOM_FRAC_PER_SEC * dur_s
    if zoom_total > ZOOM_MAX_FRAC:
        zoom_total = ZOOM_MAX_FRAC
    if zoom_total < 0.0:
        zoom_total = 0.0

    # z(on) from 1.0 → 1.0+zoom_total over all frames
    z_expr = f"1+{zoom_total:.6f}*on/{denom}"

    # Centered zoom: crop center stays center
    x_expr = "(iw - iw/zoom)/2"
    y_zoom = "(ih - ih/zoom)/2"

    # ----- FULL FILTER CHAIN -----
    vf = (
        "setsar=1,"
        f"crop=4800:2700:(in_w-4800)/2:{y_expr},"
        f"zoompan=z='{z_expr}':x='{x_expr}':y='{y_zoom}':d=1:s=1920x1080"
    )

    cmd = [
        "ffmpeg", "-y",
        "-loop", "1",
        "-i", str(img),
        "-vf", vf,
        "-frames:v", str(frames),
        "-r", "24", "-vsync", "cfr",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-movflags", "+faststart",
        str(out_mp4),
    ]
    _log(f"[seg] 🎬 {img.name} → {out_mp4.name} ({dur_s:.2f}s, dir={'up' if direction_up else 'down'})")
    _log_debug("FFMPEG " + " ".join(shlex.quote(x) for x in cmd))
    rc, out, err = _run(cmd)
    if rc != 0:
        _log(f"⚠ ffmpeg failed on {img.name}\n{err}")
        return False

    return out_mp4.exists() and out_mp4.stat().st_size > 1024

# ---------- concat segments with crossfade ----------
def _concat_segments(seg_files: List[Path], joined_txt: Path, joined_mp4: Path) -> bool:
    """
    Join segments with a small crossfade between them.
    'joined_txt' is unused but kept for compatibility.

    Crossfade duration:
      SCENE_XFADE_SEC env or default 3 frames at 24fps (~0.125s).
    """
    if not seg_files:
        return False

    fps = 24.0
    # default = 3 frames at 24fps
    default_sec = 3.0 / fps
    fade_sec = float(os.environ.get("SCENE_XFADE_SEC", str(default_sec)))

    if len(seg_files) == 1:
        cmd = [
            'ffmpeg', '-y',
            '-i', str(seg_files[0]),
            '-r', str(int(fps)), '-vsync', 'cfr',
            '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
            '-movflags', '+faststart',
            str(joined_mp4),
        ]
        _log(f"[xfade] 🎬 single segment → {joined_mp4.name}")
        _log_debug("FFMPEG " + " ".join(shlex.quote(x) for x in cmd))
        rc, out, err = _run(cmd)
        if rc != 0:
            _log(f"⚠ single-seg remux failed\n{err}")
            return False
        return joined_mp4.exists() and joined_mp4.stat().st_size > 1024

    work_dir = seg_files[0].parent
    current = seg_files[0]

    for idx, seg in enumerate(seg_files[1:], start=1):
        dur_current = _ffprobe_duration_media(current) or 0.0
        if dur_current <= fade_sec:
            offset = max(0.0, dur_current / 2.0 - fade_sec / 2.0)
        else:
            offset = max(0.0, dur_current - fade_sec)

        out_path = work_dir / f"xfade_{idx:03d}.mp4"

        cmd = [
            'ffmpeg', '-y',
            '-i', str(current),
            '-i', str(seg),
            '-filter_complex',
            f"xfade=transition=fade:duration={fade_sec}:offset={offset}",
            '-r', str(int(fps)), '-vsync', 'cfr',
            '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
            '-movflags', '+faststart',
            str(out_path),
        ]
        _log(f"[xfade] 🎬 [{idx}/{len(seg_files)-1}] {current.name} + {seg.name} → {out_path.name}")
        _log_debug("FFMPEG " + " ".join(shlex.quote(x) for x in cmd))
        rc, out, err = _run(cmd)
        if rc != 0:
            _log(f"⚠ xfade failed on {current.name} + {seg.name}\n{err}")
            return False

        current = out_path

    try:
        shutil.move(str(current), str(joined_mp4))
    except Exception:
        try:
            shutil.copy2(str(current), str(joined_mp4))
        except Exception as e:
            _log(f"⚠ could not move/copy final xfade output: {e}")
            return False

    return joined_mp4.exists() and joined_mp4.stat().st_size > 1024

# ---------- mux audio to preview (with optional ASS burn-in) ----------
def _mux_preview(
    video_mp4: Path,
    wav_path: Path,
    out_preview: Path,
    ass_path: Optional[Path] = None
) -> bool:
    """
    Mux final video + audio into preview.mp4.
    If ass_path is provided and exists, burn-in ASS subtitles.
    """
    if not video_mp4.exists():
        return False

    use_subs = ass_path is not None and ass_path.exists()

    if use_subs:
        # Burn-in ASS subtitles (re-encode video)
        cmd: List[str] = [
            'ffmpeg', '-y',
            '-i', str(video_mp4),
            '-i', str(wav_path),
            '-filter_complex', f"[0:v]subtitles={ass_path.as_posix()}[v]",
            '-map', '[v]', '-map', '1:a:0',
            '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18',
            '-ac', '2', '-c:a', 'aac', '-b:a', '192k',
            # no -shortest: do not cut audio early
            '-movflags', '+faststart',
            str(out_preview),
        ]
    else:
        # Simple mux, copy video
        cmd = [
            'ffmpeg', '-y',
            '-i', str(video_mp4),
            '-i', str(wav_path),
            '-map', '0:v:0', '-map', '1:a:0',
            '-c:v', 'copy',
            '-ac', '2', '-c:a', 'aac', '-b:a', '192k',
            # no -shortest: do not cut audio early
            '-movflags', '+faststart',
            str(out_preview),
        ]

    label = "with subs" if use_subs else "mux"
    _log(f"[preview] 🎬 {label} → {out_preview.name}")
    _log_debug("FFMPEG " + " ".join(shlex.quote(x) for x in cmd))
    rc, out, err = _run(cmd)
    if rc != 0:
        _log(f"⚠ preview mux failed\n{err}")
        return False
    return out_preview.exists() and out_preview.stat().st_size > 1024

# ---------- orchestrate ----------
def run_video(channel: str, day: str, job_id: str, wav_path: Path, cfg: dict | None = None) -> None:
    cfg = cfg or {}
    root = job_root_p(channel, day, job_id)
    # Hard guard: if the LLM sentinel is present, do not start this job.
    if _llm_disabled():
        _log(f"🛑 Aborting video compose for {channel}/{day}/{job_id} due to LLM_DISABLED sentinel.")
        return
    scenes_dir = root / "Scenes"
    mv_dir     = root / "_mv_work"
    mv_dir.mkdir(parents=True, exist_ok=True)

    imgs = sorted(scenes_dir.glob("scene_*.jpg"))
    if not imgs:
        _log("⚠ no scenes to compose")
        return

    durations: List[float] = []
    slice_based = False

    # Try slice-based durations first
    if paths and hasattr(paths, 'prompts_json_path'):
        pjson = paths.prompts_json_path(channel, day, job_id)  # type: ignore[attr-defined]
        if pjson.exists():
            try:
                data = json.loads(pjson.read_text(encoding='utf-8'))
                slices = data.get("slices") or []
                if slices:
                    durations = [
                        max(0.05, float(sl.get("t1", 0.0)) - float(sl.get("t0", 0.0)))
                        for sl in slices
                    ]
                    slice_based = True
                    _log(f"Using slice-based durations: {durations}")
            except Exception as e:
                _log(f"⚠ slice JSON read error: {e}")

    # Fallback to beats if no slices
    if not slice_based:
        wav_dur = _ffprobe_duration_media(wav_path) or 0.0
        if wav_dur <= 0.05:
            _log("⚠ bad WAV duration; aborting compose")
            return
        base_beats  = cfg.get("beats") or [5.0, 6.0, 7.0]
        tail_margin = float(os.environ.get("PREVIEW_TAIL_MARGIN", "1.0"))
        total = 0.0
        idx   = 0
        durations = []
        while total < wav_dur + tail_margin:
            dur = base_beats[idx % len(base_beats)]
            durations.append(dur)
            total += dur
            idx   += 1
        _log(f"Using beats-based durations: {durations}")

    if not durations:
        _log("⚠ no durations; aborting compose")
        return

    # Compensate crossfade overlap by extending last slice, when slices exist
    if slice_based and len(durations) > 1:
        fps = 24.0
        default_sec = 3.0 / fps
        fade_sec = float(os.environ.get("SCENE_XFADE_SEC", str(default_sec)))
        lost = fade_sec * (len(durations) - 1)
        if lost > 0.0:
            durations[-1] += lost
            _log(f"extended last slice duration by {lost:.3f}s to compensate xfade overlap")

    seg_files: List[Path] = []

    # Render each slice/beat into a segment with per-scene direction
    for i, dur in enumerate(durations, start=1):
        if slice_based:
            scene_idx = i - 1
            if scene_idx >= len(imgs):
                scene_idx %= len(imgs)
        else:
            scene_idx = (i - 1) % len(imgs)

        img = imgs[scene_idx]
        # Upscale image if enabled
        src_img = _maybe_upscale_image(img, mv_dir)
        seg = mv_dir / f"seg_{i:03d}.mp4"

        direction_up = (scene_idx % 2 == 0)  # 0,2,4.. up; 1,3,5.. down

        if not _render_scene(src_img, seg, dur, direction_up=direction_up):
            _log(f"⚠ segment failed, skipping: {seg.name}")
            continue

        seg_files.append(seg)

    if not seg_files:
        _log("⚠ no segments rendered; abort")
        return

    joined_txt = mv_dir / "joined.txt"  # unused, kept for signature compatibility
    joined_mp4 = mv_dir / "joined.mp4"
    if not _concat_segments(seg_files, joined_txt, joined_mp4):
        return

    # --- Apply Film Damage Overlay BEFORE subtitles ---
    film_layer = mv_dir / "film_applied.mp4"
    _apply_film_overlay(joined_mp4, film_layer)

    # Build karaoke ASS for this job (if builder available)
    ass_path: Optional[Path] = None
    if callable(build_karaoke_ass):
        try:
            ass_path = build_karaoke_ass(channel, day, job_id)
        except Exception as e:
            _log(f"⚠ karaoke ASS build error: {e}")

    preview_mp4 = preview_p(channel, day, job_id)
    # Burn subtitles on the film-layered video
    if _mux_preview(film_layer, wav_path, preview_mp4, ass_path=ass_path):
        _log(f"✓ preview → {preview_mp4}")
    else:
        _log("⚠ preview mux failed (video ok, audio missing?)")

# ---------- legacy entry ----------
def run_video_legacy(channel: str, day: str, job_id: str) -> None:
    if _llm_disabled():
        _log(f"🛑 Aborting legacy video compose for {channel}/{day}/{job_id} due to LLM_DISABLED sentinel.")
        return
    chroot    = channel_p(channel)
    wav_guess = None
    guess1    = chroot / "Input" / "Audio" / day / f"{job_id}.wav"
    guess2    = chroot / "Input" / "Audio" / day / f"{job_id}_final.wav"
    for g in (guess1, guess2):
        if g.exists():
            wav_guess = g
            break
    if not wav_guess:
        _log("⚠ cannot guess WAV path; legacy call aborted")
        return
    run_video(channel, day, job_id, wav_guess, None)
# ---------- logging ----------
def _log(msg: str) -> None:
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[VIDEO {timestamp}] {msg}"
    # print to terminal
    print(line, flush=True)
    # append to central log file
    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(line + "\n")
    except Exception:
        pass