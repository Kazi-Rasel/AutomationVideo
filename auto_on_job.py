# File: /Users/kazirasel/VideoAutomation/System/SysVisuals/Tools/auto_on_job.py
# #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto job runner for VideoAutomation.

Watches all Channels/*/Input/Audio/<day>/*.wav and, when it sees a new or
updated job, runs:

  1) scene_orchestrator.run_scene(...)
  2) video_orchestrator.run_video(...)
  3) import_orchestrator.run_import(...)

This replaces the old YTProjects-based auto_on_job.py. It has no references
to YTProjects / GeminiTTS and works only with the VideoAutomation layout.
"""

from __future__ import annotations

import datetime
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

# ---- Project roots ----
VA_ROOT = Path(os.environ.get("VA_ROOT", str(Path.home() / "VideoAutomation"))).resolve()
SYS     = VA_ROOT / "System"
SYSVIS  = SYS / "SysVisuals"
SHARED  = SYSVIS / "Shared"
ORCH    = SYSVIS / "Orchestrators"
TOOLS   = SYSVIS / "Tools"

# --- Central log directory ---
LOG_DIR = SYS / "Logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "auto_on_job.log"

for p in (str(SYSVIS), str(SHARED), str(ORCH)):
    if p not in sys.path:
        sys.path.insert(0, p)

import paths  # type: ignore
import scene_orchestrator  # type: ignore
import video_orchestrator  # type: ignore
import import_orchestrator  # type: ignore

# ---- LLM sentinel ----
LLM_SENTINEL = SYSVIS / "LLM" / "LLM_DISABLED"

def llm_disabled() -> bool:
    """Return True if the LLM_DISABLED sentinel exists.

    When present, the entire auto_on_job loop must stop and not
    start/continue ANY jobs (scene, video, import). This prevents
    corrupt/partial jobs when LLM quota is exhausted or deliberately paused.
    """
    try:
        if LLM_SENTINEL.exists():
            try:
                reason = LLM_SENTINEL.read_text(encoding="utf-8", errors="ignore").strip()
            except Exception:
                reason = ""
            if not reason:
                reason = "LLM disabled by sentinel file."
            log(f"🚫 LLM_DISABLED sentinel present at {LLM_SENTINEL}")
            log(f"🚫 Reason: {reason}")
            return True
    except Exception as e:
        log(f"⚠ Failed to read LLM sentinel: {e}")
    return False

# ---- Config ----
CHANNELS_ROOT = VA_ROOT / "Channels"
STAMP_PATH    = TOOLS / ".jobstamp.json"
POLL_SEC      = float(os.environ.get("VA_JOB_POLL_SEC", "5.0"))


@dataclass
class Job:
    channel: str
    day: str
    job_id: str
    wav: Path
    script: Path
    mtime: float  # latest of wav/script


def log(msg: str) -> None:
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[AUTO {timestamp}] {msg}"
    # print to terminal
    print(line, flush=True)
    # append to central log file
    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(line + "\n")
    except Exception:
        pass


def _load_stamp() -> Dict[str, float]:
    try:
        text = STAMP_PATH.read_text(encoding="utf-8")
        data = json.loads(text)
        if isinstance(data, dict):
            # keys: job_key, values: float mtime
            return {str(k): float(v) for k, v in data.items()}
    except Exception:
        pass
    return {}


def _save_stamp(stamp: Dict[str, float]) -> None:
    try:
        STAMP_PATH.parent.mkdir(parents=True, exist_ok=True)
        STAMP_PATH.write_text(json.dumps(stamp, indent=2), encoding="utf-8")
    except Exception:
        # non-fatal
        pass


def _iter_channel_dirs() -> List[Path]:
    """Return all channel root dirs under Channels/."""
    if not CHANNELS_ROOT.exists():
        return []
    return sorted([p for p in CHANNELS_ROOT.iterdir() if p.is_dir()])


def _find_jobs_for_channel(ch_root: Path) -> List[Job]:
    """
    Discover jobs for a single channel.

    Assumes layout:
      Channels/<channel>/Input/Audio/<day>/<job_id>.wav
      Channels/<channel>/Input/Script/<day>/<job_base>.txt

    where job_base is job_id without a trailing '_final'.
    """
    jobs: List[Job] = []
    channel_name = ch_root.name

    audio_root = ch_root / "Input" / "Audio"
    script_root = ch_root / "Input" / "Script"
    if not audio_root.exists():
        return jobs

    for day_dir in sorted([d for d in audio_root.iterdir() if d.is_dir()]):
        day_name = day_dir.name
        for wav in sorted(day_dir.glob("*.wav")):
            job_id = wav.stem
            job_base = re.sub(r"_final$", "", job_id)
            script_dir = script_root / day_name
            script = script_dir / f"{job_base}.txt"
            if not script.exists():
                continue
            try:
                mt = max(wav.stat().st_mtime, script.stat().st_mtime)
            except Exception:
                continue
            jobs.append(Job(channel_name, day_name, job_id, wav, script, mt))

    return jobs


def _run_scene_safe(job: Job) -> None:
    """Run scene_orchestrator.run_scene with proper signature; never crash loop."""
    fn = getattr(scene_orchestrator, "run_scene", None)
    if not callable(fn):
        kind = "None" if fn is None else type(fn).__name__
        log(f"⚠ scene_orchestrator.run_scene not callable (got {kind}); skipping scene stage.")
        return
    try:
        fn(job.channel, job.day, job.job_id, job.wav, job.script, {})
    except TypeError as e:
        log(f"⚠ run_scene() signature mismatch; please update scene_orchestrator.run_scene :: {e}")
    except Exception as e:
        log(f"⚠ run_scene() error for {job.channel}/{job.day}/{job.job_id}: {e}")


def _run_video_safe(job: Job) -> None:
    """Run video_orchestrator.run_video with proper signature; never crash loop."""
    fn = getattr(video_orchestrator, "run_video", None)
    if not callable(fn):
        kind = "None" if fn is None else type(fn).__name__
        log(f"⚠ video_orchestrator.run_video not callable (got {kind}); skipping video stage.")
        return
    try:
        fn(job.channel, job.day, job.job_id, job.wav, {})
    except TypeError as e:
        log(f"⚠ run_video() signature mismatch; please update video_orchestrator.run_video :: {e}")
    except Exception as e:
        log(f"⚠ run_video() error for {job.channel}/{job.day}/{job.job_id}: {e}")


def _run_import_safe(job: Job) -> None:
    """Run import_orchestrator.run_import with canonical signature; never crash loop."""
    fn = getattr(import_orchestrator, "run_import", None)
    if not callable(fn):
        kind = "None" if fn is None else type(fn).__name__
        log(f"⚠ import_orchestrator.run_import not callable (got {kind}); skipping Resolve import.")
        return
    try:
        # Preferred signature: (channel, day, job_id)
        fn(job.channel, job.day, job.job_id)
    except TypeError as e:
        log(f"⚠ run_import() signature mismatch; please update import_orchestrator.run_import :: {e}")
    except Exception as e:
        log(f"⚠ run_import() error for {job.channel}/{job.day}/{job.job_id}: {e}")


def _job_key(job: Job) -> str:
    return f"{job.channel}/{job.day}/{job.job_id}"


def _process_job(job: Job, stamp: Dict[str, float]) -> None:
    # Safety: if sentinel appears mid-run, abort this job immediately
    if llm_disabled():
        log(f"🛑 Skipping job due to LLM_DISABLED: {job.channel}/{job.day}/{job.job_id}")
        return
    key = _job_key(job)

    # If a final preview already exists for this job, consider it done and skip.
    # This prevents run_one_round from re-running the same job every interval
    # in the event-driven launchd mode.
    ch_root = CHANNELS_ROOT / job.channel
    preview = ch_root / "Build" / "Ready" / job.day / job.job_id / f"{job.job_id}__mv_preview.mp4"
    if preview.exists():
        log(f"ℹ preview already exists; skipping job: {key}")
        return

    log(f"▶ start {key}")
    started = time.time()

    _run_scene_safe(job)
    _run_video_safe(job)
    _run_import_safe(job)

    elapsed = time.time() - started
    log(f"✔ done  {key} in {elapsed:.1f}s")

    stamp[key] = job.mtime
    _save_stamp(stamp)


def run_one_round() -> None:
    """Perform a single auto-on-job cycle and return.

    This is intended for one-shot invocations (e.g. via run_visual_once.py
    or launchd). It:
      - loads the current stamp state
      - checks the LLM_DISABLED sentinel
      - discovers pending jobs
      - filters out jobs that already have a final preview
      - sorts remaining jobs by mtime
      - processes at most one job
      - updates the stamp file
    """
    stamp = _load_stamp()

    # Do not start any work if the global sentinel is set.
    if llm_disabled():
        log("🛑 LLM_DISABLED is set; skipping visual run.")
        return

    channels = _iter_channel_dirs()
    all_jobs: List[Job] = []
    for ch_root in channels:
        all_jobs.extend(_find_jobs_for_channel(ch_root))

    # Filter out jobs that already have a final preview
    pending: List[Job] = []
    for job in all_jobs:
        ch_root = CHANNELS_ROOT / job.channel
        preview = ch_root / "Build" / "Ready" / job.day / job.job_id / f"{job.job_id}__mv_preview.mp4"
        if preview.exists():
            # Already has a final preview: treat as done in one-shot mode
            continue
        pending.append(job)

    if not pending:
        # Nothing to do in this one-shot round; stay quiet to avoid log spam
        return

    # Process the oldest pending job
    pending.sort(key=lambda j: j.mtime)
    job = pending[0]
    _process_job(job, stamp)


def main() -> None:
    log(f"Auto runner watching Channels/ under {VA_ROOT}")
    stamp = _load_stamp()

    while True:
        try:
            # Hard guard: if LLM is disabled, do NOT run any jobs.
            if llm_disabled():
                log("🛑 Stopping auto_on_job loop due to LLM_DISABLED sentinel.")
                break

            channels = _iter_channel_dirs()
            all_jobs: List[Job] = []
            for ch_root in channels:
                all_jobs.extend(_find_jobs_for_channel(ch_root))

            # process in ascending mtime order so oldest new job goes first
            all_jobs.sort(key=lambda j: j.mtime)

            for job in all_jobs:
                _process_job(job, stamp)

        except KeyboardInterrupt:
            log("bye.")
            break
        except Exception as e:
            # keep the loop alive
            log(f"⚠ loop error: {e}")
            try:
                import traceback
                traceback.print_exc()
            except Exception:
                pass

        time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()