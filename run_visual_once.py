#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-shot visual pipeline runner for VideoAutomation.

This script runs EXACTLY ONE visual-pipeline round by delegating to
auto_on_job.run_one_round(), which:
    • Scans channels
    • Finds at most ONE pending job
    • Runs scene → video → import as needed
    • Respects the LLM_DISABLED sentinel
    • Exits immediately

It is designed for launchd or manual one-shot invocation.

NOTE: auto_on_job.py must contain a function:
    run_one_round()
that performs one job-processing cycle and then returns.
"""

import os
import sys
import datetime
from pathlib import Path

VA_ROOT = Path(os.environ.get("VA_ROOT", str(Path.home() / "VideoAutomation")))
SYS_VIS = VA_ROOT / "System" / "SysVisuals"
TOOLS   = SYS_VIS / "Tools"
ORCH    = SYS_VIS / "Orchestrators"

# --- Central log directory ---
SYS = VA_ROOT / "System"
LOG_DIR = SYS / "Logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "run_visual_once.log"

# Ensure Tools + Orchestrators are importable
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
if str(ORCH) not in sys.path:
    sys.path.insert(0, str(ORCH))

# Import one-round job runner
try:
    import auto_on_job  # type: ignore
except Exception as exc:
    auto_on_job = None
    _import_error = exc
else:
    _import_error = None


def log(msg: str) -> None:
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[run_visual_once {timestamp}] {msg}"
    # Print only if error
    if "ERROR" in msg or "🛑" in msg:
        print(line, flush=True)
    # Always write errors to central log
    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(line + "\n")
    except Exception:
        pass


def run_once() -> None:
    """Perform a single visual-pipeline job round and exit.

    This calls auto_on_job.run_one_round() and returns. All real work
    (scene/video/import logging) happens inside auto_on_job.
    """
    if auto_on_job is None:
        log("ERROR: auto_on_job module not available.")
        if _import_error is not None:
            log(f"ERROR: import error: {_import_error}")
        return

    if not hasattr(auto_on_job, "run_one_round"):
        log("ERROR: auto_on_job.run_one_round() missing.")
        log("You must implement run_one_round() inside auto_on_job.py.")
        return

    try:
        # Delegate to auto_on_job.run_one_round(); that module will log
        # real work (scene/video/import) as needed. We intentionally keep
        # this wrapper quiet to avoid noisy \"starting/completed\" messages
        # every StartInterval when there is no pending job.
        auto_on_job.run_one_round()  # type: ignore
    except SystemExit as exc:
        log(f"ERROR: auto_on_job.run_one_round exited with {exc.code}")
    except Exception as exc:
        log(f"ERROR during visual run: {exc}")


if __name__ == "__main__":
    run_once()