#!/bin/zsh
# StartServices.command
# - Clean safe leftovers in PutScript (no .txt deletion)
# - Clean transient SysTTS temp dirs (not per-job outputs)
# - Create central log dir at System/Logs
# - Start watcher.py in this terminal (foreground, logs visible here)

set -euo pipefail
setopt NULL_GLOB

VA_ROOT="${VA_ROOT:-$HOME/VideoAutomation}"
SYS="$VA_ROOT/System"
SYS_TTS="$SYS/SysTTS"
LOGS="$SYS/Logs"
PUTSCRIPT="$VA_ROOT/PutScript"

log() {
  local ts
  ts="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "[StartServices] [$ts] $*"
}

log "VA_ROOT: $VA_ROOT"

# --- Ensure base dirs exist ---
mkdir -p "$PUTSCRIPT"
mkdir -p "$LOGS"

# --- Clean safe leftovers in PutScript (no .txt removal) ---
# Lock + fail + meta markers
rm -f  "$PUTSCRIPT"/*.lock       2>/dev/null || true
rm -f  "$PUTSCRIPT"/*.fail       2>/dev/null || true
rm -f  "$PUTSCRIPT"/*.meta       2>/dev/null || true
rm -f  "$PUTSCRIPT"/*.meta.json  2>/dev/null || true
# tmp helper dirs
rm -rf "$PUTSCRIPT"/.tmp_*       2>/dev/null || true

# NOTE: We do NOT touch *.txt here, so any unfinished script stays and will be resumed.

# --- Clean obvious SysTTS temp dirs (but not job audio already routed to Channels) ---
IA="$SYS_TTS/InitialAudio"
SUB="$SYS_TTS/Subtitles"

if [ -d "$IA" ]; then
  rm -rf "$IA"/.tmp_* 2>/dev/null || true
fi

if [ -d "$SUB" ]; then
  rm -rf "$SUB"/.tmp_* 2>/dev/null || true
fi

# --- Python venv + watcher path ---
VENV="$SYS/venv"
VENV_PY="$VENV/bin/python3"
WATCHER_PY="$SYS_TTS/watcher.py"

if [ ! -x "$VENV_PY" ]; then
  log "ERROR: venv Python not found at $VENV_PY"
  exit 1
fi

if [ ! -f "$WATCHER_PY" ]; then
  log "ERROR: watcher.py not found at $WATCHER_PY"
  exit 1
fi

export VA_ROOT
log "Using venv: $VENV_PY"
log "Starting watcher.py in this terminal (logs will appear below)…"

# If already running, don't start a second watcher
if pgrep -f "$WATCHER_PY" >/dev/null 2>&1; then
  log "watcher.py already running; not starting a second instance."
else
  "$VENV_PY" -u "$WATCHER_PY"
fi

log "StartServices completed. If watcher started, it has now exited."
exit 0