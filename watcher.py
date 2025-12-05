# File: /Users/kazirasel/VideoAutomation/System/SysTTS/watcher.py
# #!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import os, re, sys, time, shutil, datetime, json, wave, audioop
from pathlib import Path
import subprocess  # for ffmpeg post step

# --- project bootstrap ---
VA_ROOT = Path(os.environ.get("VA_ROOT", str(Path.home() / "VideoAutomation")))
SYS     = VA_ROOT / "System"
SYSTTS  = SYS / "SysTTS"
SYSVIS  = SYS / "SysVisuals"
SHARED  = SYSVIS / "Shared"
ENG_TTS = SYSTTS / "EngineTTS"

# --- Central log directory ---
LOG_DIR = SYS / "Logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "watcher.log"

# Paths to the visual one-shot runner (for immediate hand-off after TTS)
VENV_PY = SYS / "venv" / "bin" / "python3"
RUN_VISUAL_ONCE = SYSVIS / "Tools" / "run_visual_once.py"

# LLM sentinel: if this file exists, the LLM layer is disabled and
# higher-level orchestration should stop creating new jobs.
LLM_SENTINEL = SYSVIS / "LLM" / "LLM_DISABLED"

def llm_disabled() -> bool:
    """Return True if the LLM_DISABLED sentinel is present, logging its reason.

    This is a hard guard: when present, the watcher must not synthesize
    new TTS jobs, because the LLM layer is in an invalid or disabled state
    (quota exhausted, billing issue, manual maintenance, etc.).
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
        log(f"⚠ Failed to inspect LLM sentinel: {e}")
    return False

# --- Continuity Controls (module-level, read directly from launcher env) ---
START_SILENCE_MS         = int(os.environ.get("START_SILENCE_MS", "350"))
FADE_MS                  = float(os.environ.get("FADE_MS", "4.0"))
HEAD_TRIM_MS             = float(os.environ.get("HEAD_TRIM_MS", "16.0"))  # trim first few ms to delete embedded pop
TIMEOUT_SEC              = int(os.environ.get("TIMEOUT_SEC", "200"))
TTS_TARGET_RMS           = int(os.environ.get("TTS_TARGET_RMS", "7500"))

# Ensure correct InitialAudio directory under SysTTS
INIT_AUDIO = SYSTTS / "InitialAudio"
INIT_AUDIO.mkdir(parents=True, exist_ok=True)

# Tell the TTS engine where to drop any "initial" wavs (some engines honor this)
os.environ["INITIAL_AUDIO_DIR"] = str(INIT_AUDIO)

# If an old path (VA_ROOT/System/InitialAudio) gets recreated by the engine,
# move those files back under SysTTS/InitialAudio and delete the stray folder.
def _rehoming_initial_audio() -> None:
    legacy = SYS / "InitialAudio"
    try:
        if legacy.exists() and legacy.is_dir():
            for p in legacy.glob("**/*"):
                if p.is_file():
                    dest = INIT_AUDIO / p.name
                    try:
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        p.replace(dest)
                    except Exception:
                        pass
            # try to remove empty legacy dir
            try:
                # remove leftover empty subdirs first
                for d in sorted([d for d in legacy.rglob("*") if d.is_dir()], key=lambda x: len(str(x)), reverse=True):
                    try: d.rmdir()
                    except Exception: pass
                legacy.rmdir()
            except Exception:
                pass
    except Exception:
        pass

for p in (str(SYSTTS), str(ENG_TTS), str(SHARED)):
    if p not in sys.path:
        sys.path.insert(0, p)

import paths
from EngineTTS.gemini_engine import GeminiEngine

PUTSCRIPT = VA_ROOT / "PutScript"
MAX_RETRIES = int(os.environ.get("TTS_MAX_RETRIES", "3"))
LOCK_STALE_SEC = int(os.environ.get("TTS_LOCK_STALE_SEC", "300"))  # 5 minutes
BASE_BACKOFF = float(os.environ.get("TTS_BACKOFF_SEC", "4.0"))

def log(msg: str):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[TTS {timestamp}] {msg}"
    # print to terminal
    print(line, flush=True)
    # append to central log file
    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(line + "\n")
    except Exception:
        pass

def _job_id_for(name: str) -> str:
    return paths.normalize_job_id(name if name.endswith("_final") else f"{name}_final")

def _stale(lock: Path) -> bool:
    try:
        age = time.time() - lock.stat().st_mtime
        return age > LOCK_STALE_SEC
    except Exception:
        return True

def _read_json(p: Path, default: dict) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default

def _write_json(p: Path, data: dict) -> None:
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")

def _already_running_guard() -> bool:
    """Prevent duplicate watcher instances (when user re-launches)."""
    me = Path(sys.argv[0]).name
    try:
        import subprocess, os as _os
        out = subprocess.check_output(["pgrep","-fl", me], text=True)
        lines = [ln for ln in out.splitlines() if "SysTTS/watcher.py" in ln and str(_os.getpid()) not in ln]
        if len(lines) > 1:
            log("watcher already running; exiting.")
            return True
    except Exception:
        pass
    return False

# --- ffmpeg post step to eliminate head-click without removing intended pad ---
def _postprocess_wav(src: Path, opts: dict) -> None:
    if not src.exists():
        return
    sr = int(opts.get("sr", 22050))
    head_s = max(0.0, HEAD_TRIM_MS) / 1000.0

    tmp = src.with_suffix('.tmp.wav')

    # Only trim the very head to remove embedded pop, enforce mono s16 + highpass.
    # All padding, chunk gaps, and end tail are handled in tts_batch to keep timing consistent.
    if head_s > 0.0:
        af = (
            f"atrim=start={head_s:.3f},asetpts=PTS-STARTPTS,"
            f"aformat=sample_fmts=s16:channel_layouts=mono,"
            f"aresample={sr}:first_pts=0,"
            f"highpass=f=20"
        )
    else:
        af = (
            f"asetpts=PTS-STARTPTS,"
            f"aformat=sample_fmts=s16:channel_layouts=mono,"
            f"aresample={sr}:first_pts=0,"
            f"highpass=f=20"
        )

    cmd = ["ffmpeg","-y","-i",str(src),"-af",af,str(tmp)]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        tmp.replace(src)
    except Exception:
        try:
            if tmp.exists(): tmp.unlink()
        except Exception:
            pass
        # non-fatal; keep original

def _normalize_rms(path: Path, target_rms: int) -> None:
    """Normalize mono 16-bit PCM WAV to a target RMS, with a safety cap."""
    try:
        with wave.open(str(path), "rb") as wf:
            params = wf.getparams()
            if params.nchannels != 1 or params.sampwidth != 2:
                return
            audio = wf.readframes(params.nframes)
        rms = audioop.rms(audio, 2)
        if rms <= 0:
            return
        factor = min(6.0, float(target_rms) / float(rms))  # cap to 6x to avoid harsh jumps
        if abs(factor - 1.0) < 0.05:
            return  # already close enough
        audio_norm = audioop.mul(audio, 2, factor)
        with wave.open(str(path), "wb") as wf:
            wf.setparams(params)
            wf.writeframes(audio_norm)
    except Exception:
        # If anything goes wrong, leave the original file unchanged
        pass

def process_all_pending_scripts(engine: GeminiEngine, synth_opts: dict) -> None:
    """Process all pending .txt scripts in PutScript once and return.

    This encapsulates the core TTS + routing logic so it can be used
    both by the continuous watcher loop and by one-shot runners such as
    run_tts_once.py without duplicating code.
    """
    produced_any = False
    # Hard guard: if LLM is disabled, do not process new TTS jobs.
    if llm_disabled():
        log("Stopping TTS processing due to LLM_DISABLED sentinel.")
        return

    _rehoming_initial_audio()
    # purge stale locks and old temp dirs
    for lock in PUTSCRIPT.glob("*.lock"):
        if _stale(lock):
            try:
                lock.unlink()
            except Exception:
                pass
    for tmp in PUTSCRIPT.glob(".tmp_*"):
        try:
            if time.time() - tmp.stat().st_mtime > 1800:
                shutil.rmtree(tmp, ignore_errors=True)
        except Exception:
            pass

    # process .txt files only
    for txt in sorted(PUTSCRIPT.glob("*.txt")):
        base = txt.stem
        lock = txt.with_suffix(".lock")
        fail = txt.with_suffix(".fail")
        meta = txt.with_suffix(".meta.json")

        # respect previous failures until user edits the text
        if fail.exists():
            try:
                if txt.stat().st_mtime <= fail.stat().st_mtime:
                    continue
                else:
                    fail.unlink()
            except Exception:
                pass

        # lock handling
        if lock.exists():
            if _stale(lock):
                try:
                    lock.unlink()
                except Exception:
                    pass
            else:
                continue
        try:
            lock.write_text(str(os.getpid()))
        except Exception:
            continue

        try:
            channel = paths.filename_to_channel(base)
            day = paths.today_str()
            job_id = _job_id_for(base)

            dest_job_name = re.sub(r"_final$", "", job_id)
            wav_dst = paths.audio_dest(channel, day, job_id)
            txt_dst = paths.script_dest(channel, day, dest_job_name)

            # If a final WAV already exists for this job/day, skip TTS and just clean/move script
            if wav_dst.exists():
                try:
                    script_dir = paths.input_script_dir(channel, day)
                    script_dir.mkdir(parents=True, exist_ok=True)
                    if not txt_dst.exists():
                        shutil.move(str(txt), str(txt_dst))
                        log(f"✓ TXT (resume) → {txt_dst.relative_to(paths.VA_ROOT)}")
                    else:
                        # Script already present at destination; drop duplicate from PutScript
                        txt.unlink()
                    log(f"↷ Skipping TTS for {txt.name}, WAV already exists at {wav_dst.relative_to(paths.VA_ROOT)}")
                except Exception as e:
                    log(f"⚠ resume skip failed for {txt.name}: {e}")
                # Ensure we don't proceed with this entry in this round
                continue

            tmp_dir = PUTSCRIPT / f".tmp_{job_id}"
            tmp_dir.mkdir(parents=True, exist_ok=True)

            state = _read_json(meta, {"retries": 0})
            r = int(state.get("retries", 0))

            wav_tmp = tmp_dir / "tts_final.wav"
            text = txt.read_text(encoding="utf-8", errors="ignore")

            try:
                # synthesize with continuity options (no video-side trim later)
                engine.synthesize(text, wav_tmp, options={"job_id": job_id, **synth_opts})

                # ensure no head click
                _postprocess_wav(wav_tmp, synth_opts)

                # normalize loudness so all chunks land at a consistent level
                _normalize_rms(wav_tmp, TTS_TARGET_RMS)

                # route on success (this removes the .txt from PutScript)
                audio_dir = paths.input_audio_dir(channel, day); audio_dir.mkdir(parents=True, exist_ok=True)
                script_dir= paths.input_script_dir(channel, day); script_dir.mkdir(parents=True, exist_ok=True)
                wav_dst = paths.audio_dest(channel, day, job_id)
                txt_dst = paths.script_dest(channel, day, re.sub(r"_final$","", job_id))

                shutil.move(str(wav_tmp), str(wav_dst))
                shutil.move(str(txt), str(txt_dst))
                log(f"✓ WAV → {wav_dst.relative_to(paths.VA_ROOT)}")
                log(f"✓ TXT → {txt_dst.relative_to(paths.VA_ROOT)}")
                produced_any = True

                # clean state
                try:
                    meta.unlink()
                except Exception:
                    pass
                shutil.rmtree(tmp_dir, ignore_errors=True)

            except Exception as e:
                r += 1
                state["retries"] = r
                _write_json(meta, state)
                if r >= MAX_RETRIES:
                    fail.write_text(f"{datetime.datetime.now().isoformat()} :: {e}\n", encoding="utf-8")
                    log(f"✗ max retries for {txt.name}. Reason: {e}  (see {fail.name})")
                else:
                    backoff = min(BASE_BACKOFF * (2 ** (r - 1)), 60.0)
                    log(f"⚠ TTS fail {txt.name}: {e}  (retry {r}/{MAX_RETRIES} after {int(backoff)}s)")
                    time.sleep(backoff)

            finally:
                try:
                    if tmp_dir.exists() and not any(tmp_dir.iterdir()):
                        tmp_dir.rmdir()
                except Exception:
                    pass

        finally:
            try:
                lock.unlink()
            except Exception:
                pass

    # If at least one new job was synthesized and routed into channel Input,
    # immediately trigger a single visual pipeline round so scenes/video/import
    # run right after TTS+Whisper, without waiting for a timer.
    if produced_any and VENV_PY.exists() and RUN_VISUAL_ONCE.exists():
        try:
            subprocess.Popen([str(VENV_PY), str(RUN_VISUAL_ONCE)])
            log("🔔 Triggered run_visual_once.py for new job(s).")
        except Exception as e:
            log(f"⚠ Failed to trigger run_visual_once.py: {e}")


# --- Helper to check if TTS worker is running ---
def _tts_worker_running() -> bool:
    """Return True if a TTS worker (run_tts_once.py) is already running.

    This prevents the watcher from spawning multiple overlapping TTS workers
    when many .txt files arrive in a short window.
    """
    try:
        out = subprocess.check_output(["pgrep", "-fl", "run_tts_once.py"], text=True)
        lines = [ln for ln in out.splitlines() if "run_tts_once.py" in ln]
        return len(lines) > 0
    except Exception:
        return False

def main():
    if _already_running_guard():
        return

    PUTSCRIPT.mkdir(parents=True, exist_ok=True)
    log(f"watching {PUTSCRIPT}")
    _rehoming_initial_audio()

    # Path to the one-shot TTS worker script
    tts_worker = SYSTTS / "run_tts_once.py"

    while True:
        try:
            # If LLM is disabled, do not spawn any new TTS workers.
            if llm_disabled():
                time.sleep(3.0)
                continue

            # Check if there are any .txt scripts waiting in PutScript.
            has_txt = any(PUTSCRIPT.glob("*.txt"))

            # Only spawn a TTS worker if there is pending work and no worker is already running.
            if has_txt and not _tts_worker_running():
                if VENV_PY.exists() and tts_worker.exists():
                    subprocess.Popen([str(VENV_PY), str(tts_worker)])
                    log("🔔 Triggered run_tts_once.py for pending script(s).")

        except KeyboardInterrupt:
            log("bye."); break
        except Exception as e:
            log(f"⚠ loop: {e}")
            time.sleep(2)

        # Prevent busy-looping when there is no work; keep CPU usage low when idle.
        time.sleep(0.5)

if __name__ == "__main__":
    main()
