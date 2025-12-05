# File: /Users/kazirasel/VideoAutomation/System/SysTTS/tts_batch.py
#!/usr/bin/env python3
# One-off TTS for a single .txt using the same path as watcher (parity input)
import sys, pathlib, os, wave, datetime, time, re, math, struct, subprocess
from typing import List
from google.cloud import texttospeech as tts

ROOT     = pathlib.Path(__file__).resolve().parents[1]   # …/System
INIT_DIR = ROOT / "SysTTS" / "InitialAudio"
SUBTITLES_DIR = ROOT / "SysTTS" / "Subtitles"
FINALAUDIO_ENV = os.environ.get("FINALAUDIO", "").strip()

# --- Central logging setup ---
LOG_DIR = ROOT / "Logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "tts_batch.log"

VOICE_NAME    = os.environ.get("VOICE_NAME", "Enceladus")
LANGUAGE_CODE = os.environ.get("LANGUAGE_CODE", "en-US")
MODEL_NAME    = (
    os.environ.get("MODEL_NAME")
    or os.environ.get("MODEL_PRIMARY")
    or "gemini-2.5-pro-tts"
)
SPEAKING_RATE = float(os.environ.get("SPEAKING_RATE", "1.035"))
PITCH         = float(os.environ.get("PITCH", "0.020"))
STYLE_PROMPT  = os.environ.get("STYLE_PROMPT", "")

BYTE_CAP                 = int(os.environ.get("BYTE_CAP", "3000"))
INCLUDE_STYLE_EACH_CHUNK = os.environ.get("INCLUDE_STYLE_EACH_CHUNK", "1") == "1"
PAD_FIRST_CHUNK_ONLY     = os.environ.get("PAD_FIRST_CHUNK_ONLY", "1") == "1"
START_SILENCE_MS         = int(os.environ.get("START_SILENCE_MS", "520"))
FADE_MS                  = float(os.environ.get("FADE_MS", "6.0"))
CHUNK_FADE_OUT_MS        = float(os.environ.get("CHUNK_FADE_OUT_MS", "10.0"))
CROSSFADE_MS             = float(os.environ.get("CROSSFADE_MS", "0.0"))
END_SILENCE_MS           = int(os.environ.get("END_SILENCE_MS", "600"))
CHUNK_GAP_MS             = int(os.environ.get("CHUNK_GAP_MS", "450"))
TIMEOUT_SEC              = int(os.environ.get("TIMEOUT_SEC", "200"))
TTS_TARGET_RMS           = int(os.environ.get("TTS_TARGET_RMS", "7500"))


def tlog(msg: str) -> None:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[TTS {timestamp}] {msg}"
    # print to terminal
    print(line, flush=True)
    # append to central log file
    try:
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def normalize_rms(path: pathlib.Path, target_rms: int = TTS_TARGET_RMS):
    try:
        with wave.open(str(path), "rb") as wf:
            params = wf.getparams()
            audio = wf.readframes(wf.getnframes())
        import audioop
        rms = audioop.rms(audio, 2)
        if rms > 0:
            factor = min(4.0, float(target_rms) / float(rms))
            audio = audioop.mul(audio, 2, factor)
        with wave.open(str(path), "wb") as wf:
            wf.setparams(params)
            wf.writeframes(audio)
    except Exception:
        pass


def normalize_text(s: str) -> str:
    return s.replace("\r\n", "\n").replace("\r", "\n").strip()


PARA_SPLIT = re.compile(r"\n\s*\n+")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


def split_sentences(block: str):
    items = [t.strip() for t in SENT_SPLIT.split(block) if t.strip()]
    return items if items else ([block.strip()] if block.strip() else [])


def pack_paragraphs_byte_safe(text: str, include_style_first: bool, byte_cap: int):
    chunks = []
    paras = [p.strip() for p in PARA_SPLIT.split(text) if p.strip()]
    cur = ""
    is_first = True
    style_overhead = len((STYLE_PROMPT + "\n\n").encode("utf-8")) if (STYLE_PROMPT and include_style_first) else 0

    def fits(s: str, first: bool) -> bool:
        return len(s.encode("utf-8")) + (style_overhead if (first and STYLE_PROMPT) else 0) <= byte_cap

    for para in paras:
        test = (cur + ("\n\n" if cur else "") + para) if cur else para
        if fits(test, is_first):
            cur = test
        else:
            if cur:
                chunks.append(cur)
                is_first = False
                cur = ""
            sent_chunk = ""
            for sent in split_sentences(para):
                cand = (sent_chunk + (" " if sent_chunk else "") + sent) if sent_chunk else sent
                if fits(cand, is_first):
                    sent_chunk = cand
                else:
                    if sent_chunk:
                        chunks.append(sent_chunk)
                        is_first = False
                        sent_chunk = ""
                    words = sent.split()
                    word_chunk = ""
                    for w in words:
                        cand2 = (word_chunk + (" " if word_chunk else "") + w) if word_chunk else w
                        if fits(cand2, is_first):
                            word_chunk = cand2
                        else:
                            if word_chunk:
                                chunks.append(word_chunk)
                                is_first = False
                                word_chunk = w
                            else:
                                word_chunk = w
                    if word_chunk:
                        chunks.append(word_chunk)
                        is_first = False
                        word_chunk = ""
            if sent_chunk:
                chunks.append(sent_chunk)
                is_first = False
    if cur:
        chunks.append(cur)
    return chunks


def spoken_input_for_chunk(text: str, idx: int) -> str:
    if INCLUDE_STYLE_EACH_CHUNK and STYLE_PROMPT:
        return f"{STYLE_PROMPT}\n\n{text}"
    if idx == 1 and STYLE_PROMPT:
        return f"{STYLE_PROMPT}\n\n{text}"
    return text


def tts_call(text: str) -> bytes:
    client = tts.TextToSpeechClient()
    voice = tts.VoiceSelectionParams(
        language_code=LANGUAGE_CODE,
        name=VOICE_NAME,
        model_name=MODEL_NAME,
    )
    audio_cfg = tts.AudioConfig(
        audio_encoding=tts.AudioEncoding.LINEAR16,
        sample_rate_hertz=24000,
        speaking_rate=SPEAKING_RATE,
        pitch=PITCH,
    )
    resp = client.synthesize_speech(
        input=tts.SynthesisInput(text=text),
        voice=voice,
        audio_config=audio_cfg,
        timeout=TIMEOUT_SEC,
    )
    return resp.audio_content


def normalize_pcm_rms(pcm: bytes, target_rms: int = 2000) -> bytes:
    try:
        import audioop
        rms = audioop.rms(pcm, 2)
        if rms <= 0:
            return pcm
        factor = min(4.0, float(target_rms) / float(rms))
        if abs(factor - 1.0) < 0.05:
            return pcm
        return audioop.mul(pcm, 2, factor)
    except Exception:
        return pcm


def write_wav(pcm: bytes, path: pathlib.Path, pad_ms: int, fade_ms: float):
    sr = 24000
    pad_samples = int(sr * (pad_ms / 1000.0))
    fade_in_samples = max(1, int(sr * (fade_ms / 1000.0)))

    total_samples = len(pcm) // 2
    if total_samples <= 0:
        total_samples = 1
        pcm = b"\x00\x00"

    out = bytearray()

    if pad_samples > 0:
        out.extend(b"\x00\x00" * pad_samples)

    for i in range(total_samples):
        sample = struct.unpack_from("<h", pcm, i * 2)[0]
        if i < fade_in_samples:
            x = i / float(fade_in_samples)
            scale_in = 0.5 - 0.5 * math.cos(math.pi * x)
        else:
            scale_in = 1.0
        sample = int(sample * scale_in)
        out.extend(struct.pack("<h", sample))

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(out)


def simple_merge_wavs(parts: List[pathlib.Path], out_path: pathlib.Path,
                      gap_ms: int, end_silence_ms: int, sr: int = 24000) -> None:
    import wave

    full = bytearray()
    gap_samples  = int(sr * (gap_ms / 1000.0))
    tail_samples = int(sr * (end_silence_ms / 1000.0))

    params = None
    for idx, part in enumerate(parts):
        with wave.open(str(part), "rb") as wf:
            p = wf.getparams()
            audio = wf.readframes(wf.getnframes())
        if params is None:
            params = p
        else:
            if (p.nchannels, p.sampwidth, p.framerate, p.comptype, p.compname) != (
                params.nchannels, params.sampwidth, params.framerate, params.comptype, params.compname
            ):
                tlog("[merge] ⚠ WAV format mismatch; using first chunk format.")

        full.extend(audio)
        if idx < len(parts) - 1 and gap_samples > 0:
            full.extend(b"\x00\x00" * gap_samples)

    if tail_samples > 0:
        full.extend(b"\x00\x00" * tail_samples)

    if params is None:
        return

    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(params.nchannels)
        wf.setsampwidth(params.sampwidth)
        wf.setframerate(params.framerate)
        wf.writeframes(full)


def add_tail_silence(path: pathlib.Path, tail_ms: int, sr: int = 24000) -> None:
    if tail_ms <= 0:
        return
    try:
        with wave.open(str(path), "rb") as wf:
            params = wf.getparams()
            audio = wf.readframes(wf.getnframes())
        extra_samples = int(sr * (tail_ms / 1000.0))
        if extra_samples <= 0:
            return
        silence = b"\x00\x00" * extra_samples
        with wave.open(str(path), "wb") as wf:
            wf.setparams(params)
            wf.writeframes(audio + silence)
    except Exception:
        pass


def main(txt_path: str):
    p = pathlib.Path(txt_path).resolve()
    if not p.exists():
        tlog(f"❌ No such file: {p}")
        sys.exit(1)

    base = p.stem
    day  = datetime.datetime.now().strftime("%Y-%m-%d")

    parts_dir = INIT_DIR / day / base
    parts_dir.mkdir(parents=True, exist_ok=True)

    sub_dir = SUBTITLES_DIR / day
    sub_dir.mkdir(parents=True, exist_ok=True)

    text  = normalize_text(p.read_text(encoding="utf-8"))
    parts = pack_paragraphs_byte_safe(text, include_style_first=True, byte_cap=BYTE_CAP)

    tlog(f"batch: {base} → {len(parts)} chunk(s) (BYTE_CAP={BYTE_CAP})")

    wavs: List[pathlib.Path] = []
    chunk_durations: List[float] = []

    for i, ch in enumerate(parts, 1):
        wavp = parts_dir / f"{base}__part_{i:03d}.wav"
        spoken = spoken_input_for_chunk(ch, i)
        tlog(f"[{i}/{len(parts)}] Processing…")
        start = time.time()
        pcm = tts_call(spoken)
        pcm = normalize_pcm_rms(pcm, target_rms=TTS_TARGET_RMS)
        pad = START_SILENCE_MS if (i == 1 and PAD_FIRST_CHUNK_ONLY) else 0
        write_wav(pcm, wavp, pad_ms=pad, fade_ms=FADE_MS)
        elapsed = time.time() - start
        tlog(f"[{i}/{len(parts)}] ✓ done in {elapsed:.1f}s")
        wavs.append(wavp)

        try:
            with wave.open(str(wavp), "rb") as wf:
                dur = wf.getnframes() / float(wf.getframerate())
        except Exception:
            dur = 0.0
        chunk_durations.append(dur)

    if FINALAUDIO_ENV:
        final_dir = pathlib.Path(FINALAUDIO_ENV)
    else:
        final_dir = parts_dir
    final_dir.mkdir(parents=True, exist_ok=True)
    out = final_dir / f"{base}_final.wav"
    if len(wavs) == 1:
        out.write_bytes(wavs[0].read_bytes())
        add_tail_silence(out, END_SILENCE_MS)
    else:
        try:
            simple_merge_wavs(
                wavs,
                out,
                gap_ms=CHUNK_GAP_MS,
                end_silence_ms=END_SILENCE_MS,
                sr=24000,
            )
        except AssertionError as e:
            tlog(f"[merge] ⚠ simple_merge_wavs AssertionError ignored: {e}")
            import wave as _wave
            sr = 24000
            gap_samples = int(sr * (CHUNK_GAP_MS / 1000.0))
            tail_samples = int(sr * (END_SILENCE_MS / 1000.0))
            full = bytearray()
            params = None
            for idx, part in enumerate(wavs):
                with _wave.open(str(part), "rb") as wf:
                    p = wf.getparams()
                    audio = wf.readframes(wf.getnframes())
                if params is None:
                    params = p
                full.extend(audio)
                if idx < len(wavs) - 1 and gap_samples > 0:
                    full.extend(b"\x00\x00" * gap_samples)
            if tail_samples > 0:
                full.extend(b"\x00\x00" * tail_samples)
            if params is not None:
                with _wave.open(str(out), "wb") as wf_out:
                    wf_out.setnchannels(params.nchannels)
                    wf_out.setsampwidth(params.sampswidth)
                    wf_out.setframerate(params.framerate)
                    wf_out.writeframes(full)

    normalize_rms(out, target_rms=TTS_TARGET_RMS)
    tlog(f"✅ Saved → {out}")

    # --- precise word timings via Whisper (external whisper-env) ---
    try:
        whisper_env   = ROOT / "whisper-env"
        whisper_py    = whisper_env / "bin" / "python3"
        whisper_script = ROOT / "SysTTS" / "whisper_transcribe.py"

        if whisper_py.exists() and whisper_script.exists():
            whisper_json = sub_dir / f"{base}_whisper.json"
            cmd = [
                str(whisper_py),
                str(whisper_script),
                str(out),
                str(whisper_json),
            ]
            tlog(f"[WHISPER] running ({out.name} → {whisper_json.name})")
            subprocess.run(cmd, check=True)
        else:
            tlog("[WHISPER] env or script missing; skipping Whisper timings")
    except Exception as e:
        tlog(f"⚠ [WHISPER] error: {e}")


def main_entry():
    if len(sys.argv) != 2:
        tlog("Usage: tts_batch.py path/to/file.txt")
        sys.exit(2)
    main(sys.argv[1])


if __name__ == "__main__":
    main_entry()