#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os, json, sqlite3, hashlib, random
import datetime
from pathlib import Path
from PIL import Image
from array import array
from typing import Optional, Iterable, Dict, Any

VA_ROOT = Path(os.environ.get("VA_ROOT", str(Path.home() / "VideoAutomation")))
SYS     = VA_ROOT / "System"
LOG_DIR = SYS / "Logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "visual_library.log"


def _log(msg: str) -> None:
    """Log Visual Library events to terminal and central log file."""
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[VL {timestamp}] {msg}"
    # print to terminal
    print(line, flush=True)
    # append to central log file
    try:
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def vlib_init(vdir: Path):
    """Initialise the visual library in the given directory.

    Returns a small handle dict containing:
      - "con": sqlite3.Connection
      - "phash_of": callable(Path) -> str
      - "luma_of": callable(Path) -> Optional[float]
      - "clip_to_blob": callable(iterable[float]) -> bytes
      - "blob_to_clip": callable(bytes) -> list[float]
      - "root": Path to the VisualLibrary root directory
    """
    vdir.mkdir(parents=True, exist_ok=True)
    db = vdir / "library.sqlite"
    con = sqlite3.connect(str(db))
    cur = con.cursor()

    # base table (keeps backwards compatibility with existing DBs)
    cur.execute(
        """
        create table if not exists images(
            id integer primary key autoincrement,
            path text unique,
            prompt text,
            channel text,
            job text,
            day text,
            phash text
        )
        """
    )
    cur.execute("create index if not exists idx_phash on images(phash)")

    # ensure new columns exist for extended metadata (backwards compatible)
    cur.execute("PRAGMA table_info(images)")
    cols = {row[1] for row in cur.fetchall()}
    for col, ddl in [
        ("luma", "REAL"),
        ("width", "INTEGER"),
        ("height", "INTEGER"),
        ("clip", "BLOB"),
        ("label", "TEXT"),
        ("topic", "TEXT"),
        ("role", "TEXT"),
        ("concept_id", "TEXT"),
    ]:
        if col not in cols:
            cur.execute(f"ALTER TABLE images ADD COLUMN {col} {ddl}")

    con.commit()

    def phash_of(img_path: Path) -> str:
        """Simple perceptual hash (average hash)."""
        with Image.open(img_path) as im:
            im = im.convert("L").resize((8, 8), Image.BICUBIC)
            data = list(im.getdata())
            if not data:
                return "0" * 16
            avg = sum(data) / float(len(data))
            bits = "".join("1" if p > avg else "0" for p in data)
            return hex(int(bits, 2))[2:].rjust(16, "0")

    def luma_of(img_path: Path) -> Optional[float]:
        """Average brightness 0–255, or None on error."""
        try:
            with Image.open(img_path) as im:
                im = im.convert("L")
                data = list(im.getdata())
                return float(sum(data)) / float(len(data)) if data else None
        except Exception:
            return None

    def clip_to_blob(vec: Optional[Iterable[float]]) -> Optional[bytes]:
        """Pack a CLIP vector into a float32 blob for SQLite.

        Returns None if vec is None or cannot be converted.
        """
        if vec is None:
            return None
        if isinstance(vec, (bytes, bytearray)):
            return bytes(vec)
        try:
            arr = array("f", [float(x) for x in vec])
            return arr.tobytes()
        except Exception:
            return None

    def blob_to_clip(blob: Optional[bytes]) -> Optional[list[float]]:
        """Decode a float32 blob back into a Python list of floats."""
        if not blob:
            return None
        try:
            arr = array("f")
            arr.frombytes(blob)
            return list(arr)
        except Exception:
            return None

    return {
        "con": con,
        "phash_of": phash_of,
        "luma_of": luma_of,
        "clip_to_blob": clip_to_blob,
        "blob_to_clip": blob_to_clip,
        "root": vdir,
    }


def vlib_store(
    vdb,
    path: Path,
    prompt: str,
    channel: str,
    job: str,
    day: str,
    *,
    label: Optional[str] = None,
    luma: Optional[float] = None,
    clip: Optional[Iterable[float]] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    topic: Optional[str] = None,
    role: Optional[str] = None,
    concept_id: Optional[str] = None,
) -> None:
    """Store a single image and its metadata in the visual library.

    The first six positional arguments are kept for backwards compatibility
    with existing callers. Extra metadata, including semantic metadata
    (topic, role, concept_id), should be passed by keyword.
    """
    img_path = Path(path)
    ph = vdb["phash_of"](img_path)

    # lazily compute luma and size if not provided
    if luma is None or width is None or height is None:
        try:
            with Image.open(img_path) as im:
                if luma is None:
                    g = im.convert("L")
                    data = list(g.getdata())
                    luma = float(sum(data)) / float(len(data)) if data else None
                if width is None or height is None:
                    width, height = im.size
        except Exception:
            pass

    clip_blob = vdb.get("clip_to_blob", lambda v: None)(clip)

    cur = vdb["con"].cursor()
    try:
        cur.execute(
            """
            insert or ignore into images(
                path,prompt,channel,job,day,phash,luma,width,height,clip,label,topic,role,concept_id
            )
            values(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                str(img_path),
                prompt,
                channel,
                job,
                day,
                ph,
                luma,
                width,
                height,
                clip_blob,
                label,
                topic,
                role,
                concept_id,
            ),
        )
        vdb["con"].commit()
    except Exception:
        # on any error we fail silently to avoid disrupting the main pipeline
        pass


def hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def vlib_search(
    vdb,
    prompt: str,
    topk: int = 5,
    min_phash_dist: int = 4,
    exclude_hashes: Optional[set[str]] = None,
    *,
    channel: Optional[str] = None,
    label: Optional[str] = None,
    topic: Optional[str] = None,
    role: Optional[str] = None,
    concept_id: Optional[str] = None,
    prompt_clip: Optional[Iterable[float]] = None,
    min_clip: float = 0.0,
    max_candidates: int = 500,
) -> Optional[dict]:
    """Search for a reusable image.

    Backwards compatible behaviour:
      - If `prompt_clip` is None, behaves like the old implementation:
        returns the most recent image that isn't too-close by phash.

    Extended behaviour (when `prompt_clip` is provided):
      - Optionally filters by channel/label/topic/role/concept_id.
      - By default, loads at most `max_candidates` recent rows.
      - Computes cosine similarity using stored CLIP embeddings.
      - Applies a CLIP threshold (`min_clip`).
      - Applies phash diversity vs `exclude_hashes`.
      - Returns the best match as a dict: {"path", "phash", "score"}.

    Full-scan mode:
      - If `max_candidates <= 0` and `prompt_clip` is provided, scans the
        entire candidate set for the given channel/label/topic/role/concept_id.
      - Starts from a random offset and wraps around so that each image
        is checked at most once per call, but starting point changes
        between calls for fairness.
      - Returns the first candidate that passes `min_clip` and phash diversity.

    Note:
      - `topic`, `role`, and `concept_id` can be used to restrict reuse
        to semantically-matching images, but they are all optional for
        backwards compatibility.
    """
    cur = vdb["con"].cursor()


    # CLIP-aware path
    # Build a simple SQL where clause for channel/label/topic/role/concept_id filtering.
    where_clauses: list[str] = []
    params: list[object] = []
    if channel:
        where_clauses.append("channel = ?")
        params.append(channel)
    if label:
        where_clauses.append("label = ?")
        params.append(label)
    if topic:
        where_clauses.append("topic = ?")
        params.append(topic)
    if role:
        where_clauses.append("role = ?")
        params.append(role)
    if concept_id:
        where_clauses.append("concept_id = ?")
        params.append(concept_id)
    where_sql = " WHERE " + " AND ".join(where_clauses) if where_clauses else ""

    # Normalize prompt CLIP vector once
    import math

    def _norm(vec: list[float]) -> float:
        return math.sqrt(sum(v * v for v in vec)) or 1.0

    prompt_vec = [float(x) for x in prompt_clip]
    prompt_norm = _norm(prompt_vec)

    blob_to_clip = vdb.get("blob_to_clip", lambda b: None)

    # Helper for phash/clip filtering on a single candidate
    def _candidate_ok(ph: str, clip_blob: Optional[bytes]) -> Optional[tuple[float]]:
        if exclude_hashes and ph in exclude_hashes:
            return None
        # phash diversity first (cheap)
        for e in (exclude_hashes or []):
            if hamming(ph, e) < min_phash_dist:
                return None
        clip_vec = blob_to_clip(clip_blob)
        if not clip_vec:
            return None
        if len(clip_vec) != len(prompt_vec):
            return None
        dot = sum(a * b for a, b in zip(prompt_vec, clip_vec))
        denom = prompt_norm * _norm(clip_vec)
        sim = dot / denom if denom else 0.0
        if sim < min_clip:
            return None
        return (sim,)

    # If full_scan is requested, scan the entire candidate set in a randomised order
    if prompt_clip is not None and max_candidates <= 0:
        full_scan = True
    else:
        full_scan = False

    if full_scan:
        sql = f"select path,phash,clip,label from images{where_sql} order by id asc"
        cur.execute(sql, params)
        rows = cur.fetchall()
        if not rows:
            return None
        n = len(rows)
        start = random.randrange(n)
        _log(f"full-scan: {len(rows)} candidates for channel={channel} label={label}")
        _log(f"full-scan start index: {start}")
        order = list(range(start, n)) + list(range(0, start))
        for idx in order:
            p, ph, clip_blob, lbl = rows[idx]
            cand = _candidate_ok(ph, clip_blob)
            if cand is None:
                # Optional details: phash or CLIP failure
                _log(f"reject: path={p}, phash={ph}")
                continue
            sim = cand[0]
            _log(f"accept: path={p}, score={sim}, label={lbl}")
            return {"path": p, "phash": ph, "score": sim, "label": lbl}
        return None


# Helper: find the most recent image matching a given concept_id (and optional channel)
def vlib_find_by_concept(
    vdb,
    concept_id: str,
    channel: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Find the most recent image matching a given concept_id (and optional channel).

    Returns a small dict with at least keys: "path", "phash", "label".
    If no match is found, returns None.
    """
    if not concept_id:
        return None
    cur = vdb["con"].cursor()
    where_clauses = ["concept_id = ?"]
    params: list[object] = [concept_id]
    if channel:
        where_clauses.append("channel = ?")
        params.append(channel)
    where_sql = " WHERE " + " AND ".join(where_clauses)
    sql = f"select path,phash,label,id from images{where_sql} order by id desc limit 1"
    try:
        cur.execute(sql, params)
        row = cur.fetchone()
    except Exception:
        return None
    if not row:
        return None
    path, phash, label, _ = row
    return {"path": path, "phash": phash, "label": label}


def _entity_from_concept_id(concept_id: str) -> str:
    """Extract the leading entity part from a concept_id.

    Concept ids have the form:
        entity|motif|era|topic|hxxxx

    This helper returns the 'entity' token, e.g. 'person:napoleon_bonaparte'.
    If the concept_id is malformed, returns an empty string.
    """
    if not concept_id:
        return ""
    parts = str(concept_id).split("|", 1)
    if not parts:
        return ""
    return parts[0] or ""


def vlib_find_entity_anchor(
    vdb,
    entity_key: str,
    channel: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Find a hero/anchor image for a given entity key (e.g. 'person:napoleon_bonaparte').

    This is used to keep a consistent face or building across scenes/jobs.

    Strategy:
      - Search for the most recent image whose concept_id starts with
        'entity_key|portrait|...' (hero portrait motif).
      - If none is found, optionally fall back to 'entity_key|hook_tableau|...'
        (hero tableau motif).
      - Optionally restrict by channel.

    Returns:
      A dict with at least: {'path', 'phash', 'label'}, or None if not found.
    """
    key = (entity_key or "").strip()
    if not key:
        return None

    cur = vdb["con"].cursor()

    def _query_for_motif(motif: str) -> Optional[Dict[str, Any]]:
        where_clauses = ["concept_id LIKE ?"]
        params: list[object] = [f"{key}|{motif}|%"]
        if channel:
            where_clauses.append("channel = ?")
            params.append(channel)
        where_sql = " WHERE " + " AND ".join(where_clauses)
        sql = f"select path,phash,label,id from images{where_sql} order by id desc limit 1"
        try:
            cur.execute(sql, params)
            row = cur.fetchone()
        except Exception:
            return None
        if not row:
            return None
        path, phash, label, _ = row
        return {"path": path, "phash": phash, "label": label}

    # Prefer explicit portrait motif
    hit = _query_for_motif("portrait")
    if hit:
        return hit

    # Fallback: use hook_tableau motif if available
    hit = _query_for_motif("hook_tableau")
    if hit:
        return hit

    return None