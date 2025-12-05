# File: /Users/kazirasel/VideoAutomation/System/SysVisuals/Shared/prompt_style.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import re
import hashlib
import importlib.util
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from dataclasses import dataclass, field


import json
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from dataclasses import dataclass, field

# reuse your existing paths helper
import paths

VA_ROOT = Path(os.environ.get("VA_ROOT", str(Path.home() / "VideoAutomation")))
PROMPT_FILE_NAME = "prompt_style.json"


# -------------------------- Semantic Heuristic API -------------------------

@dataclass
class JobProfile:
    """
    Semantic job profile for a video/story, as inferred from text and channel style.
    Used by the semantic pipeline for prompt, role, and concept selection.
    """
    job_id: str
    channel: str
    dominant_topics: List[str] = field(default_factory=list)
    key_people: List[str] = field(default_factory=list)
    key_companies: List[str] = field(default_factory=list)
    key_places: List[str] = field(default_factory=list)
    timeframe: str = "MODERN"
    overall_mood: str = "neutral"


def _style_path_for_channel(channel: str) -> Path:
    """
    Channels/<Channel>/Config/prompt_style.json
    e.g. Channels/CapitalChronicles/Config/prompt_style.json
    """
    return paths.channel_root(channel) / "Config" / PROMPT_FILE_NAME


def load_prompt_style(channel: str) -> Dict[str, str]:
    """
    Load per-channel style JSON.
    Returns {} if file is missing or invalid.
    """
    p = _style_path_for_channel(channel)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def apply_prompt_style_env(channel: str) -> None:
    """
    Read Channels/<channel>/Config/prompt_style.json and push style fields
    into generic environment variables used by the active AI image engine.
    """
    style = load_prompt_style(channel)
    if not style:
        return

    # Prompt prefix/suffix
    prefix = (style.get("prompt_prefix") or "").strip()
    suffix = (style.get("prompt_suffix") or "").strip()
    if prefix:
        os.environ["IMG_PROMPT_PREFIX"] = prefix
    if suffix:
        os.environ["IMG_PROMPT_SUFFIX"] = suffix

    # Negative prompt
    neg = (style.get("negative_prompt") or "").strip()
    if neg:
        os.environ["IMG_NEGATIVE"] = neg


# ---------------------- Semantic API helpers -------------------------------

def _load_channel_style(channel: str) -> Dict[str, Any]:
    """
    Semantic API: Loads the channel's prompt style config.
    """
    return load_prompt_style(channel)


# ------------------- Channel Policy Loader ---------------------------------
def load_channel_policy(channel: str):
    """Dynamically load a channel-specific policy module if present.

    Expected location: Channels/<Channel>/Config/channel_policy.py
    Returns the imported module or None if not found or failed.
    """
    try:
        cfg_dir = paths.channel_root(channel) / "Config"
        policy_path = cfg_dir / "channel_policy.py"
        if not policy_path.exists():
            return None
        spec = importlib.util.spec_from_file_location(f"{channel}_policy", policy_path)
        if not spec or not spec.loader:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[arg-type]
        return mod
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Advanced prompt builder (per-channel prompt_style.json)
# ---------------------------------------------------------------------------

def _simplify_slice(text: str, max_len: int = 160) -> str:
    """Clean and shorten slice text for use inside templates.

    Keeps whitespace tidy and trims to ~max_len, trying to cut at
    sentence/phrase boundaries when possible.
    """
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if len(cleaned) <= max_len:
        return cleaned

    cut = cleaned[:max_len]
    last_stop = max(cut.rfind("."), cut.rfind(","), cut.rfind(";"))
    if last_stop > 40:
        return cut[:last_stop].strip()
    return cut.strip()


def _choose_template(style_cfg: Dict[str, Any], slice_summary: str) -> str:
    """Choose a template name based on rules in style_cfg.

    Expects the new structured prompt_style.json with keys:
      - "rules": list of {id, priority, any_keywords, template}
      - "fallback_template": name of fallback template
    """
    text_l = (slice_summary or "").lower()
    rules = style_cfg.get("rules", [])
    # sort by priority (lower first), defaulting to a large number
    rules_sorted = sorted(rules, key=lambda r: r.get("priority", 9999))
    for rule in rules_sorted:
        for kw in rule.get("any_keywords", []) or []:
            if kw.lower() in text_l:
                return rule.get("template") or style_cfg.get("fallback_template", "GENERIC")
    return style_cfg.get("fallback_template", "GENERIC")


def build_prompt(style_cfg: Dict[str, Any], slice_text: str) -> tuple[str, str]:
    """Build (positive, negative) prompt from a style config and slice text.

    Supports both:
      - NEW structured style (with "global", "rules", "templates"); and
      - LEGACY flat style ("prompt_prefix", "prompt_suffix", "negative_prompt").
    """
    # NEW structured style
    if "global" in style_cfg and "templates" in style_cfg:
        global_cfg = style_cfg["global"]
        templates = style_cfg["templates"]

        slice_summary = _simplify_slice(slice_text)
        template_name = _choose_template(style_cfg, slice_summary)
        template_str = templates.get(template_name, templates.get(style_cfg.get("fallback_template", "GENERIC"), "{slice_summary}"))

        body = template_str.replace("{slice_summary}", slice_summary)

        prefix = global_cfg.get("prompt_prefix", "").strip()
        suffix = global_cfg.get("prompt_suffix", "").strip()
        negative = global_cfg.get("negative_prompt", "").strip()

        positive = f"{prefix} {body} {suffix}".strip()
        return positive, negative

    # LEGACY flat style fallback
    prefix = (style_cfg.get("prompt_prefix") or "").strip()
    suffix = (style_cfg.get("prompt_suffix") or "").strip()
    negative = (style_cfg.get("negative_prompt") or "").strip()
    positive = f"{prefix} {slice_text or ''} {suffix}".strip()
    return positive, negative


def build_scene_prompt(channel: str, slice_text: str) -> tuple[str, str]:
    """Public API: build (positive, negative) for a channel + slice.

    This does NOT touch environment variables; it only returns
    ready-to-use strings for whatever AI provider is active.
    """
    style_cfg = load_prompt_style(channel)
    if not style_cfg:
        # no style configured: raw text, no negative prompt
        return (slice_text or "", "")
    return build_prompt(style_cfg, slice_text)


# =================== New Semantic Heuristic Helpers ========================

def _is_semantic_style(style_cfg: Dict[str, Any]) -> bool:
    """
    Returns True if the config appears to be a semantic prompt style (has "style" and "templates").
    """
    return isinstance(style_cfg, dict) and "style" in style_cfg and "templates" in style_cfg


def _topic_scores(full_text: str, style_cfg: Dict[str, Any]) -> Dict[str, int]:
    """
    Heuristic: Counts occurrences of each topic's keywords in the full_text.
    Returns a dict of topic -> count.
    """
    scores = {}
    text_l = (full_text or "").lower()
    topics = style_cfg.get("topics", {})
    for topic, info in topics.items():
        keywords = info.get("keywords", [])
        count = 0
        for kw in keywords:
            if not kw:
                continue
            # count all occurrences
            count += text_l.count(kw.lower())
        scores[topic] = count
    return scores


def analyse_job(full_text: str, channel: str, job_id: str) -> JobProfile:
    """
    Analyses the full text and channel style to infer a JobProfile.
    Heuristic and optional: used for semantic prompt pipelines.
    """
    style_cfg = _load_channel_style(channel)
    if not _is_semantic_style(style_cfg):
        return JobProfile(job_id=job_id, channel=channel)

    # Dominant topics
    topic_counts = _topic_scores(full_text, style_cfg)
    sorted_topics = sorted((k for k in topic_counts if topic_counts[k] > 0), key=lambda k: -topic_counts[k])
    dominant_topics = sorted_topics[:2]

    # Key people: capitalized two-word sequences
    key_people = []
    for match in re.finditer(r"\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\b", full_text):
        person = f"{match.group(1)} {match.group(2)}"
        if person not in key_people:
            key_people.append(person)
        if len(key_people) >= 8:
            break

    # Key companies: ALLCAPS or CamelCase tokens (simplified)
    key_companies = []
    for match in re.finditer(r"\b([A-Z][A-Za-z]+(?:[A-Z][a-z]+)+|[A-Z]{2,})\b", full_text):
        company = match.group(1)
        if company not in key_companies and company not in key_people:
            key_companies.append(company)
        if len(key_companies) >= 8:
            break

    # Key places: look for common place suffixes (very naive)
    key_places = []
    for match in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", full_text):
        place = match.group(1)
        if (place.endswith(("City", "County", "State", "Province", "Island", "Land", "Valley", "Bay", "Beach", "River", "Lake", "Park"))
            and place not in key_people and place not in key_companies and place not in key_places):
            key_places.append(place)
        if len(key_places) >= 8:
            break

    # Timeframe: look for "in 19xx", "in 20xx", "last year", "recently", etc. (simple)
    timeframe = "MODERN"
    if re.search(r"\b(19\d{2}|20\d{2})\b", full_text):
        timeframe = "HISTORICAL"
    elif re.search(r"\b(last year|recently|in the past decade)\b", full_text, re.I):
        timeframe = "RECENT"

    # Mood
    crash_words = ["crash", "collapse", "panic", "bankrupt"]
    opportunity_words = ["opportunity", "advantage", "position", "profit"]
    text_l = full_text.lower()
    if any(w in text_l for w in crash_words):
        overall_mood = "alarming"
    elif any(w in text_l for w in opportunity_words):
        overall_mood = "optimistic"
    else:
        overall_mood = "neutral"

    return JobProfile(
        job_id=job_id,
        channel=channel,
        dominant_topics=dominant_topics,
        key_people=key_people,
        key_companies=key_companies,
        key_places=key_places,
        timeframe=timeframe,
        overall_mood=overall_mood,
    )


def classify_role(slice_text: str, job_profile: JobProfile) -> str:
    """
    Heuristic: Classifies the role of a slice for semantic scene prompting.
    Returns one of: HOOK, CHARACTER_INTRO, CRASH_SCENE, MAP_SCENE, MACRO_STATEMENT.
    """
    text_l = (slice_text or "").lower()
    crash_words = ["crash", "collapse", "panic", "bankrupt"]
    map_words = ["tax", "state", "city", "county", "map", "location", "region", "area"]
    # HOOK fallback: if contains "let me", "here's why", "what they're doing"
    if any(phrase in text_l for phrase in ["let me", "here's why", "what they're doing"]):
        return "HOOK"
    # CHARACTER_INTRO: mentions key_people or CEO/founder/etc.
    if any(person in slice_text for person in job_profile.key_people) or re.search(r"\b(CEO|founder|chairman|executive|president|director)\b", slice_text, re.I):
        return "CHARACTER_INTRO"
    # CRASH_SCENE
    if any(word in text_l for word in crash_words):
        return "CRASH_SCENE"
    # MAP_SCENE
    if any(word in text_l for word in map_words):
        return "MAP_SCENE"
    # Default
    return "MACRO_STATEMENT"


def choose_topic_for_slice(slice_text: str, job_profile: JobProfile, channel: str) -> str:
    """
    Heuristic: Picks the best topic for this slice, using job_profile or channel style.
    """
    if job_profile.dominant_topics:
        return job_profile.dominant_topics[0]
    style_cfg = _load_channel_style(channel)
    topics = style_cfg.get("topics", {}) if style_cfg else {}
    text_l = (slice_text or "").lower()
    best_topic = ""
    best_count = 0
    for topic, info in topics.items():
        count = 0
        for kw in info.get("keywords", []):
            count += text_l.count(kw.lower())
        if count > best_count:
            best_topic = topic
            best_count = count
    return best_topic



# -------------------- Deterministic concept key helpers --------------------

_SLUG_RE = re.compile(r"[^a-z0-9]+")

def _slug(s: str) -> str:
    s = (s or "").lower()
    return _SLUG_RE.sub("-", s).strip("-")

def _hash8(s: str) -> str:
    return hashlib.blake2s((s or "").encode("utf-8"), digest_size=4).hexdigest()

def infer_era(text: str, job_profile: JobProfile) -> str:
    """Infer a coarse era tag from the job_profile or the text.

    Normalizes a variety of free-text era descriptions into a small, stable
    vocabulary so concept IDs remain consistent across runs.
    """
    # 1) Safe normalization: timeframe may be a string, list, None, or other type
    tf = getattr(job_profile, "timeframe", "")
    if isinstance(tf, list):
        era_raw = " ".join(str(x) for x in tf)
    else:
        era_raw = str(tf)

    era_raw = era_raw.strip().lower()
    year: Optional[int] = None

    # Direct 4-digit year inside timeframe, e.g. "1805", "1929"
    m = re.search(r"(1[5-9]\d{2}|20\d{2})", era_raw)
    if m:
        year = int(m.group(1))
    else:
        # Century phrases like "18th century", "19th century"
        m = re.search(r"(\d{2})th century", era_raw)
        if m:
            century = int(m.group(1))
            # 18th -> 1700s, 19th -> 1800s, 20th -> 1900s, etc.
            if 15 <= century <= 21:
                year = (century - 1) * 100

    # Map year (if found) into stable buckets
    if year is not None:
        if year < 1700:
            return "pre-1700"
        if year < 1800:
            return "1700s"
        if year < 1900:
            return "1800s"
        if year < 2000:
            return "1900s"
        return "2000s"

    # 2) Fallback: very rough scan of the slice text itself
    tl = (text or "").lower()
    if re.search(r"17\d{2}", tl):
        return "1700s"
    if re.search(r"179\d", tl):
        return "1790s"
    if re.search(r"180\d", tl):
        return "1800s"
    if re.search(r"19\d{2}", tl):
        return "1900s"
    if re.search(r"20\d{2}", tl):
        return "2000s"

    return "generic-era"

def infer_motif(text: str, approx_role: str) -> str:
    """
    Heuristic motif tag to distinguish portraits, battlefields, bank halls, etc.
    """
    tl = (text or "").lower()
    if "coin" in tl or "mint" in tl or "silver" in tl or "gold" in tl:
        return "coin_closeup"
    if "battle" in tl or "cannon" in tl or "musket" in tl or "army" in tl or "troops" in tl:
        return "battlefield"
    if "bank" in tl or "ledger" in tl or "credit" in tl or "debt" in tl or "notes" in tl:
        return "bank_hall"
    if "street" in tl or "market" in tl or "crowd" in tl or "bread" in tl:
        return "street_market"
    if "map" in tl and ("table" in tl or "desk" in tl):
        return "war_room"
    if approx_role and approx_role.startswith("CHARACTER"):
        return "portrait"
    # default motif by role
    role = (approx_role or "").upper()
    if role == "HOOK":
        return "hook_tableau"
    if role == "EVENT_PAST":
        return "historical_scene"
    if role == "CRASH_SCENE":
        return "crisis_scene"
    return "tableau"

def canonical_entity_key(text: str, job_profile: JobProfile) -> str:
    """
    Try to derive a stable entity key (person/org/place) from job_profile + text.
    This is intentionally simple and data-driven, not a fixed list.
    """
    tl = (text or "").lower()
    # Prefer explicit key_people from job_profile
    for person in getattr(job_profile, "key_people", []) or []:
        if person.lower() in tl:
            return f"person:{_slug(person)}"
    # Common historical/finance patterns that LLM might miss
    if "napoleon" in tl or "bonaparte" in tl:
        return "person:napoleon_bonaparte"
    # Institutions
    for inst in getattr(job_profile, "key_companies", []) or []:
        if inst.lower() in tl:
            return f"org:{_slug(inst)}"
    if "bank of france" in tl or "banque de france" in tl:
        return "org:bank_of_france"
    # Places
    for place in getattr(job_profile, "key_places", []) or []:
        if place.lower() in tl:
            return f"place:{_slug(place)}"
    if "paris" in tl:
        return "place:paris"
    return ""  # generic

def make_concept_key(text: str, topic: str, approx_role: str, job_profile: JobProfile) -> str:
    """
    Build a deterministic concept key from entity, motif, era, and topic,
    with a short hash of the text to separate distinct motifs under the same bucket.
    """
    entity = canonical_entity_key(text, job_profile) or "generic"
    motif = infer_motif(text, approx_role)
    era = infer_era(text, job_profile)
    topic_slug = _slug(topic or "generic")
    base = f"{entity}|{motif}|{era}|{topic_slug}"
    return f"{base}|h{_hash8(text or '')}"


def _slugify(value: str) -> str:
    """
    Lowercase, replace non-alphanumerics with _, collapse runs, strip _.
    """
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def assign_concept_id(
    slice_text: str,
    topic: str,
    job_profile: JobProfile,
    prev_concept_id: Optional[str] = None
) -> str:
    """
    Assigns a deterministic concept ID for a slice, using inferred entity, motif, era, and topic.
    The signature is kept for compatibility; prev_concept_id is currently unused.
    """
    approx_role = "MACRO_STATEMENT"  # callers can still refine this later via role classification
    return make_concept_key(slice_text or "", topic or "", approx_role, job_profile)


def _get_semantic_templates(channel: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Loads the channel's semantic style config and templates.
    Returns (style_cfg, templates) or ({}, {}) if not semantic.
    """
    style_cfg = _load_channel_style(channel)
    if _is_semantic_style(style_cfg):
        return style_cfg, style_cfg.get("templates", {})
    return {}, {}


def build_scene_prompt_semantic(
    channel: str,
    role: str,
    topic: str,
    slice_text: str,
    job_profile: JobProfile
) -> tuple[str, str]:
    """
    Builds (positive, negative) prompts for a scene using the semantic pipeline.
    Falls back to build_scene_prompt if channel is not semantic.
    """
    style_cfg, templates = _get_semantic_templates(channel)
    if not style_cfg or not templates:
        return build_scene_prompt(channel, slice_text)

    # Pick template key
    template_key = None
    if role.startswith("CHARACTER") and "PERSON_CORP" in templates:
        template_key = "PERSON_CORP"
    elif role == "CRASH_SCENE" and "CRASH_SCENE" in templates:
        template_key = "CRASH_SCENE"
    elif role == "MAP_SCENE" and "MAP_SCENE" in templates:
        template_key = "MAP_SCENE"
    elif "MACRO_SCENE" in templates:
        template_key = "MACRO_SCENE"
    elif "GENERIC_SCENE" in templates:
        template_key = "GENERIC_SCENE"
    else:
        template_key = next(iter(templates.keys()), None)

    template_str = templates.get(template_key, "{core_idea}")
    ctx = {
        "core_idea": _simplify_slice(slice_text),
        "person_role": job_profile.key_people[0] if job_profile.key_people else "",
        "company_logo": job_profile.key_companies[0] if job_profile.key_companies else "",
        "wealth_metaphor": "a mountain of cash",
        "region": job_profile.key_places[0] if job_profile.key_places else "",
        "topic": topic,
        "role": role,
        "mood": job_profile.overall_mood,
        "timeframe": job_profile.timeframe,
    }
    try:
        positive = template_str.format(**ctx)
    except Exception:
        positive = _simplify_slice(slice_text)
    # Prepend style/tone if present
    style = style_cfg.get("style", {})
    art_style = style.get("art_style", "")
    tone = style.get("tone", "")
    prefix = f"{art_style} {tone}".strip()
    if prefix:
        positive = f"{prefix}: {positive}"
    negative = style_cfg.get("negative_prompt", "")
    return positive, negative



# -------------------------- LLM-based Helpers (Additive) -------------------------

def analyse_job_llm(full_text: str, channel: str, job_id: str) -> JobProfile:
    """
    Uses an LLM to analyze the full text and channel to infer a richer JobProfile.
    Falls back to heuristic analyse_job on error or bad output.
    """
    from LLM.gpt_client import gpt_chat, LLMQuotaError  # local import to ensure LLM path is configured
    import json
    # Load channel-level visual + LLM identity
    channel_style_cfg = load_prompt_style(channel) or {}
    channel_style = channel_style_cfg.get("style", {})
    channel_llm = channel_style_cfg.get("llm", {})
    policy = load_channel_policy(channel)
    ctx = {
        "job_id": job_id,
        "channel": channel,
        "full_text": full_text,
        "channel_style": channel_style,
        "channel_llm": channel_llm,
    }
    if policy and hasattr(policy, "refine_llm_scene_context"):
        try:
            ctx = policy.refine_llm_scene_context(ctx)  # type: ignore[misc]
        except Exception:
            pass
    channel_style = ctx.get("channel_style", channel_style)
    channel_llm = ctx.get("channel_llm", channel_llm)
    system_prompt = (
        "You are an expert analyst for YouTube documentary scripts. "
        "You ALWAYS consider the channel's style and LLM identity when analyzing. "
        "Given the full script text and channel context, extract semantic fields as compact JSON. "
        "The JSON MUST contain exactly: themes, era, people, institutions, places, mood.\n\n"
        f"Channel Visual Style: {json.dumps(channel_style, ensure_ascii=False)}\n"
        f"Channel LLM Notes: {json.dumps(channel_llm, ensure_ascii=False)}\n"
        "Be concise but accurate. Return ONLY the JSON with no commentary."
    )
    user_prompt = (
        f"Channel: {channel}\n"
        f"Script:\n{full_text}\n"
        "Extract the semantic fields as described."
    )
    try:
        resp = gpt_chat(user_prompt, system_prompt=system_prompt, temperature=0.25)
        # Try to find first JSON object in response
        match = re.search(r"\{.*\}", resp, re.DOTALL)
        if match:
            resp_json = match.group(0)
        else:
            resp_json = resp
        data = json.loads(resp_json)
        # Defensive: ensure types
        dominant_topics = data.get("themes", []) or []
        if not isinstance(dominant_topics, list):
            dominant_topics = [str(dominant_topics)]
        key_people = data.get("people", []) or []
        if not isinstance(key_people, list):
            key_people = [str(key_people)]
        key_companies = data.get("institutions", []) or []
        if not isinstance(key_companies, list):
            key_companies = [str(key_companies)]
        key_places = data.get("places", []) or []
        if not isinstance(key_places, list):
            key_places = [str(key_places)]
        timeframe = data.get("era", "") or "MODERN"
        overall_mood = data.get("mood", "") or "neutral"
        return JobProfile(
            job_id=job_id,
            channel=channel,
            dominant_topics=dominant_topics,
            key_people=key_people,
            key_companies=key_companies,
            key_places=key_places,
            timeframe=timeframe,
            overall_mood=overall_mood,
        )
    except LLMQuotaError:
        # Propagate so orchestrator can abort
        raise
    except Exception:
        # Fallback to heuristic
        return analyse_job(full_text, channel, job_id)


def build_scene_prompt_llm(
    channel: str,
    slice_text: str,
    job_profile: JobProfile,
    approx_role: str = "",
    approx_topic: str = "",
) -> tuple[str, str, str]:
    """
    Uses LLM to classify the role and generate (positive, negative) prompts for a scene.
    Falls back to semantic builder on error.
    """
    from LLM.gpt_client import gpt_chat, LLMQuotaError  # local import to ensure LLM path is configured
    import json
    # Load channel-level style + llm identity
    channel_style_cfg = load_prompt_style(channel) or {}
    channel_style = channel_style_cfg.get("style", {})
    channel_llm = channel_style_cfg.get("llm", {})
    policy = load_channel_policy(channel)
    system_prompt = (
        "You are a cinematic art director for AI-generated YouTube documentaries. "
        "ALWAYS consider the channel's visual style and LLM identity. "
        "For each script line, you must:\n"
        "- Decide the correct 'role' label (HOOK, CHARACTER_INTRO, CHARACTER_FACT, EVENT_PAST, EVENT_PRESENT, FINANCE_SCENE, WAR_SCENE, EMPIRE_SCENE, MAP_SCENE, MACRO_STATEMENT, PATTERN_STATEMENT, WARNING, CONSEQUENCE).\n"
        "- Write one or two sentences describing a single, concrete, cinematic scene that visualizes the line.\n"
        "- STRICT RULE: No readable text, captions, subtitles, UI, logos, or infographics in the scene.\n"
        "- The scene MUST follow the channel's style guidelines.\n\n"
        f"Channel Visual Style: {json.dumps(channel_style, ensure_ascii=False)}\n"
        f"Channel LLM Notes: {json.dumps(channel_llm, ensure_ascii=False)}\n"
        "Respond ONLY with a JSON object: {\"role\":..., \"scene\":..., \"negative\":...(optional)}."
    )
    # Serialize job_profile fields for LLM context
    job_profile_dict = {
        "themes": job_profile.dominant_topics,
        "era": job_profile.timeframe,
        "people": job_profile.key_people,
        "companies": job_profile.key_companies,
        "places": job_profile.key_places,
        "mood": job_profile.overall_mood,
    }
    # Extract chapter label if encoded by orchestrator: [CHAPTER: X]
    chapter = ""
    m = re.search(r"\[CHAPTER:\s*([^\]]+)\]", slice_text)
    if m:
        chapter = m.group(1).strip()

    # Allow the channel policy to refine the LLM context (if present)
    ctx = {
        "job_profile": job_profile_dict,
        "channel_style": channel_style,
        "channel_llm": channel_llm,
        "topic": approx_topic,
        "role": approx_role,
        "text": slice_text,
        "chapter": chapter,
    }
    if policy and hasattr(policy, "refine_llm_scene_context"):
        try:
            ctx = policy.refine_llm_scene_context(ctx)  # type: ignore[misc]
        except Exception:
            pass
    job_profile_dict = ctx.get("job_profile", job_profile_dict)
    channel_style = ctx.get("channel_style", channel_style)
    channel_llm = ctx.get("channel_llm", channel_llm)
    approx_topic = ctx.get("topic", approx_topic)
    approx_role = ctx.get("role", approx_role)
    slice_text = ctx.get("text", slice_text)
    chapter = ctx.get("chapter", chapter)

    user_prompt = (
        f"Job profile: {json.dumps(job_profile_dict, ensure_ascii=False)}\n"
        f"Chapter: {chapter}\n"
        f"Approximate role: {approx_role}\n"
        f"Approximate topic: {approx_topic}\n"
        f"Script line: {slice_text}\n"
        "Produce the role + scene JSON now."
    )
    try:
        resp = gpt_chat(user_prompt, system_prompt=system_prompt, temperature=0.35)
        # Extract first JSON object in response
        match = re.search(r"\{.*\}", resp, re.DOTALL)
        if match:
            resp_json = match.group(0)
        else:
            resp_json = resp
        data = json.loads(resp_json)
        role = str(data.get("role", approx_role or "MACRO_STATEMENT")).upper()
        scene = data.get("scene", _simplify_slice(slice_text))
        # Allow channel policy to refine the scene text before it becomes the positive prompt
        if policy and hasattr(policy, "refine_scene_prompt_text"):
            try:
                scene = policy.refine_scene_prompt_text(scene, approx_topic, role)  # type: ignore[misc]
            except Exception:
                pass
        neg_extra = data.get("negative", "")
        negative = "text, words, letters, captions, subtitles, logos, UI, infographics"
        if neg_extra:
            negative = f"{negative}, {neg_extra}"
        positive = scene
        return (role, positive, negative)
    except LLMQuotaError:
        raise
    except Exception:
        # Fallback to semantic builder
        positive, negative = build_scene_prompt_semantic(
            channel,
            approx_role or "MACRO_STATEMENT",
            approx_topic or "",
            slice_text,
            job_profile,
        )
        return (approx_role or "MACRO_STATEMENT", positive, negative)