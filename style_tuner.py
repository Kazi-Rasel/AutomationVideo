# File: /Users/kazirasel/VideoAutomation/System/SysVisuals/Tools/style_tuner.py
#!/usr/bin/env python3
"""
style_tuner.py

Observes past jobs (manifest.json files) and produces a report about how well
the current visual style behaves, especially for finance/stock-related scenes.

This script is intentionally READ-ONLY. It does NOT modify any config.
It is meant to sit beside prompt_style.json and help you decide how to evolve it.
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List


STOCK_KEYWORDS = [
    "stock", "stocks", "equity", "equities", "market", "markets",
    "share price", "valuation", "valuations", "index", "indices",
    "sell off", "sell-off", "bear market", "bull market", "portfolio"
]

CHART_KEYWORDS = [
    "chart", "charts", "graph", "graphs", "line graph", "candlestick",
    "candlesticks", "screen", "monitor", "screen full of", "price chart",
    "market screen", "trading app", "phone screen", "laptop screen"
]

PORTRAIT_KEYWORDS = [
    "portrait", "close-up", "close up", "glamor", "glamour",
    "beauty shot", "face lit", "cinematic close", "head and shoulders",
    "posed", "posing"
]


def is_stock_line(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in STOCK_KEYWORDS)


def prompt_mentions_chart(prompt: str) -> bool:
    p = (prompt or "").lower()
    return any(k in p for k in CHART_KEYWORDS)


def prompt_looks_portraity(prompt: str) -> bool:
    p = (prompt or "").lower()
    return any(k in p for k in PORTRAIT_KEYWORDS)


def find_manifests(manifest_root: Path, channel: str) -> List[Path]:
    """
    Locate manifest.json files under:
      <manifest_root>/<channel>/Build/Ready/**/manifest.json

    manifest_root is typically something like ~/YTProjects or VA_ROOT/YTProjects,
    but is configurable via CLI.
    """
    manifests: List[Path] = []
    base = manifest_root / channel / "Build" / "Ready"
    if not base.is_dir():
        return []
    for path in base.rglob("manifest.json"):
        manifests.append(path)
    return manifests


def analyze_manifest(path: Path, max_problem_samples: int = 50) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    frames = data.get("frames") or []
    problems_no_chart: List[Dict[str, Any]] = []
    problems_portrait: List[Dict[str, Any]] = []

    total = 0
    stock_total = 0
    stock_with_chart = 0
    stock_without_chart = 0
    portrait_in_stock = 0

    for f in frames:
        total += 1
        text = f.get("text") or f.get("caption") or ""
        prompt = f.get("prompt") or ""
        topic = f.get("topic") or ""
        role = f.get("role") or ""

        stock = is_stock_line(text) or topic in ("STOCK_MARKET", "FINANCE_CHARTS", "MARKET_CRASH")
        if stock:
            stock_total += 1
            has_chart = prompt_mentions_chart(prompt)
            is_portrait = prompt_looks_portraity(prompt)

            if has_chart:
                stock_with_chart += 1
            else:
                stock_without_chart += 1
                if len(problems_no_chart) < max_problem_samples:
                    problems_no_chart.append({
                        "manifest": str(path),
                        "index": f.get("index"),
                        "text": text,
                        "prompt": prompt,
                        "topic": topic,
                        "role": role,
                    })

            if is_portrait:
                portrait_in_stock += 1
                if len(problems_portrait) < max_problem_samples:
                    problems_portrait.append({
                        "manifest": str(path),
                        "index": f.get("index"),
                        "text": text,
                        "prompt": prompt,
                        "topic": topic,
                        "role": role,
                    })

    return {
        "total_frames": total,
        "stock_frames": stock_total,
        "stock_frames_with_chart_prompt": stock_with_chart,
        "stock_frames_without_chart_prompt": stock_without_chart,
        "portrait_like_frames_in_stock_context": portrait_in_stock,
        "samples_no_chart": problems_no_chart,
        "samples_portrait_in_stock": problems_portrait,
    }


def merge_stats(stats_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {
        "total_frames": 0,
        "stock_frames": 0,
        "stock_frames_with_chart_prompt": 0,
        "stock_frames_without_chart_prompt": 0,
        "portrait_like_frames_in_stock_context": 0,
        "samples_no_chart": [],
        "samples_portrait_in_stock": [],
    }
    for st in stats_list:
        merged["total_frames"] += st.get("total_frames", 0)
        merged["stock_frames"] += st.get("stock_frames", 0)
        merged["stock_frames_with_chart_prompt"] += st.get("stock_frames_with_chart_prompt", 0)
        merged["stock_frames_without_chart_prompt"] += st.get("stock_frames_without_chart_prompt", 0)
        merged["portrait_like_frames_in_stock_context"] += st.get("portrait_like_frames_in_stock_context", 0)
        merged["samples_no_chart"].extend(st.get("samples_no_chart", []))
        merged["samples_portrait_in_stock"].extend(st.get("samples_portrait_in_stock", []))
    # limit samples
    merged["samples_no_chart"] = merged["samples_no_chart"][:50]
    merged["samples_portrait_in_stock"] = merged["samples_portrait_in_stock"][:50]
    return merged


def default_va_root() -> Path:
    return Path(os.environ.get("VA_ROOT", str(Path.home() / "VideoAutomation")))


def default_manifest_root() -> Path:
    # Many setups use ~/YTProjects; adjust if your manifests live elsewhere.
    env = os.environ.get("YT_MANIFEST_ROOT", "")
    if env:
        return Path(env)
    # Fallback: try VA_ROOT / \"YTProjects\"
    return default_va_root() / "YTProjects"


def main():
    parser = argparse.ArgumentParser(description="Style tuner (analysis only) for channel manifests.")
    parser.add_argument("--channel", type=str, default="CapitalChronicles", help="Channel name (folder under manifest root)")
    parser.add_argument("--manifest-root", type=str, default=str(default_manifest_root()), help="Root folder containing channel manifests (default: YTProjects under VA_ROOT or $YT_MANIFEST_ROOT)")
    parser.add_argument("--max-samples", type=int, default=50, help="Max sample problem frames to record for each category")
    args = parser.parse_args()

    manifest_root = Path(args.manifest_root)
    channel = args.channel

    print(f"[STYLE TUNER] Analyzing channel={channel} manifests under {manifest_root}")
    manifests = find_manifests(manifest_root, channel)
    if not manifests:
        print(f"[STYLE TUNER] No manifest.json files found for channel '{channel}' under {manifest_root}")
        return

    all_stats: List[Dict[str, Any]] = []
    for mpath in manifests:
        try:
            st = analyze_manifest(mpath, max_problem_samples=args.max_samples)
            all_stats.append(st)
        except Exception as e:
            print(f"[STYLE TUNER] ⚠ Failed to analyze {mpath}: {e}")

    merged = merge_stats(all_stats)

    total_frames = merged["total_frames"]
    stock_frames = merged["stock_frames"]
    if total_frames == 0:
        print("[STYLE TUNER] No frames found.")
        return

    print("\n===== STYLE TUNER SUMMARY =====")
    print(f"Total frames analyzed: {total_frames}")
    print(f"Stock-related frames: {stock_frames}")

    if stock_frames > 0:
        pct_stock = 100.0 * stock_frames / total_frames
        pct_with_chart = 100.0 * merged["stock_frames_with_chart_prompt"] / stock_frames
        pct_without_chart = 100.0 * merged["stock_frames_without_chart_prompt"] / stock_frames
        pct_portrait_in_stock = 100.0 * merged["portrait_like_frames_in_stock_context"] / stock_frames

        print(f"Stock frames % of total: {pct_stock:.1f}%")
        print(f"Stock frames WITH chart keywords in prompt: {merged['stock_frames_with_chart_prompt']} ({pct_with_chart:.1f}%)")
        print(f"Stock frames WITHOUT chart keywords in prompt: {merged['stock_frames_without_chart_prompt']} ({pct_without_chart:.1f}%)")
        print(f"Stock frames that look portrait-like: {merged['portrait_like_frames_in_stock_context']} ({pct_portrait_in_stock:.1f}%)")
    else:
        print("No stock-related frames detected in these manifests.")

    # Write report next to the channel's Config folder
    va_root = default_va_root()
    channel_cfg_dir = va_root / "Channels" / channel / "Config"
    channel_cfg_dir.mkdir(parents=True, exist_ok=True)
    report_path = channel_cfg_dir / "style_tuner_report.json"
    report_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[STYLE TUNER] Report written to {report_path}\n")

    print("Sample stock frames WITHOUT chart keywords in prompt (up to 50):")
    for s in merged["samples_no_chart"]:
        print(f" - {s['manifest']} [index={s['index']}] topic={s['topic']} role={s['role']}")
        print(f"   line: {s['text']}")
        print(f"   prompt: {s['prompt'][:200]}{'...' if len(s['prompt'])>200 else ''}")

    print("\nSample STOCK frames that look portrait-like (up to 50):")
    for s in merged["samples_portrait_in_stock"]:
        print(f" - {s['manifest']} [index={s['index']}] topic={s['topic']} role={s['role']}")
        print(f"   line: {s['text']}")
        print(f"   prompt: {s['prompt'][:200]}{'...' if len(s['prompt'])>200 else ''}")


if __name__ == "__main__":
    main()