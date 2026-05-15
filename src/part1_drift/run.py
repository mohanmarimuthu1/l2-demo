"""End-to-end Part 1 pipeline.

    python -m src.part1_drift.run [--bins 7]

Steps:
    1. bin conversations into N day-buckets
    2. compute daily personas (sentiment, tone features, mood, tone, topics, entities)
    3. detect drift between consecutive days
    4. find the trigger for each drift day
    5. write artifacts/part1/drift_timeline.json
    6. pretty-print the timeline to stdout
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.part1_drift.build_daily_persona import build as build_personas
from src.part1_drift.detect_drift import detect as detect_drift
from src.part1_drift.find_triggers import attach_triggers


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "artifacts" / "part1"
OUT_PATH = OUT_DIR / "drift_timeline.json"


def _format_topics(topics: list[dict], n: int = 5) -> str:
    return ", ".join(t["term"] for t in topics[:n]) or "—"


def _format_entities(ents: list[dict], n: int = 5) -> str:
    if not ents:
        return "—"
    return ", ".join(f"{e['text']}({e['label']})" for e in ents[:n])


def _format_trigger(trig: dict | None) -> str:
    if trig is None:
        return "—"
    if trig["kind"] == "entity":
        return f"entity:{trig['value']} ({trig['label']}, n={trig['count']})"
    if trig["kind"] == "topic":
        return f"topic:{trig['value']} (s={trig['score']:.3f})"
    return "—"


def pretty_print(timeline: list[dict]) -> None:
    print()
    print("=" * 100)
    print("DRIFT TIMELINE")
    print("=" * 100)
    print(f"{'day':>3}  {'mood':<10} {'tone':<8} {'compound':>9} "
          f"{'drift':>6} {'score':>6}  trigger")
    print("-" * 100)
    for r in timeline:
        flag = "yes" if r["drift_from_prev"] else "—"
        print(f"{r['day']:>3}  {r['mood']:<10} {r['tone']:<8} "
              f"{r['avg_compound']:>9.4f} {flag:>6} {r['drift_score']:>6.3f}  "
              f"{_format_trigger(r['trigger'])}")
    print()
    for r in timeline:
        print(f"  day {r['day']}  topics:   {_format_topics(r['top_topics'])}")
        print(f"          entities: {_format_entities(r['top_entities'])}")
    print()


def run(n_bins: int = 7) -> list[dict]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    personas = build_personas(n_bins=n_bins)
    days = personas["days"]

    drift_records = detect_drift(days)
    triggers = attach_triggers(days, drift_records)

    timeline = []
    for rec, drift, trig in zip(days, drift_records, triggers):
        timeline.append({
            "day": rec["day"],
            "mood": rec["mood"],
            "tone": rec["tone"],
            "avg_compound": rec["avg_compound"],
            "features": rec["features"],
            "top_topics": rec["top_topics"],
            "top_entities": rec["top_entities"],
            "drift_from_prev": drift["drift_from_prev"],
            "drift_score": drift["drift_score"],
            "drift_detail": {
                "sentiment_delta": drift["sentiment_delta"],
                "topic_jaccard": drift["topic_jaccard"],
                "mood_changed": drift["mood_changed"],
                "tone_changed": drift["tone_changed"],
            },
            "trigger": trig,
        })

    OUT_PATH.write_text(json.dumps({
        "info": personas["info"],
        "timeline": timeline,
    }, indent=2))
    print(f"wrote {OUT_PATH}")
    pretty_print(timeline)
    return timeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bins", type=int, default=7)
    args = ap.parse_args()
    run(n_bins=args.bins)


if __name__ == "__main__":
    main()
