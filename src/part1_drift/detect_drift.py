"""Pairwise drift detection across consecutive day-buckets.

For each (day_{n-1}, day_n) pair:
    sentiment_delta = |compound_n - compound_{n-1}|
    topic_jaccard   = |topics_n ∩ topics_{n-1}| / |topics_n ∪ topics_{n-1}|
    mood_changed    = bool
    tone_changed    = bool
    drift_score     = weighted composite, see below

A day is flagged as a drift day if any of these is true:
    sentiment_delta > 0.30
    topic_jaccard   < 0.30
    mood_changed
    tone_changed
"""

from __future__ import annotations

from typing import Iterable


SENTIMENT_DELTA_THRESHOLD = 0.30
TOPIC_JACCARD_THRESHOLD = 0.30


def _terms(rec: dict) -> set[str]:
    return {t["term"] for t in rec.get("top_topics", [])}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 1.0


def _pair_metrics(prev: dict, cur: dict) -> dict:
    sd = abs(cur["avg_compound"] - prev["avg_compound"])
    tj = _jaccard(_terms(prev), _terms(cur))
    mood_changed = prev["mood"] != cur["mood"]
    tone_changed = prev["tone"] != cur["tone"]

    # drift score in [0, 1+]: sentiment shift contributes up to ~1.0, topic
    # divergence up to 1.0, mood/tone change each adds a fixed bump
    drift_score = (
        min(sd / 0.5, 1.0) * 0.45
        + (1.0 - tj) * 0.35
        + (0.10 if mood_changed else 0.0)
        + (0.10 if tone_changed else 0.0)
    )
    drift = (
        sd > SENTIMENT_DELTA_THRESHOLD
        or tj < TOPIC_JACCARD_THRESHOLD
        or mood_changed
        or tone_changed
    )
    return {
        "sentiment_delta": round(sd, 4),
        "topic_jaccard": round(tj, 4),
        "mood_changed": bool(mood_changed),
        "tone_changed": bool(tone_changed),
        "drift_score": round(drift_score, 4),
        "drift_from_prev": bool(drift),
    }


def detect(days: Iterable[dict]) -> list[dict]:
    days = list(days)
    out = []
    prev = None
    for rec in days:
        if prev is None:
            out.append({
                "day": rec["day"],
                "sentiment_delta": 0.0,
                "topic_jaccard": 1.0,
                "mood_changed": False,
                "tone_changed": False,
                "drift_score": 0.0,
                "drift_from_prev": False,
            })
        else:
            m = _pair_metrics(prev, rec)
            m["day"] = rec["day"]
            out.append(m)
        prev = rec
    return out
