"""Sentiment + mood labeling.

VADER gives us the compound sentiment score on [-1, 1]. The mood label
combines that score with light surface features (question rate, exclamation
density, caps ratio) so the label is more than just positive/negative — it
captures formality and energy too.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


_QUESTION = re.compile(r"\?")
_EXCLAIM = re.compile(r"!")
_WORD = re.compile(r"[A-Za-z']+")


@lru_cache(maxsize=1)
def _analyzer() -> SentimentIntensityAnalyzer:
    return SentimentIntensityAnalyzer()


def vader_compound(text: str) -> float:
    if not text:
        return 0.0
    return float(_analyzer().polarity_scores(text)["compound"])


@dataclass
class SurfaceFeatures:
    n_chars: int
    n_words: int
    question_rate: float   # questions per sentence-ish unit
    exclaim_rate: float
    caps_ratio: float      # ratio of all-caps words (>=2 letters) to total words
    avg_word_len: float


def surface_features(text: str) -> SurfaceFeatures:
    s = text or ""
    words = _WORD.findall(s)
    n_words = len(words)
    n_chars = len(s)
    q = len(_QUESTION.findall(s))
    e = len(_EXCLAIM.findall(s))
    # use n_words as the denominator for rates — sentence-count is too noisy in chat text
    denom = max(n_words, 1)
    caps_words = sum(1 for w in words if len(w) >= 2 and w.isupper())
    avg_len = (sum(len(w) for w in words) / n_words) if n_words else 0.0
    return SurfaceFeatures(
        n_chars=n_chars,
        n_words=n_words,
        question_rate=q / denom,
        exclaim_rate=e / denom,
        caps_ratio=caps_words / n_words if n_words else 0.0,
        avg_word_len=avg_len,
    )


def mood_label(compound: float, features: SurfaceFeatures) -> str:
    """Map (compound, features) to a short tone label.

    Buckets chosen to read naturally in a timeline rather than be exhaustive.
    """
    valence = "positive" if compound >= 0.25 else "negative" if compound <= -0.25 else "neutral"

    # energy: lots of exclamation marks, caps, or short bursty messages -> high
    energy_score = features.exclaim_rate * 3 + features.caps_ratio * 2
    energy = "high" if energy_score >= 0.6 else "low" if energy_score < 0.15 else "medium"

    # formality: long average word length + low question rate + neutral valence reads formal
    formality_score = features.avg_word_len + (0.5 if features.question_rate < 0.05 else 0.0)
    formal = formality_score >= 4.8

    # name the bucket combos
    if valence == "negative" and energy == "high":
        return "frustrated"
    if valence == "negative" and energy in ("medium", "low"):
        return "down"
    if valence == "positive" and energy == "high":
        return "playful"
    if valence == "positive" and energy == "medium":
        return "warm"
    if valence == "positive" and energy == "low":
        return "content"
    if valence == "neutral" and features.question_rate >= 0.08:
        return "curious"
    if valence == "neutral" and formal:
        return "formal"
    return "neutral"
