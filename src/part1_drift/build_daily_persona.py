"""Build one persona per day-bucket.

For each day we record:
    n_turns, n_conversations
    avg_compound                       (VADER compound, mean over turns)
    features.question_rate             (sum(?) / sum(words))
    features.exclamation_rate          (sum(!) / sum(words))
    features.caps_ratio
    features.avg_word_len
    features.contraction_rate          (contractions / words)
    mood                               (curious/frustrated/happy/sad/playful/neutral)
    tone                               (formal/casual/playful)
    top_topics                         (top 10 TF-IDF 1-2 grams)
    top_entities                       (top 5 PERSON/ORG/GPE/EVENT)

Output: artifacts/part1/daily_personas.json
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

from src.common.io import load_conversations, explode_turns
from src.common.sentiment import vader_compound
from src.common.windows import assign_days


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "artifacts" / "part1"
OUT_PATH = OUT_DIR / "daily_personas.json"

ENTITY_LABELS = {"PERSON", "ORG", "GPE", "EVENT"}

# tokens we collapse before TF-IDF so the topic list reads like content, not
# pronoun / filler noise. sklearn's English stop set covers most of this but
# misses a few high-frequency chat fillers.
EXTRA_STOPS = {
    # apostrophe-stripped (kept for older tokenizers)
    "im", "ive", "ill", "thats", "dont", "youre", "youve", "youll",
    "yeah", "yep", "ok", "okay", "lol", "haha", "hehe",
    "user", "user 1", "user 2",
    # apostrophe-form contractions that the current tokenizer leaves intact
    "i'm", "i've", "i'll", "i'd",
    "you're", "you've", "you'll", "you'd",
    "we're", "we've", "we'll", "we'd",
    "they're", "they've", "they'll", "they'd",
    "he's", "she's", "it's", "that's", "what's", "there's", "here's",
    "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't",
    "won't", "can't", "couldn't", "shouldn't", "wouldn't", "haven't",
    "hasn't", "hadn't", "let's", "y'all", "ya'll",
}

_CONTRACTION = re.compile(
    r"\b(?:don'?t|won'?t|can'?t|i'?m|you'?re|we'?re|they'?re|it'?s|that'?s|"
    r"i'?ve|you'?ve|we'?ve|they'?ve|i'?ll|you'?ll|he'?ll|she'?ll|we'?ll|"
    r"they'?ll|isn'?t|aren'?t|wasn'?t|weren'?t|didn'?t|doesn'?t|haven'?t|"
    r"hasn'?t|hadn'?t|wouldn'?t|couldn'?t|shouldn'?t|let'?s|what'?s|"
    r"there'?s|here'?s|gonna|wanna|gotta)\b",
    re.I,
)
_WORD = re.compile(r"[A-Za-z']+")
_Q = re.compile(r"\?")
_E = re.compile(r"!")


def _stop_words() -> list[str]:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    return sorted(set(ENGLISH_STOP_WORDS) | EXTRA_STOPS)


def _aggregate_features(texts: list[str]) -> dict:
    total_words = 0
    total_chars = 0
    q_hits = 0
    e_hits = 0
    caps_words = 0
    word_len_sum = 0
    contraction_hits = 0
    for t in texts:
        s = t or ""
        words = _WORD.findall(s)
        total_words += len(words)
        total_chars += len(s)
        q_hits += len(_Q.findall(s))
        e_hits += len(_E.findall(s))
        word_len_sum += sum(len(w) for w in words)
        caps_words += sum(1 for w in words if len(w) >= 2 and w.isupper())
        contraction_hits += len(_CONTRACTION.findall(s))
    denom = max(total_words, 1)
    return {
        "question_rate":    round(q_hits / denom, 5),
        "exclamation_rate": round(e_hits / denom, 5),
        "caps_ratio":       round(caps_words / denom, 5),
        "avg_word_len":     round(word_len_sum / denom, 4),
        "contraction_rate": round(contraction_hits / denom, 5),
        "_total_words": total_words,
        "_total_chars": total_chars,
    }


def daily_mood(compound: float, features: dict) -> str:
    """Spec labels: curious / frustrated / happy / sad / playful / neutral."""
    qr = features["question_rate"]
    er = features["exclamation_rate"]
    cr = features["caps_ratio"]
    if compound >= 0.5 and (er >= 0.05 or cr >= 0.03):
        return "playful"
    if compound >= 0.4:
        return "happy"
    if compound <= -0.4 and (er >= 0.05 or cr >= 0.03):
        return "frustrated"
    if compound <= -0.2:
        return "sad"
    if abs(compound) < 0.25 and qr >= 0.06:
        return "curious"
    return "neutral"


def daily_tone(features: dict) -> str:
    """Spec labels: formal / casual / playful."""
    er = features["exclamation_rate"]
    cr = features["caps_ratio"]
    if er >= 0.08 or cr >= 0.05:
        return "playful"
    formality = features["avg_word_len"] - (features["contraction_rate"] * 4)
    if formality >= 4.5 and er < 0.03:
        return "formal"
    return "casual"


def _top_topics_per_day(day_texts: dict[int, str], k: int = 10) -> dict[int, list[dict]]:
    days = sorted(day_texts.keys())
    docs = [day_texts[d] for d in days]
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=1.0,                 # only 7 docs, so cap is moot
        stop_words=_stop_words(),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z']+\b",
    )
    mat = vec.fit_transform(docs)
    vocab = vec.get_feature_names_out()
    out: dict[int, list[dict]] = {}
    for row, d in enumerate(days):
        row_arr = mat.getrow(row).toarray().ravel()
        top_idx = row_arr.argsort()[::-1][:k]
        out[d] = [
            {"term": str(vocab[i]), "score": round(float(row_arr[i]), 4)}
            for i in top_idx if row_arr[i] > 0
        ]
    return out


def _top_entities_per_day(turns: pd.DataFrame, k: int = 5) -> dict[int, list[dict]]:
    print("  loading spaCy (ner only)...")
    nlp = spacy.load(
        "en_core_web_sm",
        disable=["parser", "tagger", "attribute_ruler", "lemmatizer"],
    )
    print(f"  running NER over {len(turns):,} turns...")
    per_day: dict[int, Counter] = {}
    per_day_label: dict[int, dict[str, str]] = {}
    per_day_convos: dict[int, dict[str, set]] = {}
    texts = turns["text"].astype(str).tolist()
    days = turns["day"].tolist()
    conv_ids = turns["conversation_id"].tolist()
    t0 = time.time()
    for i, doc in enumerate(nlp.pipe(texts, batch_size=256)):
        d = days[i]
        cid = conv_ids[i]
        for ent in doc.ents:
            if ent.label_ not in ENTITY_LABELS:
                continue
            key = ent.text.strip()
            # drop one-letter or trivially short entities
            if len(key) < 2 or key.lower() in {"i", "you", "we", "they"}:
                continue
            per_day.setdefault(d, Counter())[key] += 1
            per_day_label.setdefault(d, {})[key] = ent.label_
            per_day_convos.setdefault(d, {}).setdefault(key, set()).add(cid)
    dt = time.time() - t0
    print(f"  NER done in {dt:.1f}s")

    out: dict[int, list[dict]] = {}
    for d, counter in per_day.items():
        top = counter.most_common(k)
        out[d] = [
            {
                "text": txt,
                "label": per_day_label[d][txt],
                "count": int(n),
                "distinct_convos": int(len(per_day_convos[d][txt])),
            }
            for txt, n in top
        ]
    return out


def _attach_topic_concentration(turns: pd.DataFrame, topics: dict[int, list[dict]]) -> dict[int, list[dict]]:
    """For each top-topic of each day, count distinct conversations whose
    turns contain the term (word-bounded) and the total occurrences. This
    lets the trigger picker apply the same concentration check used for
    entities.
    """
    out: dict[int, list[dict]] = {}
    for d, topic_list in topics.items():
        day_turns = turns[turns["day"] == d]
        per_conv = (
            day_turns.assign(text_lc=day_turns["text"].astype(str).str.lower())
            .groupby("conversation_id")["text_lc"]
            .apply(lambda s: " \n ".join(s))
        )
        enhanced = []
        for t in topic_list:
            term = t["term"].lower()
            pat = re.compile(r"\b" + re.escape(term) + r"\b")
            total = 0
            convos: set = set()
            for cid, blob in per_conv.items():
                hits = len(pat.findall(blob))
                if hits:
                    total += hits
                    convos.add(cid)
            enhanced.append({
                **t,
                "count": int(total),
                "distinct_convos": int(len(convos)),
            })
        out[d] = enhanced
    return out


def build(n_bins: int = 7) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    convos = load_conversations()
    binned, info = assign_days(convos, n_bins=n_bins)
    print(f"binning: {info}")

    turns = explode_turns(binned[["conversation_id", "raw"]])
    # propagate the day onto each turn via the conversation_id
    day_lookup = dict(zip(binned["conversation_id"], binned["day"]))
    turns["day"] = turns["conversation_id"].map(day_lookup).astype(int)
    print(f"exploded into {len(turns):,} turns")

    # --- per-turn sentiment, then aggregate per day ------------------------
    print("scoring sentiment per turn...")
    turns["compound"] = turns["text"].astype(str).map(vader_compound)

    # --- day-level aggregates ----------------------------------------------
    day_text_concat: dict[int, str] = {}
    days = sorted(turns["day"].unique().tolist())
    day_records: list[dict] = []
    for d in days:
        sub = turns[turns["day"] == d]
        texts = sub["text"].astype(str).tolist()
        feats = _aggregate_features(texts)
        avg_c = float(sub["compound"].mean())
        mood = daily_mood(avg_c, feats)
        tone = daily_tone(feats)
        day_text_concat[d] = " \n ".join(texts)
        day_records.append({
            "day": int(d),
            "n_turns": int(len(sub)),
            "n_conversations": int(sub["conversation_id"].nunique()),
            "avg_compound": round(avg_c, 4),
            "features": {
                k: v for k, v in feats.items() if not k.startswith("_")
            },
            "mood": mood,
            "tone": tone,
        })

    # --- topics + entities -------------------------------------------------
    # entities kept at top-10 so trigger detection has a wider history window
    # to filter against (the pretty-printer still shows top-5).
    print("computing per-day TF-IDF topics...")
    topics_raw = _top_topics_per_day(day_text_concat, k=10)
    print("annotating topics with concentration stats...")
    topics = _attach_topic_concentration(turns, topics_raw)
    print("computing per-day entities...")
    entities = _top_entities_per_day(turns, k=10)

    for rec in day_records:
        d = rec["day"]
        rec["top_topics"] = topics.get(d, [])
        rec["top_entities"] = entities.get(d, [])

    result = {
        "info": info,
        "days": day_records,
    }
    OUT_PATH.write_text(json.dumps(result, indent=2))
    print(f"wrote {OUT_PATH}")
    return result


if __name__ == "__main__":
    build(n_bins=7)
