"""Scan conversations for natural-contradiction entity candidates.

For each PERSON/ORG/GPE entity that appears across 3+ different day-buckets
(reusing Part 1's bin assignment), compute:
    total_mentions
    distinct_conversations
    distinct_day_buckets
    sentiment_variance      (variance of per-turn VADER compound over mentions)
    concentration_ratio     (Part 1 definition: 1 - distinct_convos/total_mentions)

Filter: concentration_ratio > 0.5 AND distinct_day_buckets >= 3.
Rank by sentiment_variance descending; print top 10.

Outputs:
    artifacts/part3/candidates.json         ranked list (top 10 + filter info)
    artifacts/part3/entity_mentions.csv     per-mention table cached for reuse
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import spacy

from src.common.io import load_conversations, explode_turns
from src.common.sentiment import vader_compound
from src.common.windows import assign_days


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "artifacts" / "part3"
OUT_PATH = OUT_DIR / "candidates.json"
CACHE_PATH = OUT_DIR / "entity_mentions.csv"

ENTITY_LABELS = {"PERSON", "ORG", "GPE"}
N_BINS = 7

CONCENTRATION_MIN = 0.5         # strict >, mirrors Part 1
DISTINCT_DAYS_MIN = 3
TOTAL_MENTIONS_MIN = 3          # "appearing 3+ times"

_PRONOUN_LIKE = {"i", "you", "we", "they", "he", "she", "it", "u"}


def build_mentions_table() -> pd.DataFrame:
    convos = load_conversations()
    binned, info = assign_days(convos, n_bins=N_BINS)
    print(f"binning: {info}")

    turns = explode_turns(binned[["conversation_id", "raw"]])
    day_lookup = dict(zip(binned["conversation_id"], binned["day"]))
    turns["day"] = turns["conversation_id"].map(day_lookup).astype(int)
    print(f"exploded into {len(turns):,} turns")

    print("scoring sentiment per turn (VADER)...")
    turns["compound"] = turns["text"].astype(str).map(vader_compound)

    print("loading spaCy (ner only)...")
    nlp = spacy.load(
        "en_core_web_sm",
        disable=["parser", "tagger", "attribute_ruler", "lemmatizer"],
    )

    texts = turns["text"].astype(str).tolist()
    days = turns["day"].tolist()
    conv_ids = turns["conversation_id"].tolist()
    compounds = turns["compound"].tolist()

    rows: list[tuple] = []
    print(f"running NER on {len(turns):,} turns...")
    t0 = time.time()
    for i, doc in enumerate(nlp.pipe(texts, batch_size=256)):
        d = days[i]
        cid = conv_ids[i]
        comp = compounds[i]
        for ent in doc.ents:
            if ent.label_ not in ENTITY_LABELS:
                continue
            txt = ent.text.strip()
            if len(txt) < 2 or txt.lower() in _PRONOUN_LIKE:
                continue
            rows.append((txt, ent.label_, d, cid, comp))
    dt = time.time() - t0
    print(f"NER done in {dt:.1f}s, {len(rows):,} mentions extracted")
    return pd.DataFrame(rows, columns=["entity", "label", "day", "conv_id", "compound"])


def aggregate(mentions: pd.DataFrame) -> pd.DataFrame:
    """Roll up mentions to one row per entity surface form."""
    # primary label = most-common label seen for this surface form
    label_mode = (
        mentions.groupby("entity")["label"]
        .agg(lambda s: s.value_counts().index[0])
        .rename("label")
    )
    grp = mentions.groupby("entity")
    total_mentions = grp.size().rename("total_mentions")
    distinct_conversations = grp["conv_id"].nunique().rename("distinct_conversations")
    distinct_day_buckets = grp["day"].nunique().rename("distinct_day_buckets")
    # population variance across mentions (consistent for small N; ddof=0)
    sentiment_variance = grp["compound"].var(ddof=0).rename("sentiment_variance").fillna(0.0)
    sentiment_mean = grp["compound"].mean().rename("sentiment_mean")
    sentiment_min = grp["compound"].min().rename("sentiment_min")
    sentiment_max = grp["compound"].max().rename("sentiment_max")

    df = pd.concat(
        [label_mode, total_mentions, distinct_conversations,
         distinct_day_buckets, sentiment_variance,
         sentiment_mean, sentiment_min, sentiment_max],
        axis=1,
    ).reset_index()
    df["concentration_ratio"] = 1.0 - df["distinct_conversations"] / df["total_mentions"]
    return df


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    mentions = build_mentions_table()
    mentions.to_csv(CACHE_PATH, index=False)
    print(f"cached per-mention table -> {CACHE_PATH}")

    df = aggregate(mentions)
    print(f"unique entities: {len(df):,}")

    mask = (
        (df["total_mentions"] >= TOTAL_MENTIONS_MIN)
        & (df["concentration_ratio"] > CONCENTRATION_MIN)
        & (df["distinct_day_buckets"] >= DISTINCT_DAYS_MIN)
    )
    qualified = (
        df[mask]
        .sort_values("sentiment_variance", ascending=False)
        .reset_index(drop=True)
    )
    print(
        f"qualified (mentions>={TOTAL_MENTIONS_MIN}, "
        f"concentration>{CONCENTRATION_MIN}, "
        f"distinct_days>={DISTINCT_DAYS_MIN}): {len(qualified):,}"
    )

    records: list[dict] = []
    for _, row in qualified.head(10).iterrows():
        records.append({
            "entity": str(row["entity"]),
            "label": str(row["label"]),
            "total_mentions": int(row["total_mentions"]),
            "distinct_conversations": int(row["distinct_conversations"]),
            "distinct_day_buckets": int(row["distinct_day_buckets"]),
            "concentration_ratio": round(float(row["concentration_ratio"]), 4),
            "sentiment_variance": round(float(row["sentiment_variance"]), 4),
            "sentiment_mean": round(float(row["sentiment_mean"]), 4),
            "sentiment_min": round(float(row["sentiment_min"]), 4),
            "sentiment_max": round(float(row["sentiment_max"]), 4),
        })

    payload = {
        "info": {
            "n_bins": N_BINS,
            "entity_labels": sorted(ENTITY_LABELS),
            "filters": {
                "total_mentions_min": TOTAL_MENTIONS_MIN,
                "concentration_ratio_min_exclusive": CONCENTRATION_MIN,
                "distinct_day_buckets_min": DISTINCT_DAYS_MIN,
            },
            "rank_by": "sentiment_variance_desc",
            "n_unique_entities": int(len(df)),
            "n_qualified": int(len(qualified)),
        },
        "top": records,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"wrote {OUT_PATH}")

    # pretty print
    print()
    print("=" * 112)
    print("TOP 10 NATURAL-CONTRADICTION CANDIDATES  (rank by sentiment_variance desc)")
    print("=" * 112)
    print(
        f"{'#':>2}  {'entity':<22} {'lbl':<6} "
        f"{'mentions':>8} {'convos':>7} {'days':>4} "
        f"{'conc':>6} {'var':>7} {'mean':>7} {'min':>7} {'max':>7}"
    )
    print("-" * 112)
    for i, r in enumerate(records, 1):
        print(
            f"{i:>2}  {r['entity'][:22]:<22} {r['label']:<6} "
            f"{r['total_mentions']:>8} {r['distinct_conversations']:>7} "
            f"{r['distinct_day_buckets']:>4} "
            f"{r['concentration_ratio']:>6.3f} {r['sentiment_variance']:>7.4f} "
            f"{r['sentiment_mean']:>7.3f} {r['sentiment_min']:>7.3f} {r['sentiment_max']:>7.3f}"
        )
    print()


if __name__ == "__main__":
    main()
