"""Top-k retrieval with a composite re-ranker.

    cosine_norm       = minmax(cos_sim across retrieved candidates)
    recency_norm      = day_bucket / MAX_DAY
    emotional_weight  = abs(sentiment)
    score             = 0.5*cosine_norm + 0.3*recency_norm + 0.2*emotional_weight

Returns a ranked list of dicts; every sub-score is visible so the caller
(resolve.py / answer formatter) can explain *why* a chunk was surfaced.

    python -m src.part3_rag.retrieve "did I mention my sister?"
"""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

from src.part3_rag.embed_index import MODEL_NAME, open_collection


MAX_DAY = 6                   # n_bins=7 -> days 0..6
W_COS, W_REC, W_EMO = 0.5, 0.3, 0.2


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME)


def _minmax(xs: list[float]) -> list[float]:
    if not xs:
        return []
    lo, hi = min(xs), max(xs)
    if hi - lo < 1e-12:
        return [1.0 for _ in xs]
    return [(x - lo) / (hi - lo) for x in xs]


def _embed(q: str):
    return _model().encode([q], normalize_embeddings=True).tolist()


def query(q: str, k: int = 15) -> list[dict]:
    col = open_collection()
    res = col.query(
        query_embeddings=_embed(q),
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    ids = res["ids"][0]
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    # cosine distance (chroma hnsw:space=cosine) -> similarity
    cos_sims = [1.0 - float(d) for d in dists]
    cos_norms = _minmax(cos_sims)

    out: list[dict] = []
    for i in range(len(ids)):
        m = metas[i]
        sentiment = float(m.get("sentiment", 0.0))
        day = int(m.get("day_bucket", 0))
        rec_norm = day / MAX_DAY if MAX_DAY else 0.0
        emo = abs(sentiment)
        score = W_COS * cos_norms[i] + W_REC * rec_norm + W_EMO * emo
        out.append({
            "id": ids[i],
            "text": docs[i],
            "conversation_id": int(m.get("conversation_id", -1)),
            "turn_index": int(m.get("turn_index", 0)),
            "day_bucket": day,
            "sentiment": sentiment,
            "synthetic": bool(m.get("synthetic", False)),
            "cosine_sim": round(cos_sims[i], 4),
            "cosine_norm": round(cos_norms[i], 4),
            "recency_norm": round(rec_norm, 4),
            "emotional_weight": round(emo, 4),
            "score": round(score, 4),
        })
    out.sort(key=lambda r: r["score"], reverse=True)
    return out


def _trim(s: str, n: int = 90) -> str:
    s = s.replace("\n", " ⏎ ")
    return s if len(s) <= n else s[: n - 1] + "…"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("q")
    ap.add_argument("-k", type=int, default=15)
    args = ap.parse_args()
    results = query(args.q, k=args.k)
    print(f"\nquery: {args.q!r}")
    print(f"{'#':>2}  {'id':<22} {'day':>3} {'sent':>6} "
          f"{'cos':>6} {'cosN':>6} {'recN':>6} {'emo':>6} {'score':>6}  syn  text")
    print("-" * 130)
    for i, r in enumerate(results, 1):
        print(
            f"{i:>2}  {r['id']:<22} {r['day_bucket']:>3} "
            f"{r['sentiment']:>+6.3f} {r['cosine_sim']:>6.3f} "
            f"{r['cosine_norm']:>6.3f} {r['recency_norm']:>6.3f} "
            f"{r['emotional_weight']:>6.3f} {r['score']:>6.3f}  "
            f"{'Y' if r['synthetic'] else ' '}    {_trim(r['text'])}"
        )


if __name__ == "__main__":
    main()
