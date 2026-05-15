"""Conflict-resolving answer formatter.

1. Retrieve top-k via composite re-ranker (retrieve.query).
2. Run pairwise NLI on the top-5 (10 unordered pairs) using
   cross-encoder/nli-deberta-v3-xsmall — flag pairs with
   contradiction_prob > NLI_CONTRADICTION_MIN.
3. Group top-5 by sentiment sign and produce a templated answer that quotes
   an early and a late chunk. If a contradiction is detected we surface the
   conflicting attribute; otherwise the answer reads as a chronological
   summary.

Returns: {answer, source_chunks, contradictions, confidence}.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import re
from functools import lru_cache
from pathlib import Path

log = logging.getLogger(__name__)

from src.part3_rag import retrieve


NLI_MODEL = "cross-encoder/nli-deberta-v3-xsmall"
# label order on this checkpoint (per HF model card): [contradiction, entailment, neutral]
NLI_LABELS = ("contradiction", "entailment", "neutral")

TOP_FOR_NLI = 5
NLI_CONTRADICTION_MIN = 0.6
SENTIMENT_SIGN_THRESHOLD = 0.05


@lru_cache(maxsize=1)
def _ce():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(NLI_MODEL)


def _sentiment_sign(s: float) -> str:
    if s > SENTIMENT_SIGN_THRESHOLD:
        return "pos"
    if s < -SENTIMENT_SIGN_THRESHOLD:
        return "neg"
    return "neutral"


def _nli_pairwise(chunks: list[dict]) -> list[dict]:
    if len(chunks) < 2:
        return []
    pairs = list(itertools.combinations(range(len(chunks)), 2))
    inputs = [[chunks[i]["text"], chunks[j]["text"]] for i, j in pairs]
    probs = _ce().predict(inputs, apply_softmax=True, show_progress_bar=False)
    flagged: list[dict] = []
    for (i, j), p in zip(pairs, probs):
        c = float(p[0]); e = float(p[1]); n = float(p[2])
        if c > NLI_CONTRADICTION_MIN:
            flagged.append({
                "a_id": chunks[i]["id"],
                "b_id": chunks[j]["id"],
                "a_day": chunks[i]["day_bucket"],
                "b_day": chunks[j]["day_bucket"],
                "a_sentiment": chunks[i]["sentiment"],
                "b_sentiment": chunks[j]["sentiment"],
                "contradiction_prob": round(c, 4),
                "entailment_prob": round(e, 4),
                "neutral_prob": round(n, 4),
            })
    flagged.sort(key=lambda r: r["contradiction_prob"], reverse=True)
    return flagged


def _group(chunks: list[dict]) -> dict[str, list[dict]]:
    g = {"pos": [], "neg": [], "neutral": []}
    for c in chunks:
        g[_sentiment_sign(c["sentiment"])].append(c)
    return g


# pronouns / determiners stripped when matching `entity` against a line —
# the entity passed in is often a possessive phrase ("your sister", "my mom"),
# but chunks say "sister" / "my sister", never "your sister".
_ENTITY_STOPWORDS = {
    "the", "a", "an", "my", "your", "our", "their", "his", "her",
    "this", "that", "these", "those", "some", "any",
}


def _entity_content_tokens(entity: str) -> list[str]:
    toks = re.findall(r"[A-Za-z][A-Za-z'\-]*", entity.lower())
    content = [t for t in toks if t not in _ENTITY_STOPWORDS]
    return content or toks  # if everything was a stopword, keep original


def _excerpt(text: str, n: int = 180, entity: str | None = None) -> str:
    """Pick the most informative single line for the answer.

    Strategy:
      1. If `entity` is given, prefer the User-prefixed line that contains
         the entity (or one of its content tokens — pronouns stripped).
         Tie-break by most-tokens-matched, then by longer body.
      2. Otherwise, return the first User-prefixed line with body length >= 8.
      3. If nothing matched at step 1, log a warning and fall back to step 2.
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    user_bodies: list[str] = []
    for line in lines:
        m = re.match(r"User\s+\d+\s*:\s*(.+)$", line, re.I)
        if m:
            body = m.group(1).strip()
            if len(body) >= 8:
                user_bodies.append(body)

    def _trim(s: str) -> str:
        return s if len(s) <= n else s[: n - 1] + "…"

    if entity:
        tokens = _entity_content_tokens(entity)
        full = entity.strip().lower()
        scored: list[tuple[int, int, str]] = []
        for body in user_bodies:
            lb = body.lower()
            hits = sum(1 for t in tokens if t and t in lb)
            if full and full in lb:
                hits += 1  # bonus when full phrase appears verbatim
            if hits:
                scored.append((hits, len(body), body))
        if scored:
            scored.sort(reverse=True)  # most hits, then longest body
            return _trim(scored[0][2])
        log.warning(
            "_excerpt: no user line matched entity %r (tokens=%s) in chunk; "
            "falling back to first user line",
            entity, tokens,
        )

    if user_bodies:
        return _trim(user_bodies[0])
    raw = " ".join(lines)
    return _trim(raw)


def _day_label(day_bucket: int) -> str:
    # the chronology is synthetic_row_order (Part 1's info), so we report
    # bucket positions, not wallclock dates.
    return f"day-bucket {day_bucket}"


def _closing_note_for_contradiction(
    top_pair: dict, chunks_by_id: dict[str, dict]
) -> str:
    """Closing sentence describing *what* the contradicting chunks disagree on.

    Two cases:
      - sentiment signs differ → "conflict on emotional tone"
      - same sign + NLI fired  → "appear inconsistent on factual details"

    No "tone" fallback when signs match — that was misleading on cases like
    the sibling-count contradiction where both turns are warmly positive
    but state different facts (two sisters vs one sister).
    """
    a = chunks_by_id[top_pair["a_id"]]
    b = chunks_by_id[top_pair["b_id"]]
    sa, sb = _sentiment_sign(a["sentiment"]), _sentiment_sign(b["sentiment"])
    if {sa, sb} == {"pos", "neg"}:
        return "Note: these accounts conflict on emotional tone (one positive, one negative)."
    return "Note: these accounts appear inconsistent on factual details."


def _build_answer(
    entity: str,
    top: list[dict],
    contradictions: list[dict],
) -> str:
    n = len(top)
    days = sorted({c["day_bucket"] for c in top})
    span = (
        f"day-bucket {days[0]}" if len(days) == 1
        else f"day-buckets {days[0]}–{days[-1]}"
    )

    if not contradictions:
        # chronological summary fallback
        by_day = sorted(top, key=lambda c: c["day_bucket"])
        first, last = by_day[0], by_day[-1]
        early = f"Earlier ({_day_label(first['day_bucket'])}): \"{_excerpt(first['text'], entity=entity)}\""
        later = f"Later ({_day_label(last['day_bucket'])}): \"{_excerpt(last['text'], entity=entity)}\""
        return (
            f"You've mentioned {entity} across {n} occasions over {span}. "
            f"{early}. {later}. "
            "Note: these accounts read as consistent in tone."
        )

    top_pair = contradictions[0]
    chunks_by_id = {c["id"]: c for c in top}
    a = chunks_by_id[top_pair["a_id"]]
    b = chunks_by_id[top_pair["b_id"]]
    if a["day_bucket"] <= b["day_bucket"]:
        earlier_c, later_c = a, b
    else:
        earlier_c, later_c = b, a
    closing = _closing_note_for_contradiction(top_pair, chunks_by_id)
    earlier = (
        f"Earlier ({_day_label(earlier_c['day_bucket'])}): "
        f"\"{_excerpt(earlier_c['text'], entity=entity)}\""
    )
    later = (
        f"Later ({_day_label(later_c['day_bucket'])}): "
        f"\"{_excerpt(later_c['text'], entity=entity)}\""
    )
    return (
        f"You've mentioned {entity} across {n} occasions over {span}. "
        f"{earlier}. {later}. "
        f"{closing}"
    )


def _confidence(results: list[dict], contradictions: list[dict]) -> float:
    if not results:
        return 0.0
    top3 = results[:3]
    mean_cos = sum(r["cosine_sim"] for r in top3) / len(top3)
    # contradictions don't mean the *retrieval* is wrong, but they do mean
    # the synthesized answer is less monolithic — lightly penalize.
    penalty = 0.85 if contradictions else 1.0
    return round(max(0.0, min(1.0, mean_cos * penalty)), 4)


def _entity_from_query(q: str) -> str:
    m = re.search(r"about\s+(?:my\s+)?([A-Za-z][A-Za-z'\-]*)", q, re.I)
    if m:
        return m.group(1)
    # "...my X..." fallback
    m = re.search(r"\bmy\s+([A-Za-z][A-Za-z'\-]*)", q, re.I)
    if m:
        return m.group(1)
    return "this topic"


def resolve(q: str, k: int = 15, entity: str | None = None) -> dict:
    if entity is None:
        entity = _entity_from_query(q)
    results = retrieve.query(q, k=k)
    top = results[:TOP_FOR_NLI]
    contradictions = _nli_pairwise(top)
    answer = _build_answer(entity, top, contradictions)
    return {
        "query": q,
        "entity": entity,
        "answer": answer,
        "source_chunks": top,
        "contradictions": contradictions,
        "confidence": _confidence(results, contradictions),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("q")
    ap.add_argument("-k", type=int, default=15)
    ap.add_argument("--entity", default=None)
    args = ap.parse_args()
    out = resolve(args.q, k=args.k, entity=args.entity)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
