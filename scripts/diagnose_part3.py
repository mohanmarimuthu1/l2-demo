"""Sanity-check Part 3 outputs:
  1. NLI on synthetic sister chunks directly (warm vs frustrated vs distant).
  2. Where do synthetic chunks rank for the spec query?
  3. Inspect the natural NLI-flagged pairs — real conflict or NLI artifact?
  4. Weight ablation: composite vs cosine-only — does ranking actually move?

Run:  .venv/Scripts/python.exe scripts/diagnose_part3.py
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

from src.part3_rag import retrieve
from src.part3_rag.resolve import _ce, NLI_LABELS
from src.part3_rag.chunk import SYNTHETIC_SISTER_CHUNKS


ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = ROOT / "artifacts" / "part3" / "chunks.json"

QUERY = "Did I mention anything about my sister?"


def _short(s: str, n: int = 140) -> str:
    s = s.replace("\n", " | ")
    return s if len(s) <= n else s[: n - 1] + "…"


def nli(pairs: list[tuple[str, str]]) -> list[tuple[float, float, float]]:
    probs = _ce().predict(list(pairs), apply_softmax=True, show_progress_bar=False)
    return [(float(p[0]), float(p[1]), float(p[2])) for p in probs]


def step1_synthetic_nli():
    print("\n" + "=" * 100)
    print("STEP 1  NLI directly on the 3 synthetic sister chunks (pair-wise)")
    print("=" * 100)
    syn = SYNTHETIC_SISTER_CHUNKS
    print(f"  s0 day1 warm     : {_short(syn[0]['text'])}")
    print(f"  s1 day4 frustrated: {_short(syn[1]['text'])}")
    print(f"  s2 day6 distant  : {_short(syn[2]['text'])}")
    pair_idxs = list(itertools.combinations(range(3), 2))
    pairs = [(syn[i]['text'], syn[j]['text']) for i, j in pair_idxs]
    probs = nli(pairs)
    print(f"\n  {'pair':<10} {'contradict':>10} {'entail':>8} {'neutral':>8}   "
          f"verdict")
    for (i, j), (c, e, n) in zip(pair_idxs, probs):
        verdict = NLI_LABELS[max(range(3), key=lambda k: (c, e, n)[k])]
        flagged = "FLAGGED" if c > 0.6 else ""
        print(f"  s{i} <-> s{j}  {c:>10.3f} {e:>8.3f} {n:>8.3f}   {verdict:<14} {flagged}")


def step2_synthetic_in_results():
    print("\n" + "=" * 100)
    print("STEP 2  Where do synthetic chunks rank in retrieval for the spec query?")
    print("=" * 100)
    results = retrieve.query(QUERY, k=30)
    print(f"  asked k=30, got {len(results)} results")
    print(f"  {'rank':>4}  {'id':<22} {'cos':>6} {'cosN':>6} {'recN':>6} {'emo':>6} "
          f"{'score':>6}  syn   text")
    for rank, r in enumerate(results, 1):
        if r["synthetic"] or rank <= 5:
            mark = "Y" if r["synthetic"] else " "
            print(f"  {rank:>4}  {r['id']:<22} {r['cosine_sim']:>6.3f} "
                  f"{r['cosine_norm']:>6.3f} {r['recency_norm']:>6.3f} "
                  f"{r['emotional_weight']:>6.3f} {r['score']:>6.3f}  "
                  f"{mark}    {_short(r['text'], 90)}")


def step3_inspect_natural_pairs():
    print("\n" + "=" * 100)
    print("STEP 3  Inspect the natural pairs that NLI flagged as contradictions")
    print("=" * 100)
    results = retrieve.query(QUERY, k=15)
    top5 = results[:5]
    pair_idxs = list(itertools.combinations(range(len(top5)), 2))
    pairs = [(top5[i]['text'], top5[j]['text']) for i, j in pair_idxs]
    probs = nli(pairs)
    rows = []
    for (i, j), (c, e, n) in zip(pair_idxs, probs):
        rows.append({
            "a_id": top5[i]['id'], "b_id": top5[j]['id'],
            "a_sent": top5[i]['sentiment'], "b_sent": top5[j]['sentiment'],
            "a_day": top5[i]['day_bucket'], "b_day": top5[j]['day_bucket'],
            "a_text": top5[i]['text'], "b_text": top5[j]['text'],
            "c": c, "e": e, "n": n,
        })
    flagged = [r for r in rows if r['c'] > 0.6]
    print(f"  {len(rows)} pairs evaluated, {len(flagged)} above 0.6\n")
    for r in flagged:
        print(f"  {r['a_id']} <-> {r['b_id']}   "
              f"contradict={r['c']:.3f}  entail={r['e']:.3f}  neutral={r['n']:.3f}")
        print(f"    A (day {r['a_day']}, sent {r['a_sent']:+.2f}): {_short(r['a_text'], 200)}")
        print(f"    B (day {r['b_day']}, sent {r['b_sent']:+.2f}): {_short(r['b_text'], 200)}")
        print()


def step4_ablation():
    print("\n" + "=" * 100)
    print("STEP 4  Weight ablation — does composite re-rank actually move ordering?")
    print("=" * 100)
    results = retrieve.query(QUERY, k=15)
    by_score = sorted(results, key=lambda r: r['score'], reverse=True)
    by_cos = sorted(results, key=lambda r: r['cosine_sim'], reverse=True)

    score_rank = {r['id']: i for i, r in enumerate(by_score)}
    cos_rank = {r['id']: i for i, r in enumerate(by_cos)}

    print(f"  k=15 candidates ranked by composite vs by raw cosine\n")
    print(f"  {'composite-rank':>14}  {'cos-rank':>9}  {'id':<22} "
          f"{'cos':>6} {'cosN':>6} {'recN':>6} {'emo':>6} {'score':>6}  syn")
    for r in by_score:
        delta = cos_rank[r['id']] - score_rank[r['id']]
        delta_s = f"({delta:+d})" if delta else "( =)"
        mark = "Y" if r['synthetic'] else " "
        print(f"  {score_rank[r['id']]+1:>14}  {cos_rank[r['id']]+1:>4} {delta_s:>4}  "
              f"{r['id']:<22} {r['cosine_sim']:>6.3f} {r['cosine_norm']:>6.3f} "
              f"{r['recency_norm']:>6.3f} {r['emotional_weight']:>6.3f} "
              f"{r['score']:>6.3f}  {mark}")

    moved = sum(1 for r in by_score if score_rank[r['id']] != cos_rank[r['id']])
    print(f"\n  {moved}/{len(by_score)} candidates changed rank vs cosine-only.")

    # top-5 set diff
    top5_score = {r['id'] for r in by_score[:5]}
    top5_cos = {r['id'] for r in by_cos[:5]}
    dropped = top5_cos - top5_score
    gained = top5_score - top5_cos
    print(f"  top-5 set diff:  +{sorted(gained)}  -{sorted(dropped)}")


def main():
    step1_synthetic_nli()
    step2_synthetic_in_results()
    step3_inspect_natural_pairs()
    step4_ablation()


if __name__ == "__main__":
    main()
