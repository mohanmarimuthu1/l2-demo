"""CLI for end-to-end retrieval + conflict resolution.

    python -m src.part3_rag.run_query                       # run all demo queries
    python -m src.part3_rag.run_query "my own question"     # one-off query

The three demo queries are deliberately phrased differently to probe how
retrieval, recency weighting, and the synthetic seed chunks interact:

    1. "Did I mention anything about my sister?"
         spec-literal phrasing. Natural sister-mentions dominate retrieval
         because they share the corpus's "do you have siblings?" register.
         Surfaces real factual contradictions from the multi-persona corpus
         (e.g. "I have two sisters, we're close" vs "I have one sister,
         we're not close").
    2. "What did I say about my sister recently?"
         more direct + recency-cued. Aims to pull the latest chunks forward
         and give the synthetic late-bucket seed a fighting chance.
    3. "Tell me about my relationship with my sister over time"
         temporal/relationship phrasing. Closest semantic match to the
         affect-laden synthetic chunks ("amazing", "impossible", "haven't
         spoken in a while") — the case where the resolver can demonstrate
         its behavior under controlled contradiction.

Saves a side-by-side report to artifacts/part3/sample_answers.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.part3_rag.resolve import resolve


ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = ROOT / "artifacts" / "part3" / "sample_answers.md"

DEMO_QUERIES = [
    {
        "query": "Did I mention anything about my sister?",
        "entity": "your sister",
        "label": "Query 1 — spec-literal phrasing (natural retrieval; surfaces real corpus contradictions)",
    },
    {
        "query": "What did I say about my sister recently?",
        "entity": "your sister",
        "label": "Query 2 — recency-cued phrasing (may pull synthetic late-bucket seed forward)",
    },
    {
        "query": "Tell me about my relationship with my sister over time",
        "entity": "your sister",
        "label": "Query 3 — temporal/relationship phrasing (closest match to synthetic affective chunks; resolver demo under controlled contradiction)",
    },
]


def _excerpt(text: str, n: int = 160) -> str:
    s = text.replace("\n", "  /  ")
    return s if len(s) <= n else s[: n - 1] + "…"


def _format_one(r: dict) -> str:
    lines: list[str] = []
    lines.append(f"### Query: `{r['query']}`")
    lines.append("")
    lines.append(f"**Entity:** {r['entity']}  ")
    lines.append(f"**Confidence:** {r['confidence']:.3f}  ")
    lines.append(f"**Contradictions flagged:** {len(r['contradictions'])}")
    lines.append("")
    lines.append("**Answer:**")
    lines.append("")
    lines.append(f"> {r['answer']}")
    lines.append("")
    lines.append("**Top source chunks (after composite re-rank):**")
    lines.append("")
    lines.append("| # | id | day | sent | cos | cosN | recN | emo | score | syn | excerpt |")
    lines.append("|--:|----|----:|-----:|----:|-----:|-----:|----:|------:|:---:|---------|")
    for i, c in enumerate(r["source_chunks"], 1):
        lines.append(
            f"| {i} | `{c['id']}` | {c['day_bucket']} | "
            f"{c['sentiment']:+.3f} | {c['cosine_sim']:.3f} | {c['cosine_norm']:.3f} | "
            f"{c['recency_norm']:.3f} | {c['emotional_weight']:.3f} | "
            f"**{c['score']:.3f}** | {'Y' if c['synthetic'] else ' '} | "
            f"{_excerpt(c['text'])} |"
        )
    lines.append("")
    if r["contradictions"]:
        lines.append("**NLI-flagged contradictions:**")
        lines.append("")
        lines.append("| a_id | b_id | a_day | b_day | a_sent | b_sent | contradiction_p |")
        lines.append("|------|------|------:|------:|-------:|-------:|----------------:|")
        for x in r["contradictions"]:
            lines.append(
                f"| `{x['a_id']}` | `{x['b_id']}` | {x['a_day']} | {x['b_day']} | "
                f"{x['a_sentiment']:+.3f} | {x['b_sentiment']:+.3f} | "
                f"**{x['contradiction_prob']:.3f}** |"
            )
        lines.append("")
    return "\n".join(lines)


def _print_one(r: dict) -> None:
    print()
    print("=" * 100)
    print(f"QUERY: {r['query']}")
    print(f"  entity: {r['entity']}    confidence: {r['confidence']:.3f}    "
          f"contradictions: {len(r['contradictions'])}")
    print("=" * 100)
    print("ANSWER:")
    print(f"  {r['answer']}")
    print()
    print("TOP SOURCE CHUNKS (composite re-rank):")
    print(f"  {'#':>2}  {'id':<22} {'day':>3} {'sent':>6} {'cos':>6} "
          f"{'cosN':>6} {'recN':>6} {'emo':>6} {'score':>6}  syn")
    for i, c in enumerate(r["source_chunks"], 1):
        print(
            f"  {i:>2}  {c['id']:<22} {c['day_bucket']:>3} "
            f"{c['sentiment']:>+6.3f} {c['cosine_sim']:>6.3f} {c['cosine_norm']:>6.3f} "
            f"{c['recency_norm']:>6.3f} {c['emotional_weight']:>6.3f} "
            f"{c['score']:>6.3f}  {'Y' if c['synthetic'] else ' '}"
        )
    if r["contradictions"]:
        print()
        print("NLI CONTRADICTIONS:")
        for x in r["contradictions"]:
            print(f"  {x['a_id']} <-> {x['b_id']}  "
                  f"(days {x['a_day']}/{x['b_day']}, "
                  f"sents {x['a_sentiment']:+.2f}/{x['b_sentiment']:+.2f})  "
                  f"contradiction={x['contradiction_prob']:.3f}")


def run_all(write_md: bool = True) -> list[dict]:
    out_records: list[dict] = []
    md_parts: list[str] = [
        "# Part 3 — Conflict-Resolving RAG: Sample Answers",
        "",
        "Three queries against the same index, phrased differently to probe how "
        "retrieval and the synthetic seed interact. The `syn` column flags chunks "
        "from the synthetic sister seed (`synthetic_seed`); rows without `Y` are "
        "natural turns from the multi-persona corpus.",
        "",
    ]
    for spec in DEMO_QUERIES:
        md_parts.append(f"## {spec['label']}")
        md_parts.append("")
        r = resolve(spec["query"], k=15, entity=spec["entity"])
        n_syn = sum(1 for c in r["source_chunks"] if c["synthetic"])
        md_parts.append(
            f"_Synthetic chunks in top-{len(r['source_chunks'])}: **{n_syn}**_"
        )
        md_parts.append("")
        out_records.append(r)
        _print_one(r)
        md_parts.append(_format_one(r))
        md_parts.append("")
    if write_md:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text("\n".join(md_parts), encoding="utf-8")
        print(f"\nwrote {OUT_PATH}")
    return out_records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("q", nargs="?", default=None,
                    help="optional one-off query; omit to run the three demo queries")
    ap.add_argument("--entity", default=None)
    ap.add_argument("-k", type=int, default=15)
    args = ap.parse_args()
    if args.q is None:
        run_all()
        return
    r = resolve(args.q, k=args.k, entity=args.entity)
    _print_one(r)
    print()
    print(json.dumps({k: v for k, v in r.items() if k != "source_chunks"}, indent=2))


if __name__ == "__main__":
    main()
