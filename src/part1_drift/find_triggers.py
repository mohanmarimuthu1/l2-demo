"""Find the trigger that likely caused each drift day.

Strategy:
    1. Candidate entities = day_n top-10 entities that didn't appear in any
       prior day's top-10.
    2. Filter candidates by a *concentration* check (see _passes_concentration).
    3. If any entity passes -> pick top by count.
    4. Else: same procedure for topics (terms in day_n top-10 not in any prior
       day's top-10), filtered by the same concentration check.
    5. Else: trigger = None.

The picked trigger is one of:
    {"kind": "entity", "value": "Portland", "label": "GPE",
     "count": 23, "distinct_convos": 2, "concentration": 0.913}
    {"kind": "topic",  "value": "yoga class",
     "count": 41, "distinct_convos": 3, "concentration": 0.927}
    None
"""

from __future__ import annotations


# Concentration thresholds. Rationale:
#   distinct_convos < DISTINCT_CONVOS_MAX
#       A real "trigger" should plausibly point at one event, person, or
#       story arc. If an entity is mentioned across 5+ unrelated conversations
#       in a day, it's a generic noun (a common first name, a popular country,
#       a household-name pop-culture reference), not a drift cause.
#   concentration > CONCENTRATION_MIN
#       Defined as 1 - (distinct_convos / total_mentions). High concentration
#       means the entity is mentioned multiple times *within* the same
#       conversation, which is the signature of a real arc (someone keeps
#       bringing it up). Low concentration means it's dropped once per
#       conversation across many speakers.
#
# Threshold values picked to be strict on this corpus, where most entities
# are diffuse (e.g. "Sarah" across 30+ unrelated stories). Loosen if a future
# corpus is more concentrated by default.
DISTINCT_CONVOS_MAX = 5
CONCENTRATION_MIN = 0.5


def _entity_keys(rec: dict) -> set[str]:
    return {e["text"] for e in rec.get("top_entities", [])}


def _topic_keys(rec: dict) -> set[str]:
    return {t["term"] for t in rec.get("top_topics", [])}


def _concentration(item: dict) -> float:
    """1 - (distinct_convos / total_mentions). High = concentrated."""
    n_convos = int(item.get("distinct_convos", 0))
    mentions = int(item.get("count", 0))
    if mentions <= 0:
        return 0.0
    return 1.0 - (n_convos / mentions)


def _passes_concentration(item: dict) -> bool:
    n_convos = int(item.get("distinct_convos", 0))
    mentions = int(item.get("count", 0))
    if mentions <= 0 or n_convos <= 0:
        return False
    if n_convos >= DISTINCT_CONVOS_MAX:
        return False
    return _concentration(item) > CONCENTRATION_MIN


def _pick(history: list[dict], cur: dict) -> dict | None:
    """An entity/topic counts as new only if it is absent from the top lists
    of *every* prior day. New candidates then have to pass the concentration
    check before they qualify as a trigger.
    """
    if not history:
        return None

    seen_ents: set[str] = set()
    seen_topics: set[str] = set()
    for prev in history:
        seen_ents |= _entity_keys(prev)
        seen_topics |= _topic_keys(prev)

    new_ents = [e for e in cur.get("top_entities", []) if e["text"] not in seen_ents]
    qualified_ents = [e for e in new_ents if _passes_concentration(e)]
    if qualified_ents:
        qualified_ents.sort(key=lambda e: e["count"], reverse=True)
        e = qualified_ents[0]
        return {
            "kind": "entity",
            "value": e["text"],
            "label": e["label"],
            "count": int(e["count"]),
            "distinct_convos": int(e["distinct_convos"]),
            "concentration": round(_concentration(e), 4),
            "alternatives": [
                {
                    "value": x["text"],
                    "label": x["label"],
                    "count": int(x["count"]),
                    "distinct_convos": int(x["distinct_convos"]),
                    "concentration": round(_concentration(x), 4),
                }
                for x in qualified_ents[1:3]
            ],
        }

    new_topics = [t for t in cur.get("top_topics", []) if t["term"] not in seen_topics]
    qualified_topics = [t for t in new_topics if _passes_concentration(t)]
    if qualified_topics:
        qualified_topics.sort(key=lambda t: t.get("score", 0.0), reverse=True)
        t = qualified_topics[0]
        return {
            "kind": "topic",
            "value": t["term"],
            "score": float(t.get("score", 0.0)),
            "count": int(t["count"]),
            "distinct_convos": int(t["distinct_convos"]),
            "concentration": round(_concentration(t), 4),
            "alternatives": [
                {
                    "value": x["term"],
                    "score": float(x.get("score", 0.0)),
                    "count": int(x["count"]),
                    "distinct_convos": int(x["distinct_convos"]),
                    "concentration": round(_concentration(x), 4),
                }
                for x in qualified_topics[1:3]
            ],
        }

    return None


def attach_triggers(days: list[dict], drift_records: list[dict]) -> list[dict | None]:
    triggers: list[dict | None] = []
    history: list[dict] = []
    for rec, drift in zip(days, drift_records):
        if drift["drift_from_prev"]:
            triggers.append(_pick(history, rec))
        else:
            triggers.append(None)
        history.append(rec)
    return triggers
