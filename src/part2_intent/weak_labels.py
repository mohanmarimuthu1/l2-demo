"""Rule-based weak labeler.

Walks every turn in conversations.csv, applies a priority-ordered set of
regex/keyword rules, and writes (text, label, source_rule) to
artifacts/part2/labeled.csv.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "conversations.csv"
OUT_DIR = ROOT / "artifacts" / "part2"

LABELS = ["reminder", "emotional-support", "action-item", "small-talk", "unknown"]

# rules are evaluated in priority order, first match wins
REMINDER_PAT = re.compile(
    r"\b(remind me|don'?t forget|tomorrow at|schedule\b|set a |set an |reminder|"
    r"in the morning|next week|next month)\b",
    re.I,
)
EMO_PAT = re.compile(
    r"\b(feeling|i feel|i'm feeling|sad|anxious|anxiety|sorry to hear|tough day|"
    r"stressed|depress(ed|ing)?|lonely|overwhelmed|crying|exhausted|miserable|"
    r"heart ?broken|burn(ed|t) out)\b",
    re.I,
)
NEED_PAT = re.compile(r"\b(i need to|i have to|gotta|got to|i must|i should)\b", re.I)
IMPERATIVE_VERBS = {
    "send", "call", "book", "buy", "finish", "email", "text", "pick",
    "grab", "remind", "schedule", "cancel", "pay", "submit", "order",
    "fix", "review", "check", "update",
}
SMALL_PAT = re.compile(
    r"\b(lol|haha+|hehe+|how are you|how's it going|how was your day|"
    r"good morning|good night|good evening|hey there|hi there|hello|"
    r"see you|see ya|take care|bye|thanks|thank you|no problem|"
    r"you're welcome|what's up|nice to meet you)\b",
    re.I,
)

USER_PREFIX = re.compile(r"^\s*User\s*\d+\s*:\s*", re.I)


def split_turns(conv_text: str) -> list[str]:
    out = []
    for line in conv_text.split("\n"):
        line = USER_PREFIX.sub("", line).strip()
        if line:
            out.append(line)
    return out


def label_one(text: str) -> tuple[str, str]:
    t = text.strip()
    if not t:
        return "unknown", "empty"

    if REMINDER_PAT.search(t):
        return "reminder", "kw:reminder"

    # sentence-initial imperative: first token is an action verb, not a question
    first_tok = re.split(r"[\s,.!?]+", t.lower(), maxsplit=1)[0]
    if first_tok in IMPERATIVE_VERBS and not t.endswith("?"):
        return "action-item", f"imperative:{first_tok}"
    if NEED_PAT.search(t):
        return "action-item", "kw:need-to"

    if EMO_PAT.search(t):
        return "emotional-support", "kw:emotion"

    if SMALL_PAT.search(t):
        return "small-talk", "kw:smalltalk"

    # very short utterances (acks, "ok", "yeah") read as small-talk
    wc = len(t.split())
    if wc <= 3:
        return "small-talk", "short-utterance"

    return "unknown", "fallback"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA, header=None, names=["conversation"])

    rows = []
    for conv in df["conversation"].dropna():
        for turn in split_turns(conv):
            label, rule = label_one(turn)
            rows.append((turn, label, rule))

    out = pd.DataFrame(rows, columns=["text", "label", "source_rule"])
    out_path = OUT_DIR / "labeled.csv"
    out.to_csv(out_path, index=False, quoting=csv.QUOTE_MINIMAL)

    print(f"wrote {len(out):,} rows to {out_path}")
    print()
    print("class distribution:")
    counts = out["label"].value_counts()
    total = len(out)
    for label in LABELS:
        n = int(counts.get(label, 0))
        pct = 100.0 * n / total if total else 0.0
        print(f"  {label:<20s} {n:>8,}  ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
