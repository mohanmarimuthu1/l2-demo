"""Build a 150-row dev set, stratified across the five intent classes.

Outputs two files:

  dev_unlabeled.csv   text-only sheet for hand-labeling
  dev_gold_auto.csv   heuristic auto-gold so train.py has something to evaluate
                      against immediately. Replace by writing dev_gold.csv with
                      the same rows + a 'label' column; train.py prefers
                      dev_gold.csv when it exists.

The auto-gold uses tighter rules than weak_labels.py so it isn't a perfect
mirror of the training labels. Where the tight rules and the weak label
disagree, the row is marked needs_review=True.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts" / "part2"
WEAK_CSV = ART / "labeled.csv"

PER_CLASS = 30
SEED = 42

# tighter rules used to produce auto-gold. each rule needs at least two signals
# to fire, which makes it stricter than the single-keyword weak rules.

FUTURE_TIME = re.compile(
    r"\b(tomorrow|tonight|monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"next (week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
    r"at \d{1,2}(:\d{2})?\s*(am|pm)?|in the morning|in the evening|later today)\b",
    re.I,
)
REMINDER_KW = re.compile(r"\b(remind me|don'?t forget|reminder|schedule)\b", re.I)

EMO_STRONG = re.compile(
    r"\b(sad|anxious|anxiety|stressed|depress(ed|ing)?|lonely|overwhelmed|"
    r"crying|miserable|heart ?broken|burn(ed|t) out|exhausted|hurt|grieving)\b",
    re.I,
)
EMO_RECEIVE = re.compile(r"\b(sorry to hear|that sucks|hugs|sending love|i'm here for you|here for you)\b", re.I)

IMPER_VERBS = {
    "send", "call", "book", "buy", "finish", "email", "text", "pick",
    "grab", "remind", "schedule", "cancel", "pay", "submit", "order",
    "fix", "review", "check", "update",
}
NEED_KW = re.compile(r"\b(i need to|i have to|gotta|got to|i must|i should)\b", re.I)

SMALL_KW = re.compile(
    r"\b(lol|haha+|how are you|how's it going|good morning|good night|"
    r"hey there|hi there|hello|see you|take care|bye|thanks|thank you|"
    r"no problem|you're welcome|what's up|nice to meet you)\b",
    re.I,
)


def auto_gold(text: str) -> tuple[str, str]:
    """Return (label, reasoning). Tighter than weak_labels.label_one."""
    t = text.strip()
    if not t:
        return "unknown", "empty"

    # reminder: need an explicit reminder keyword AND a future-time anchor,
    # OR an explicit "remind me" / "don't forget" which is unambiguous on its own
    if re.search(r"\b(remind me|don'?t forget)\b", t, re.I):
        return "reminder", "explicit remind-me phrase"
    if REMINDER_KW.search(t) and FUTURE_TIME.search(t):
        return "reminder", "reminder kw + future time"

    # emotional support: either a strong feeling word about self, or a
    # consoling phrase aimed at the other person
    if EMO_STRONG.search(t) and re.search(r"\b(i|me|my|i'm|im)\b", t, re.I):
        return "emotional-support", "first-person emotion word"
    if EMO_RECEIVE.search(t):
        return "emotional-support", "consoling phrase"

    # action-item: sentence-initial imperative verb, not a question
    first = re.split(r"[\s,.!?]+", t.lower(), maxsplit=1)[0]
    if first in IMPER_VERBS and not t.endswith("?"):
        return "action-item", f"imperative verb '{first}'"
    if NEED_KW.search(t) and re.search(r"\b(today|tomorrow|tonight|before|by)\b", t, re.I):
        return "action-item", "need-to + deadline"

    # small-talk: explicit greeting/closing/filler tokens, or very short ack
    if SMALL_KW.search(t):
        return "small-talk", "greeting/closing keyword"
    if len(t.split()) <= 3:
        return "small-talk", "very short utterance"

    return "unknown", "no rule fired"


def main():
    df = pd.read_csv(WEAK_CSV)
    rng = pd.Series(range(len(df))).sample(frac=1, random_state=SEED).tolist()
    df = df.iloc[rng].reset_index(drop=True)

    picks = []
    for cls in ["reminder", "emotional-support", "action-item", "small-talk", "unknown"]:
        sub = df[df["label"] == cls].head(PER_CLASS)
        picks.append(sub)
    sample = pd.concat(picks, ignore_index=True)

    # unlabeled sheet for humans
    unlabeled = sample[["text"]].copy()
    unlabeled["label"] = ""  # to be filled in
    unlabeled.to_csv(ART / "dev_unlabeled.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    # auto-gold for now
    gold_rows = []
    for _, r in sample.iterrows():
        gl, why = auto_gold(r["text"])
        needs_review = gl != r["label"]
        gold_rows.append((r["text"], gl, r["label"], why, needs_review))

    gold = pd.DataFrame(
        gold_rows,
        columns=["text", "label", "weak_label", "auto_gold_reason", "needs_review"],
    )
    gold.to_csv(ART / "dev_gold_auto.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    print(f"sampled {len(sample)} rows ({PER_CLASS} per class)")
    print(f"wrote {ART / 'dev_unlabeled.csv'} (hand-label here)")
    print(f"wrote {ART / 'dev_gold_auto.csv'} (auto-gold, used by train.py)")
    print()
    print(f"auto-gold vs weak-label agreement: "
          f"{(~gold['needs_review']).sum()}/{len(gold)} "
          f"({100 * (~gold['needs_review']).mean():.1f}%)")
    print()
    print("auto-gold distribution:")
    for cls, n in gold["label"].value_counts().items():
        print(f"  {cls:<20s} {n}")


if __name__ == "__main__":
    main()
