"""Train TF-IDF + LogisticRegression on the weak labels, evaluate on dev.

Eval prefers a hand-labeled dev_gold.csv if present; otherwise falls back to
dev_gold_auto.csv from dev_set.py. Saves model, vectorizer, and metrics.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts" / "part2"

LABELED_CSV = ART / "labeled.csv"
DEV_GOLD = ART / "dev_gold.csv"           # hand-labeled, preferred
DEV_GOLD_AUTO = ART / "dev_gold_auto.csv"  # heuristic fallback

VEC_PATH = ART / "vectorizer.joblib"
MODEL_PATH = ART / "intent_model.joblib"
METRICS_PATH = ART / "metrics.json"

LABELS = ["reminder", "emotional-support", "action-item", "small-talk", "unknown"]


def load_dev() -> tuple[pd.DataFrame, str]:
    if DEV_GOLD.exists():
        df = pd.read_csv(DEV_GOLD)
        if "label" in df.columns and df["label"].notna().any():
            return df.dropna(subset=["label"]).reset_index(drop=True), "hand"
    df = pd.read_csv(DEV_GOLD_AUTO)
    return df, "auto"


def main():
    print("loading weak-labeled training data...")
    train_df = pd.read_csv(LABELED_CSV)
    train_df = train_df.dropna(subset=["text", "label"]).reset_index(drop=True)
    print(f"  {len(train_df):,} training rows")

    dev_df, dev_source = load_dev()
    print(f"loaded dev set ({dev_source}): {len(dev_df)} rows")

    vec = TfidfVectorizer(
        max_features=10_000,
        ngram_range=(1, 2),
        min_df=2,
        lowercase=True,
        strip_accents="unicode",
    )
    X_train = vec.fit_transform(train_df["text"].astype(str))
    y_train = train_df["label"].values

    print(f"vocab size: {len(vec.vocabulary_):,}")

    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
    )
    print("fitting LogisticRegression...")
    clf.fit(X_train, y_train)

    joblib.dump(vec, VEC_PATH)
    joblib.dump(clf, MODEL_PATH)
    print(f"saved {VEC_PATH.name} ({VEC_PATH.stat().st_size/1024:.1f} KB)")
    print(f"saved {MODEL_PATH.name} ({MODEL_PATH.stat().st_size/1024:.1f} KB)")

    # eval on dev
    X_dev = vec.transform(dev_df["text"].astype(str))
    y_dev = dev_df["label"].values
    y_pred = clf.predict(X_dev)

    labels_in_dev = sorted(set(y_dev) | set(y_pred))
    macro_f1 = f1_score(y_dev, y_pred, average="macro", labels=labels_in_dev, zero_division=0)
    report = classification_report(
        y_dev, y_pred,
        labels=LABELS,
        zero_division=0,
        output_dict=True,
    )
    cm = confusion_matrix(y_dev, y_pred, labels=LABELS).tolist()

    metrics = {
        "dev_source": dev_source,
        "n_train": int(len(train_df)),
        "n_dev": int(len(dev_df)),
        "vocab_size": int(len(vec.vocabulary_)),
        "macro_f1_dev": round(float(macro_f1), 4),
        "per_class": {
            cls: {
                "precision": round(report.get(cls, {}).get("precision", 0.0), 4),
                "recall": round(report.get(cls, {}).get("recall", 0.0), 4),
                "f1": round(report.get(cls, {}).get("f1-score", 0.0), 4),
                "support_in_dev": int(report.get(cls, {}).get("support", 0)),
            }
            for cls in LABELS
        },
        "confusion_matrix": {
            "labels": LABELS,
            "rows_are_true": True,
            "values": cm,
        },
        "model_size_bytes": int(MODEL_PATH.stat().st_size + VEC_PATH.stat().st_size),
    }

    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print()
    print(f"macro F1 on dev ({dev_source}-labeled): {macro_f1:.3f}")
    print()
    print("per-class:")
    print(f"  {'label':<20s} {'P':>6s} {'R':>6s} {'F1':>6s} {'n':>5s}")
    for cls in LABELS:
        row = metrics["per_class"][cls]
        print(f"  {cls:<20s} {row['precision']:>6.3f} {row['recall']:>6.3f} "
              f"{row['f1']:>6.3f} {row['support_in_dev']:>5d}")
    print()
    print("confusion matrix (rows=true, cols=pred):")
    header = "                     " + " ".join(f"{c[:8]:>9s}" for c in LABELS)
    print(header)
    for cls, row in zip(LABELS, cm):
        print(f"  {cls:<18s} " + " ".join(f"{v:>9d}" for v in row))
    print()
    print(f"metrics written to {METRICS_PATH}")


if __name__ == "__main__":
    main()
