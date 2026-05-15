"""Intent inference CLI and library.

    python -m src.part2_intent.infer "remind me to call mom tomorrow"

Or from code:

    from src.part2_intent.infer import classify
    classify("how are you?")
    # -> {"label": "small-talk", "confidence": 0.87, "latency_ms": 1.2}
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import TypedDict

import joblib

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts" / "part2"
VEC_PATH = ART / "vectorizer.joblib"
MODEL_PATH = ART / "intent_model.joblib"


class Prediction(TypedDict):
    label: str
    confidence: float
    latency_ms: float


_cache: dict = {}


def _load():
    if "vec" not in _cache:
        _cache["vec"] = joblib.load(VEC_PATH)
        _cache["clf"] = joblib.load(MODEL_PATH)
    return _cache["vec"], _cache["clf"]


def classify(text: str) -> Prediction:
    vec, clf = _load()
    t0 = time.perf_counter()
    X = vec.transform([text])
    probs = clf.predict_proba(X)[0]
    idx = int(probs.argmax())
    label = clf.classes_[idx]
    conf = float(probs[idx])
    dt = (time.perf_counter() - t0) * 1000.0
    return {"label": str(label), "confidence": round(conf, 4), "latency_ms": round(dt, 3)}


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: python -m src.part2_intent.infer \"text to classify\"")
        return 2
    text = " ".join(argv[1:])
    # warm load so the reported latency reflects steady-state inference,
    # not the first-call model load
    _load()
    result = classify(text)
    print(f"text       : {text}")
    print(f"label      : {result['label']}")
    print(f"confidence : {result['confidence']:.4f}")
    print(f"latency    : {result['latency_ms']:.2f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
