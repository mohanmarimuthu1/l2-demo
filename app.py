"""Flask demo wiring Parts 1, 2, 3 into a single page.

Routes:
    GET  /          index.html (3 tabs: Drift / Classify / RAG)
    GET  /drift     drift_timeline.json as a clean table + line-chart points
    POST /classify  {text} -> {label, confidence, latency_ms}
    POST /rag       {query, entity?} -> {answer, contradictions, source_chunks, confidence}

Run via:  python run_all.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from src.part2_intent.infer import classify as classify_intent


ROOT = Path(__file__).resolve().parent
DRIFT_PATH = ROOT / "artifacts" / "part1" / "drift_timeline.json"

log = logging.getLogger("app")
app = Flask(__name__, template_folder="templates")


# ---- helpers ---------------------------------------------------------------

def _load_drift() -> dict:
    if not DRIFT_PATH.exists():
        return {"info": {}, "timeline": []}
    with DRIFT_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _trim_timeline_for_ui(payload: dict) -> dict:
    """Strip features down to what the table renders, keep the chart-relevant
    fields, and limit top_topics / top_entities to 5 each. Cuts payload to a
    fraction of the on-disk size without losing anything the UI shows."""
    out = {"info": payload.get("info", {}), "timeline": []}
    for d in payload.get("timeline", []):
        out["timeline"].append({
            "day": d.get("day"),
            "mood": d.get("mood"),
            "tone": d.get("tone"),
            "avg_compound": d.get("avg_compound"),
            "drift_score": d.get("drift_score"),
            "drift_from_prev": d.get("drift_from_prev"),
            "top_topics": [t.get("term") for t in (d.get("top_topics") or [])[:5]],
            "top_entities": [e.get("text") for e in (d.get("top_entities") or [])[:5]],
            "trigger": d.get("trigger"),
        })
    return out


# ---- routes ----------------------------------------------------------------

@app.get("/")
def index():
    return render_template("index.html")


@app.get("/drift")
def drift():
    payload = _trim_timeline_for_ui(_load_drift())
    return jsonify(payload)


@app.post("/classify")
def classify_route():
    body = request.get_json(silent=True) or {}
    text = (body.get("text") or "").strip()
    if not text:
        return jsonify({"error": "text required"}), 400
    result = classify_intent(text)
    return jsonify(result)


@app.post("/rag")
def rag_route():
    # Import lazily — pulling chroma + sentence-transformers + NLI on import
    # would make the first /classify request pay the cost.
    from src.part3_rag.resolve import resolve

    body = request.get_json(silent=True) or {}
    query = (body.get("query") or "").strip()
    entity = body.get("entity")
    k = int(body.get("k") or 15)
    if not query:
        return jsonify({"error": "query required"}), 400
    result = resolve(query, k=k, entity=entity)
    return jsonify(result)


@app.get("/healthz")
def healthz():
    return jsonify({"ok": True, "drift_loaded": DRIFT_PATH.exists()})


if __name__ == "__main__":
    # Direct invocation is fine for ad-hoc debugging, but run_all.py is the
    # supported entry point and gates on artifact freshness first.
    app.run(host="127.0.0.1", port=5000, debug=False)
