---
title: L2 Demo
emoji: ""
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# L2 Demo — Persona Drift, Intent Classification, Conflict-Resolving RAG

A small Flask app that wires three offline NLP pipelines into one page:

1. **Drift.** Daily persona snapshots over the conversation corpus, with sentiment, mood/tone, top topics, top entities, and per-day drift detection.
2. **Classify.** TF-IDF + LogisticRegression intent classifier (5 classes), trained on a weak-labeled bootstrap and evaluated on a 150-example dev set.
3. **RAG.** Top-15 retrieval with a composite re-rank (cosine + recency + emotional weight), then pairwise NLI on the top-5 to flag contradictions. The answer is extractive — no LLM, no API.

Everything runs offline on CPU. No external calls at request time.

## Stack

- Python 3.11, Flask + gunicorn (2 workers, 120 s timeout)
- VADER for sentiment, scikit-learn TF-IDF + LR for intent
- `sentence-transformers/all-MiniLM-L6-v2` for embeddings
- `cross-encoder/nli-deberta-v3-xsmall` for contradiction detection
- ChromaDB persistent client for the vector index (98,083 chunks pre-built into the image)
- spaCy `en_core_web_sm` for entity extraction in Part 1

## Routes

- `GET /` — the three-tab UI (Drift / Classify / RAG)
- `GET /drift` — drift timeline as JSON (table + sentiment chart source)
- `POST /classify` — body `{text}`, returns `{label, confidence, latency_ms}`
- `POST /rag` — body `{query}`, returns `{answer, contradictions, source_chunks, confidence}`
- `GET /healthz` — `{ok: true, drift_loaded: bool}`

## Local

```bash
python -m venv .venv && .venv/Scripts/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python run_all.py    # builds any missing artifacts, then serves on :5000
```

## Docker

```bash
docker build -t l2-demo:latest .
docker run --rm -p 5000:5000 l2-demo:latest
```

Final image is ~1.65 GB. The chroma index, intent model, and drift timeline are baked into the image, so the first request is instant — no cold-start build of the index.

## Deploy

See `DEPLOY.md` for the HuggingFace Spaces walkthrough (this Space).

## Notes on the corpus

The CSV is 11,001 conversations from a multi-persona dataset with no timestamps. Part 1's "daily" timeline is derived from row-order bins, declared in the artifact as `chronology: "synthetic_row_order"`. The Part 3 resolver treats the corpus as one user's history, which means the contradictions it surfaces (e.g. "I have two sisters, we're close" vs "I have one sister, we're not close") are real text-level inconsistencies — they exist in the corpus because different rows are different speakers. Three synthetic sister-mention chunks (`source: synthetic_seed`) are also seeded into the index to demonstrate resolver behavior under controlled contradiction; one of them surfaces when the query phrasing is recency-cued.
