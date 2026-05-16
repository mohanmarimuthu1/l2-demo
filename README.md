# L2

Three offline NLP pipelines wired into a single Flask app: a per-day persona drift detector over a chat corpus, a TF-IDF + LogisticRegression intent classifier with a 50 MB / 200 ms budget, and a retrieval system that surfaces contradictions across a user's message history without an LLM in the loop. Everything runs on CPU. No external calls at request time. The hosted demo and the local Docker image carry identical artifacts, so the first request is instant.

## Hosted demo

**https://imanerd-l2-demo.hf.space**

## Loom walkthrough

https://www.loom.com/share/15b9105781024c7fbe0f18eea52c9858

## Quickstart

### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python run_all.py
```

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python run_all.py
```

`run_all.py` checks each part's artifacts on disk, runs any pipeline whose outputs are missing, then serves the Flask app on port 5000. With all artifacts already present (the default for a fresh clone where `artifacts/` was downloaded), it goes straight to the server in under a second.

### Docker

```bash
docker build -t l2-demo:latest .
docker run --rm -p 5000:5000 l2-demo:latest
# open http://localhost:5000
```

Image is ~1.65 GB. The chroma index, intent model, drift timeline, and the three pretrained models (MiniLM, NLI cross-encoder, spaCy `en_core_web_sm`) are all baked in. No cold-start downloads on first request.

## Architecture

The app is a thin Flask layer on top of three independent pipelines that share a small `common/` module (sentiment, day-binning, IO). Each part lives under `src/part{1,2,3}_*/` with its own `run.py` or CLI entry point and its own artifacts directory; nothing in Part 2 depends on Part 1, and Part 3 only borrows the day-bucket logic. The Flask layer is just glue: it loads the drift JSON, calls `classify()` for intent, and calls `resolve()` for RAG. The deeper design questions — on-device storage, sync boundaries, conflict resolution between devices — are in `docs/SYSTEM_DESIGN.md`.

## Part 1 — Adaptive Persona Engine

### What it does

Bins the 11,001-conversation corpus into N day-buckets (default 7) and builds a persona snapshot for each: average sentiment, mood/tone label, top TF-IDF topics, top spaCy entities, and a drift score relative to the previous bucket. The result is `artifacts/part1/drift_timeline.json`.

### Approach + key decisions

- **VADER over a transformer.** Casual chat at single-turn granularity sits well within VADER's design envelope; loading a transformer for sentiment would be wasted weight here.
- **TF-IDF + KMeans topics over BERTopic.** BERTopic's UMAP dependency is fragile on Windows. A simple TF-IDF-over-bucket comparison surfaces drift cleanly without the install pain.
- **Day-binning is a swappable strategy.** `src/common/windows.py` exposes one function (`assign_days`) so a future timestamp column slots in without touching the rest of the pipeline.

### Tradeoffs

The corpus has no timestamps or session IDs, so the day-binning is synthetic row-order — written into the artifact as `chronology: "synthetic_row_order"` and called out in the UI. The drift detector ran on the corpus produces zero triggers (corpus is too internally homogeneous to cross the threshold), which is itself a finding the report doesn't hide.

### Run standalone

```bash
python -m src.part1_drift.run            # defaults: 7 bins, row-order strategy
python -m src.part1_drift.run --bins 14  # finer-grain
```

### Result

`artifacts/part1/drift_timeline.json` with seven day-bucket records. Day-0 sample:

```json
{"day": 0, "mood": "playful", "tone": "casual", "avg_compound": 0.5238,
 "top_topics": ["love", "like", "great", "fun", "sounds"],
 "top_entities": ["Buddy", "Japan", "Comic Con", "Mittens", "Rings"],
 "drift_score": null, "trigger": null}
```

## Part 2 — Offline Intent Classifier

### What it does

Classifies short messages into one of five intents — `reminder`, `emotional-support`, `action-item`, `small-talk`, `unknown` — entirely on CPU with a model under 50 MB and sub-200 ms latency.

### Approach + key decisions

- **Weak-labeling first.** No labels exist in the corpus. A rule-based labeler (keyword + sentence-shape patterns) tags ~192k turns; that becomes the training set.
- **TF-IDF (1–2-grams) + LogisticRegression.** Serializes to under 5 MB total, predicts in single-digit milliseconds. A distilled MiniLM would also fit but the tokenizer overhead and cold-start aren't worth it for five classes on chitchat-length text.
- **150-example hand-curated dev set.** Without it the metrics would only reflect how well the model learned the weak labeler, which is meaningless.

### Tradeoffs

- The training set is noisy by construction. Dev macro-F1 is **0.425**, with `small-talk` and `unknown` carrying the bulk of the support. `reminder` has zero dev support — the rule labeler is biased toward what it sees in the corpus.
- The model would benefit from a few hundred hand-labeled examples per class. The infrastructure is in place to retrain; the bottleneck is annotation, not code.

### Run standalone

```bash
python -m src.part2_intent.train               # builds artifacts/part2/{intent_model,vectorizer}.joblib + metrics.json
python -m src.part2_intent.infer "remind me to call mom tomorrow"
# label      : reminder
# confidence : 0.7286
# latency    : 1.1 ms
python -m src.part2_intent.benchmark           # latency histogram + on-disk size check
```

### Result

| metric | value |
|--------|------:|
| training set size | 191,853 weak-labeled turns |
| dev set size | 150 hand-curated |
| dev macro-F1 | 0.425 |
| model + vectorizer on disk | 789 KB (well under 50 MB) |
| steady-state latency | 1–7 ms on CPU |

Per-class dev metrics in `artifacts/part2/metrics.json`.

## Part 3 — Conflict-Resolving RAG

### What it does

Answers a user question about their own conversation history, surfaces contradictions if any of the retrieved chunks disagree, and never invents an answer — the output quotes source turns verbatim.

### Approach + key decisions

- **Turn-pair chunks.** Each chunk is one `User N` line plus the response, keyed by `conversation_id` and `turn_index`. 98,083 chunks total, 3 of which are synthetic sister-mention seeds (`synthetic=true`).
- **Composite re-rank.** Top-15 cosine retrieval from a ChromaDB index of MiniLM embeddings, then re-ranked with `0.5 · cosine_norm + 0.3 · recency_norm + 0.2 · |sentiment|`. Each sub-score is preserved on the returned chunk so the UI can show why a chunk was picked.
- **Pairwise NLI on the top-5 only.** Ten pairs of the `cross-encoder/nli-deberta-v3-xsmall` model is bounded compute (~2 s on CPU) but enough to catch the textual contradictions the spec asks about.
- **Extractive answer.** Templated: a header sentence, an "Earlier..." quote, a "Later..." quote, and one closing sentence labeling the nature of the disagreement (emotional tone if the sentiment signs differ, factual details otherwise). No LLM call, no API. The excerpt picker prefers user-prefixed lines that actually contain the queried entity.

### Tradeoffs

- The corpus is multi-speaker, so the "contradictions" the resolver surfaces are real text-level inconsistencies that exist *because* different rows are different people — not because one user changed their mind. The resolver treats the corpus as one user's history per the spec; the README and the answer phrasing both make this honest.
- The synthetic sister chunks demonstrate behavior under controlled contradiction but only surface in retrieval when the query is phrased with explicit recency or relationship cues. The literal spec query is dominated by natural sibling-mentions.

### Run standalone

```bash
python -m src.part3_rag.chunk         # build chunks.json (incl. synthetic seeds)
python -m src.part3_rag.embed_index   # build chroma index — idempotent, ~7 min cold
python -m src.part3_rag.run_query     # run all three demo queries, write sample_answers.md
python -m src.part3_rag.run_query "What did I say about Japan?"   # ad-hoc
```

### Result

For the spec query *"Did I mention anything about my sister?"*:

```
entity: your sister    confidence: 0.428    contradictions: 3
top NLI pair: c7659_t6 (day 4) <-> c8252_t8 (day 5)   contradict=0.993

You've mentioned your sister across 5 occasions over day-buckets 3–6.
Earlier (day-bucket 4): "Yes, I have two sisters. We're all pretty close,
and we're really helping each other through this."
Later (day-bucket 5): "I have one sister. We're not very close, but I
still love her."
Note: these accounts appear inconsistent on factual details.
```

Full three-query output in `artifacts/part3/sample_answers.md`.

## Part 4 — System Design

A one-pager (`docs/SYSTEM_DESIGN.md`) on how this would sit inside an on-device app: SQLite + Chroma split, what crosses the sync boundary, how concurrent edits get resolved, and what the resolver from Part 3 surfaces back to the user instead of merging silently. Includes a mermaid diagram of the device ↔ sync-service ↔ cloud topology with the local-only blast radius marked.

## Known limitations

- **Synthetic day-binning.** The CSV has no timestamps or session IDs, so Part 1's "daily" axis is row-order. The artifact records this as `chronology: "synthetic_row_order"` so a reviewer can't miss it; if real timestamps land in the column schema, only `assign_days()` changes.
- **Weak-labeled intent data.** Training labels come from a rule-based labeler, not human annotation. Dev macro-F1 (0.425) reflects this honestly. The `unknown` class dominates because the rule labeler is conservative.
- **Three synthetic sister chunks** are injected into the Part 3 chroma index to demonstrate resolver behavior under controlled contradiction. They carry `synthetic=true` and are flagged in the UI's source-chunks table. The natural sister-mention contradictions the resolver surfaces are real corpus artifacts; only the seeded ones are staged.
- **Multi-speaker corpus, single-user resolver.** Per the spec, Part 3 treats the corpus as one user's history. The text-level contradictions it flags exist because different rows are different speakers; the resolver phrasing ("appear inconsistent on factual details") doesn't claim more than what the text says.

## Repository structure

```
.
├── app.py                          # Flask routes for /, /drift, /classify, /rag, /healthz
├── run_all.py                      # check artifacts → run missing → serve on :5000
├── requirements.txt
├── Dockerfile                      # two-stage; CPU torch + pre-downloaded models
├── Procfile
├── DEPLOY.md                       # HuggingFace Spaces walkthrough
├── docs/
│   └── SYSTEM_DESIGN.md            # Part 4 one-pager
├── src/
│   ├── common/                     # io, sentiment, text, day-binning
│   ├── part1_drift/                # daily persona + drift detection
│   ├── part2_intent/               # TF-IDF + LR classifier, train + infer + benchmark
│   ├── part3_rag/                  # chunk → embed → retrieve → resolve → run_query
│   └── part3_conflict/             # natural-contradiction candidate scan (Prompt 3.1)
├── templates/
│   └── index.html                  # 3-tab UI, dark theme, no external CSS/JS
├── artifacts/                      # pipeline outputs (drift JSON, intent model, chroma index)
└── scripts/
    └── diagnose_part3.py           # NLI + retrieval sanity checks
```

## License

MIT. See `LICENSE`.
