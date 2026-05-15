# Self-evaluation

Honest scoring per part, then a candid retrospective. Not graded by an automated rubric — these are my own numbers.

## Part 1 — Adaptive Persona Engine

**Score: 7 / 10**

The pipeline is clean and the artifact is what the spec asks for: per-day persona, drift score, top topics, top entities, trigger candidates. Day-binning is properly factored behind `assign_days()` so a real timestamp column can be swapped in without touching anything else. The choices (VADER for sentiment, TF-IDF + KMeans for topics, spaCy NER for entities) are deliberate — they're the right size for casual-chat data on CPU. Where it loses points: the drift detector fires zero triggers on this corpus because the corpus is internally homogeneous, so the most visible output of the pipeline is a row of nulls. That's an honest result, not a bug, but it means the demo can't show the "trigger detected!" path. With real session timestamps the same code would surface meaningful drift; without them, the row-order chronology is doing the best it can.

## Part 2 — Offline Intent Classifier

**Score: 6 / 10**

The bones are right and the infrastructure is honest: weak labeling at scale, a hand-curated 150-example dev set, a model that fits in well under 50 MB and predicts in single-digit milliseconds. Per-class metrics are reported with their full ugliness — `small-talk` and `unknown` carry most of the support, `reminder` has zero dev examples and zero recall, `action-item` has three dev examples and a 17% F1. Dev macro-F1 is 0.425. The model isn't bad for what it is; the dev set is just where the truth lives. What it doesn't do: handle out-of-distribution phrasing well, calibrate confidence (probabilities skew sharp), or recover from the rule labeler's biases. A few hundred human-labeled examples per class would probably move macro-F1 above 0.75 with no code changes. With a transformer that would happen too, but the model size and latency budget are the whole point of this part.

## Part 3 — Conflict-Resolving RAG

**Score: 8 / 10**

This is the part I'd defend hardest. The composite re-rank works (14 of 15 candidates change rank vs raw cosine; top-5 shifts by one element on the spec query). NLI catches real contradictions at high probability (0.99 on the strongest pair). The answer template reads naturally, the excerpt picker resolves to the actually-relevant user line (not just the first one), and the closing sentence distinguishes "emotional tone" from "factual details" instead of falling back to the meaningless "tone" label. Three synthetic sister chunks are injected for the controlled-contradiction demo, the natural contradictions found in the corpus are surfaced honestly, and both are labeled in the output table. Where it could be tighter: the synthetic chunks don't surface for the literal spec query because their embeddings sit further from "Did I mention anything about my sister?" than the natural sibling-mentions; only the recency-cued phrasing pulls one of them into top-5. That's diagnosed in the diagnostic script, documented in the README, and not papered over. The other weakness is that the corpus is multi-speaker, so the "contradictions" exist by construction; the README is upfront about this.

## Part 4 — System Design

**Score: 7 / 10**

One page, written in plain prose, no marketing tone. The split between what syncs and what stays local is concrete (SQLite for structured rows, Chroma for embeddings on device only, vector clocks on persona fields, append-only message log). The pros/cons table doesn't dodge — cold sync is expensive, last-writer-wins can drop concurrent edits, index drift between devices on different model versions is called out. Where it's weak: it's a sketch, not a real design document. There's no schema, no API contract, no concrete encryption-key-rotation flow, no failure-mode matrix for the sync service. It answers the spec as stated; it wouldn't survive review by a senior systems engineer who wanted to actually build it.

## What worked

- **Treating each part as independent.** No cross-imports between Part 1, 2, 3 except through `common/`. Each part has its own `artifacts/` subtree, its own CLI, its own README section. Refactors in one didn't touch the others.
- **Honest artifacts.** `chronology: "synthetic_row_order"` in the drift JSON, `synthetic: true` on the seeded chunks, weak-label provenance in the training set, dev macro-F1 reported as-is. Reviewers can audit every claim from the files alone.
- **The composite re-rank and the diagnose script.** Recency and emotional weight actually move ranks (not cosine cosplay), and the diagnose script proves it ablation-style with the top-5 set diff before I shipped the answer.
- **The Docker pre-bake.** First request to `/rag` on the hosted Space is ~2 s instead of 10–30 s because the NLI cross-encoder is baked in. That's a meaningfully better demo experience.

## What broke / what I'd redesign

- **The first hosted deploy got stuck in APP_STARTING for 16 minutes** because I assumed HF Spaces injected `PORT=7860`. It doesn't, reliably. I'd default the Dockerfile to 7860 from the start. Fixed mid-deploy; the second build went from push to RUNNING in 2 minutes.
- **The intent dev set is too small to back the macro-F1 number with confidence intervals.** With 150 examples the std-err on a per-class F1 is huge. I'd either grow the dev set to 500+ or report bootstrap CIs.
- **The Part 1 trigger threshold was tuned for a corpus with more drift than this one has.** It produces zero triggers on the actual data. I'd either calibrate the threshold against an injected synthetic drift event, or report drift as a continuous score without a trigger boolean.
- **The excerpt-picker fallback is silent in the UI.** When `_excerpt` can't find a user line containing the entity, it falls back and emits a warning to the log. The user-facing answer doesn't say "this excerpt may not be directly about the entity." I'd surface the fallback into the API response and let the UI mark those quotes as low-confidence.

## What I'd do with one more week

- Hand-label 500 intent examples and re-train. This is the single biggest lever on the demo's perceived quality.
- Add per-entity contradiction scoring instead of just pairwise NLI. "Two sisters" vs "one sister" is a number-disagreement we could catch more reliably than NLI's general entailment signal.
- Wire up a `/admin/rebuild` route gated by a token so the index can be regenerated without redeploying. Right now the chroma index is baked into the image; updating it means a full image rebuild.
- Replace Flask's dev server log with structured JSON logs and add a `/metrics` endpoint with request counts and latency histograms.

## What I'd do with one more month

- Replace the row-order day-binning with a proper session-segmentation heuristic that uses content cues (greetings, topic shifts, time-of-day phrases) to recover an approximate temporal order from the corpus. The honest version: this is an open problem and the heuristic would have to be evaluated against any timestamped corpus we can find.
- Cross-validate the composite re-rank weights against a labeled "which chunk is most relevant" dev set. Right now `0.5 / 0.3 / 0.2` is a defensible default, not a tuned choice.
- Build a real conflict-resolution UI: when the resolver flags inconsistency, the user gets a side-by-side view of the conflicting chunks with day-bucket context and can mark one as canonical. The system records the decision and uses it to bias future retrievals.
- Take Part 4 from a one-page sketch to a working `device-sync-service` prototype with the SQLite schema, a real CRDT for persona snapshots, and a vector-clock implementation tested under simulated partitions.
- Calibrate intent confidence with Platt scaling or isotonic regression on a held-out set.

## Where it's weakest, honestly

The intent classifier. The pipeline around it is solid (weak labeler, train script, dev set, benchmark, latency budget), but the dev macro-F1 of 0.425 is what a reviewer will see first. If you measure each part by its single most important number, this is the one I'd want to redo. Every other part has a defensible primary number — drift artifacts shipped, 0.993 NLI on the spec query, one-page system design that answers the brief. Part 2 has a 0.425 that needs the per-class breakdown and the labels-provenance story to make sense, and that's a worse demo than a single confident number.

The second-weakest is the Part 1 trigger detector firing zero triggers on this corpus. Same root cause: the data doesn't ask the right question of the design. With real timestamps, it would.
