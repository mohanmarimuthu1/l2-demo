# Deployment

This service ships as a Docker image with everything baked in: the chroma index, the intent classifier, the drift timeline, and the three pretrained models (`all-MiniLM-L6-v2`, `nli-deberta-v3-xsmall`, `en_core_web_sm`). First request is instant — no cold-start downloads.

## Target host

**HuggingFace Spaces (Docker SDK), free tier.**

Final image size measured locally: **1.65 GB** (`docker image inspect` reports `Size: 1654099477`). The `docker images` column reports a larger number because it sums the multi-platform manifest, but the per-platform image pushed and pulled is 1.65 GB.

Reasoning: 1.65 GB is well above the 500 MB threshold where Render's free tier starts to feel the pain — cold starts on Render free spin a container that has to pull the full image and warm up, and the free tier evicts idle services aggressively. HF Spaces is purpose-built for ML demos: free CPU tier comfortably handles 2 GB images, the chroma index and model files come along for the ride without per-pull bandwidth surprises, and it natively understands `Dockerfile`-based projects (`sdk: docker` in the `README.md` front matter).

The biggest individual chunks of that 1.65 GB:

| Component | Size |
|-----------|-----:|
| Python site-packages (torch CPU 698 MB, spacy 120 MB, scipy 113 MB, transformers 104 MB, sympy 78 MB, pandas 76 MB, others) | 1.84 GB layer (overlay-shared down) |
| HF model cache (MiniLM + NLI cross-encoder) | 386 MB |
| `artifacts/` (chroma index 296 MB + chunks 25 MB) | 337 MB |

If you want to move to Render later, nothing in this repo is HF-specific — the same image works there too.

## Files involved

| File | Role |
|------|------|
| `Dockerfile` | Two-stage build; pre-downloads models in builder stage, copies caches + artifacts into the runtime image |
| `Procfile` | `web: gunicorn app:app --workers=2 --timeout=120 --bind 0.0.0.0:$PORT` — for hosts that read it (HF Spaces ignores this; Render/Fly/Heroku honor it) |
| `.dockerignore` | Keeps `.git`, `.venv`, raw CSV inputs, and dev-set files out of the build context |
| `app.py` | Flask app exposing `/`, `/drift`, `/classify`, `/rag` |
| `artifacts/` | Pre-built outputs from Parts 1–3 (drift timeline, intent model, chroma index) |

## Required env vars

None at deploy time. The container honors these optional overrides:

| Var | Default | Use |
|-----|---------|-----|
| `PORT` | `5000` | Bind port. HF Spaces sets `7860`, Render sets `10000`, Fly sets `8080`. |
| `HF_HUB_OFFLINE` | `1` | Forces offline mode so a flaky network won't trigger a re-download. Set to `0` to allow on-the-fly fetches. |
| `TRANSFORMERS_OFFLINE` | `1` | Same idea for the transformers cache. |

No API keys, no secrets, no managed-service credentials.

## Step-by-step: HuggingFace Spaces

### 1. Create the Space

1. Go to https://huggingface.co/new-space
2. Name: `l2-demo` (or anything else)
3. License: pick one (MIT is fine for a demo)
4. **SDK: Docker** → "Blank" template
5. Hardware: **CPU basic (free)**
6. Visibility: Public or Private — your call

The Space is created with a single-file repo: a `README.md` containing the YAML front matter Spaces uses for metadata.

### 2. Push the repo

Spaces is a regular git remote. From this repo on your machine:

```bash
# replace YOUR_USERNAME with your HF username
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/l2-demo

# Spaces expects a README.md with a YAML header. The minimum is shown below
# under "Spaces README front matter" — write it before pushing.

git add .
git commit -m "initial deploy"
git push space master:main
```

If your local branch is `master` and Spaces wants `main`, the `master:main` refspec handles the rename in one shot.

The push triggers a build on Spaces. The image takes 8–12 minutes to build there because the model pre-downloads happen inside their builder, not yours.

### 3. Spaces README front matter

The `README.md` at the repo root needs this YAML block at the very top — Spaces reads it to know how to run your container. Add it as the first thing in your existing `README.md`, or create a new `README.md` if you don't have one yet:

```yaml
---
title: L2 Demo
emoji: ""
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---
```

Two notes:

- `app_port` must match the port the Spaces runtime will route traffic to. Spaces sets `$PORT=7860` and expects your app to bind there. Our `Dockerfile`'s `CMD` reads `$PORT`, so this works without code changes.
- `emoji` is required by the front matter parser but can be empty. Don't put an emoji in the title.

### 4. Verify the deploy

Once the Space build finishes, the URL pattern is:

```
https://YOUR_USERNAME-l2-demo.hf.space
```

Verification checklist (do all four):

1. **Index loads.** Open the URL in a browser. You should see the dark-themed page with the three tabs (Drift / Classify / RAG). The Drift tab should render the SVG line chart and the 7-row table within a second.
2. **Healthz.** `curl https://YOUR_USERNAME-l2-demo.hf.space/healthz` → `{"drift_loaded": true, "ok": true}`.
3. **Classify.**
   ```bash
   curl -X POST -H 'Content-Type: application/json' \
        -d '{"text":"remind me to call mom tomorrow"}' \
        https://YOUR_USERNAME-l2-demo.hf.space/classify
   ```
   Expect `{"label": "reminder", "confidence": 0.7..., "latency_ms": <single-digit>}`.
4. **RAG.**
   ```bash
   curl -X POST -H 'Content-Type: application/json' \
        -d '{"query":"Did I mention anything about my sister?"}' \
        https://YOUR_USERNAME-l2-demo.hf.space/rag
   ```
   First call is slow (10–30 s) — it loads the NLI cross-encoder into memory. Expect `contradictions` length ≥ 1 and the answer ending in "appear inconsistent on factual details." Subsequent calls drop to ~2 s.

If any of these fail, the Spaces UI has a `Logs` tab — look for the gunicorn output. Common failure modes are listed at the bottom of this file.

## Local Docker test (recommended before pushing)

```bash
docker build -t l2-demo:latest .
docker run --rm -p 5000:5000 l2-demo:latest
# in another shell:
curl http://localhost:5000/healthz
curl -X POST -H 'Content-Type: application/json' \
     -d '{"text":"hey how are you?"}' http://localhost:5000/classify
```

If `/healthz` returns `200` and `/classify` returns a label, the image is good. Stop the container with `Ctrl+C`.

## Common failure modes

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Build OOMs on the `pip install torch` step | HF Spaces free tier has a 16 GB build cache cap | The CPU-only wheel (`torch==2.4.1+cpu`) is already pinned. If it still OOMs, drop `torch` to `2.2.x` |
| Container starts but `/rag` hangs forever | `HF_HUB_OFFLINE=1` is set but the model cache didn't copy correctly | Check the builder stage actually downloaded the models: `docker run --rm l2-demo:latest ls /opt/hf-cache` |
| `404` at the Space URL | Build is still running or failed silently | Open the Space's `Logs` tab; rebuild if needed |
| Healthz returns `{"drift_loaded": false}` | `artifacts/part1/drift_timeline.json` wasn't in the COPY | Check `.dockerignore` didn't exclude it |
| Site loads but Drift tab is empty | Browser cached an old 404 | Hard refresh (`Ctrl+F5`) |

## What I would change for production

These are intentionally out of scope for the demo but listed for reference:

- Move the chroma index off the image and into a mounted volume so the image stays small and the index can be regenerated independently.
- Add a `/metrics` endpoint with request count + latency histograms.
- Wrap the NLI cross-encoder load in an explicit eager-load at startup so the first `/rag` call doesn't pay the 10 s tax.
- Replace the Flask dev-server hint with gunicorn-only deployment (already done — Flask only runs if `app.py` is executed directly).
