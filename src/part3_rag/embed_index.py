"""Embed chunks with all-MiniLM-L6-v2 and persist into a ChromaDB collection.

Idempotent: if the collection already exists with the expected chunk count,
the embedding step is skipped. Pass --force to rebuild.

    python -m src.part3_rag.embed_index           # build or skip-if-current
    python -m src.part3_rag.embed_index --force   # always rebuild
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from src.part3_rag.chunk import load as load_chunks


ROOT = Path(__file__).resolve().parents[2]
INDEX_DIR = ROOT / "artifacts" / "part3" / "index"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION = "chunks"
EMBED_BATCH = 128
ADD_BATCH = 1000


def _model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME)


def _client():
    import chromadb
    from chromadb.config import Settings
    return chromadb.PersistentClient(
        path=str(INDEX_DIR),
        settings=Settings(anonymized_telemetry=False),
    )


def _get_or_create(client, drop: bool = False):
    if drop:
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass
    return client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def build(force: bool = False) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    chunks = load_chunks()
    print(f"loaded {len(chunks):,} chunks")

    client = _client()
    col = _get_or_create(client, drop=False)
    existing = col.count()
    if not force and existing == len(chunks):
        print(f"collection '{COLLECTION}' already has {existing:,} items — skipping rebuild")
        return
    if existing and existing != len(chunks):
        print(f"collection count mismatch ({existing:,} vs {len(chunks):,}) — rebuilding")
    col = _get_or_create(client, drop=True)

    print(f"loading {MODEL_NAME}...")
    model = _model()

    texts = [c["text"] for c in chunks]
    print(f"encoding {len(texts):,} chunks (batch={EMBED_BATCH})...")
    t0 = time.time()
    embs = model.encode(
        texts,
        batch_size=EMBED_BATCH,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    dt = time.time() - t0
    print(f"encoded in {dt:.1f}s ({len(texts)/dt:.0f} chunks/s) — shape {embs.shape}")

    # bulk insert in slabs to keep chroma's transaction overhead down
    ids = [c["id"] for c in chunks]
    metas = [
        {
            "conversation_id": int(c["conversation_id"]),
            "turn_index": int(c["turn_index"]),
            "day_bucket": int(c["day_bucket"]),
            "sentiment": float(c["sentiment"]),
            "synthetic": bool(c["synthetic"]),
        }
        for c in chunks
    ]

    n = len(chunks)
    t0 = time.time()
    for i in range(0, n, ADD_BATCH):
        j = min(i + ADD_BATCH, n)
        col.add(
            ids=ids[i:j],
            embeddings=embs[i:j].tolist(),
            documents=texts[i:j],
            metadatas=metas[i:j],
        )
    print(f"added {n:,} items to chroma in {time.time()-t0:.1f}s")
    print(f"final count: {col.count():,}")


def open_collection():
    """Used by retrieve.py."""
    client = _client()
    return client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    build(force=args.force)


if __name__ == "__main__":
    main()
