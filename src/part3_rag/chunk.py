"""Turn-pair chunking + synthetic sister-chunk injection.

A chunk is one consecutive turn-pair within a conversation (User 1 turn +
User 2 response). Odd trailing turns become standalone chunks. Each chunk
carries the day-bucket from Part 1's binning so retrieval can apply a recency
weight downstream.

Three synthetic "sister" chunks are injected at day-buckets 1, 4, 6 (the
literal spec asks for "1, 4, 7"; with n_bins=7 the valid range is [0,6], so
7 is clipped to 6 — same intent of an early/mid/late spread).

Output: artifacts/part3/chunks.json
"""

from __future__ import annotations

import json
from pathlib import Path

from src.common.io import load_conversations, explode_turns
from src.common.sentiment import vader_compound
from src.common.windows import assign_days


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "artifacts" / "part3"
OUT_PATH = OUT_DIR / "chunks.json"

N_BINS = 7

# Spec asked for 1/4/7; 7 is out of range for a 0..6 bucket index — clip to 6.
SYNTHETIC_SISTER_CHUNKS = [
    {
        "day_bucket": 1,
        "text": "User 1: my sister called today, we talked for almost an hour. she's "
                "honestly amazing — i feel so lucky to have her in my life.\n"
                "User 2: that's beautiful — sounds like you two are really close.",
    },
    {
        "day_bucket": 4,
        "text": "User 1: my sister is being impossible again. i can't deal with her "
                "right now, every conversation turns into an argument.\n"
                "User 2: that sounds exhausting. has it been building up for a while?",
    },
    {
        "day_bucket": 6,
        "text": "User 1: haven't spoken to my sister in a while. not really sure how "
                "she's doing — life just got busy on both sides i guess.\n"
                "User 2: do you think you'll reach out, or just let it be for now?",
    },
]


def _build_natural_chunks() -> list[dict]:
    convos = load_conversations()
    binned, info = assign_days(convos, n_bins=N_BINS)
    turns = explode_turns(binned[["conversation_id", "raw"]])
    day_lookup = dict(zip(binned["conversation_id"], binned["day"]))
    turns["day"] = turns["conversation_id"].map(day_lookup).astype(int)

    chunks: list[dict] = []
    # iterate per-conversation so pairs never cross a conversation boundary
    for cid, group in turns.groupby("conversation_id", sort=True):
        rows = group.sort_values("turn_index").reset_index(drop=True)
        i = 0
        while i < len(rows):
            r1 = rows.iloc[i]
            if i + 1 < len(rows):
                r2 = rows.iloc[i + 1]
                text = (
                    f"User {int(r1['speaker'])}: {r1['text']}\n"
                    f"User {int(r2['speaker'])}: {r2['text']}"
                )
                turn_idx = int(r1["turn_index"])
                i += 2
            else:
                text = f"User {int(r1['speaker'])}: {r1['text']}"
                turn_idx = int(r1["turn_index"])
                i += 1
            chunks.append({
                "id": f"c{int(cid)}_t{turn_idx}",
                "text": text,
                "conversation_id": int(cid),
                "turn_index": turn_idx,
                "day_bucket": int(r1["day"]),
                "sentiment": round(vader_compound(text), 4),
                "synthetic": False,
            })
    print(f"natural chunks: {len(chunks):,}")
    return chunks, info


def _build_synthetic_chunks() -> list[dict]:
    out = []
    for i, sp in enumerate(SYNTHETIC_SISTER_CHUNKS):
        out.append({
            "id": f"syn_sister_{i}",
            "text": sp["text"],
            "conversation_id": -1,
            "turn_index": 0,
            "day_bucket": sp["day_bucket"],
            "sentiment": round(vader_compound(sp["text"]), 4),
            "synthetic": True,
        })
    return out


def build() -> list[dict]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    natural, info = _build_natural_chunks()
    synthetic = _build_synthetic_chunks()
    chunks = natural + synthetic
    payload = {
        "info": {
            "n_bins": N_BINS,
            "binning": info,
            "n_natural": len(natural),
            "n_synthetic": len(synthetic),
            "n_total": len(chunks),
        },
        "chunks": chunks,
    }
    OUT_PATH.write_text(json.dumps(payload))
    print(f"wrote {OUT_PATH}  ({len(chunks):,} chunks, {len(synthetic)} synthetic)")
    for s in synthetic:
        print(f"  injected {s['id']} day={s['day_bucket']} sentiment={s['sentiment']:+.3f}")
    return chunks


def load() -> list[dict]:
    return json.loads(OUT_PATH.read_text())["chunks"]


if __name__ == "__main__":
    build()
