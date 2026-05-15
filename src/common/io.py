"""Loading utilities for the project's source data."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CONVERSATIONS_CSV = ROOT / "data" / "conversations.csv"

_USER_PREFIX = re.compile(r"^\s*User\s*(\d+)\s*:\s*", re.I)


def load_conversations(path: Path | None = None) -> pd.DataFrame:
    """Return a DataFrame with one row per conversation.

    Columns:
        conversation_id   stable int id (= row index)
        raw               original cell text
    """
    p = Path(path) if path else CONVERSATIONS_CSV
    df = pd.read_csv(p, header=None, names=["raw"])
    df = df.dropna(subset=["raw"]).reset_index(drop=True)
    df.insert(0, "conversation_id", df.index.astype(int))
    return df


def split_turns(conv_text: str) -> list[tuple[int, str]]:
    """Return [(speaker, text), ...] where speaker is 1 or 2."""
    out: list[tuple[int, str]] = []
    for line in str(conv_text).split("\n"):
        m = _USER_PREFIX.match(line)
        if not m:
            continue
        body = line[m.end():].strip()
        if body:
            out.append((int(m.group(1)), body))
    return out


def explode_turns(df: pd.DataFrame) -> pd.DataFrame:
    """Long-form turn-per-row table.

    Columns:
        conversation_id, turn_index, speaker, text
    """
    rows = []
    for cid, raw in zip(df["conversation_id"], df["raw"]):
        for i, (spk, body) in enumerate(split_turns(raw)):
            rows.append((int(cid), i, spk, body))
    return pd.DataFrame(rows, columns=["conversation_id", "turn_index", "speaker", "text"])
