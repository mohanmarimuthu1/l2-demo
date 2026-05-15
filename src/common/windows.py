"""Day-bucket assignment.

The CSV has no timestamps and no session/conversation_id column (each row is
already one full session). So in this dataset, row-order binning and
session-binning collapse to the same operation. The functions below keep both
strategies available — if a real timestamp or session id ever shows up,
``pick_strategy`` will route to the right one.

A binned DataFrame has a ``day`` column with integer values in [0, n_bins).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


SUPPORTED_STRATEGIES = ("row", "session", "timestamp")


def _candidate_time_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        lc = str(c).lower()
        if lc in {"timestamp", "created_at", "date", "datetime", "ts"}:
            return c
    return None


def _candidate_session_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        lc = str(c).lower()
        if lc in {"session_id", "conversation_id"} and lc != "conversation_id":
            # conversation_id alone isn't a session axis here — every row already
            # is its own conversation. Only treat it as session if multiple rows
            # share the same id (real session grouping).
            return c
        if lc == "conversation_id":
            # check if it actually groups multiple rows
            if df[c].nunique(dropna=True) < len(df):
                return c
    return None


def pick_strategy(df: pd.DataFrame) -> tuple[str, str]:
    """Return (strategy, reason)."""
    tcol = _candidate_time_col(df)
    if tcol is not None:
        return "timestamp", f"found temporal column '{tcol}'"
    scol = _candidate_session_col(df)
    if scol is not None:
        return "session", f"found grouping column '{scol}'"
    return "row", "no temporal or grouping column present, falling back to row order"


def bin_by_rows(df: pd.DataFrame, n_bins: int = 7) -> pd.DataFrame:
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    out = df.copy()
    n = len(out)
    if n == 0:
        out["day"] = pd.Series(dtype="int64")
        return out
    # equal-width by index; final bin absorbs any remainder
    edges = np.linspace(0, n, n_bins + 1, dtype=int)
    day = np.zeros(n, dtype=int)
    for b in range(n_bins):
        day[edges[b]:edges[b + 1]] = b
    out["day"] = day
    return out


def bin_by_session(df: pd.DataFrame, n_bins: int = 7, session_col: str = "session_id") -> pd.DataFrame:
    """Group by session_col, order sessions by first-appearance, then bin
    consecutive sessions into n_bins. Within a session, all rows share the
    same day.
    """
    if session_col not in df.columns:
        raise KeyError(f"session column '{session_col}' not in df")
    order = df[session_col].drop_duplicates().tolist()
    session_to_day: dict = {}
    n_sessions = len(order)
    if n_sessions == 0:
        out = df.copy()
        out["day"] = pd.Series(dtype="int64")
        return out
    edges = np.linspace(0, n_sessions, n_bins + 1, dtype=int)
    for b in range(n_bins):
        for s in order[edges[b]:edges[b + 1]]:
            session_to_day[s] = b
    out = df.copy()
    out["day"] = out[session_col].map(session_to_day).astype(int)
    return out


def bin_by_timestamp(df: pd.DataFrame, n_bins: int = 7, time_col: str | None = None) -> pd.DataFrame:
    if time_col is None:
        time_col = _candidate_time_col(df)
    if time_col is None:
        raise KeyError("no temporal column found")
    out = df.copy()
    t = pd.to_datetime(out[time_col], errors="coerce", utc=True)
    if t.isna().all():
        raise ValueError(f"could not parse '{time_col}' as datetimes")
    days = (t - t.min()).dt.total_seconds() / 86400.0
    out["day"] = pd.cut(days, bins=n_bins, labels=False, include_lowest=True).astype(int)
    return out


def assign_days(df: pd.DataFrame, n_bins: int = 7, strategy: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Add a 'day' column. Auto-pick strategy unless one is forced.

    Returns (df_with_day, info) where info is a small dict suitable for
    embedding in output artifacts:

        {"strategy": "row", "reason": "...", "n_bins": 7, "chronology": "synthetic_row_order"}
    """
    if strategy is None:
        strategy, reason = pick_strategy(df)
    else:
        if strategy not in SUPPORTED_STRATEGIES:
            raise ValueError(f"unknown strategy '{strategy}'")
        reason = f"forced by caller"

    if strategy == "row":
        out = bin_by_rows(df, n_bins=n_bins)
        chronology = "synthetic_row_order"
    elif strategy == "session":
        scol = _candidate_session_col(df) or "session_id"
        out = bin_by_session(df, n_bins=n_bins, session_col=scol)
        chronology = f"session_order:{scol}"
    elif strategy == "timestamp":
        out = bin_by_timestamp(df, n_bins=n_bins)
        chronology = "wallclock"
    else:
        raise AssertionError(strategy)

    info = {
        "strategy": strategy,
        "reason": reason,
        "n_bins": int(n_bins),
        "chronology": chronology,
    }
    return out, info
