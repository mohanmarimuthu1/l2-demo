"""Benchmark model size + inference latency against the budget.

Budgets:
  - combined .joblib size < 50 MB
  - average latency per message < 200 ms (CPU)
  - p99 latency per message < 200 ms (CPU)

Sample: 100 random messages from data/conversations.csv (one turn each).
"""

from __future__ import annotations

import random
import re
import statistics
import time
from pathlib import Path

import pandas as pd

from src.part2_intent.infer import classify, _load, VEC_PATH, MODEL_PATH

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "conversations.csv"

SIZE_BUDGET_MB = 50.0
LATENCY_BUDGET_MS = 200.0
N_SAMPLES = 100
SEED = 7

USER_PREFIX = re.compile(r"^\s*User\s*\d+\s*:\s*", re.I)


def sample_messages(n: int, seed: int) -> list[str]:
    df = pd.read_csv(DATA, header=None, names=["conversation"])
    rng = random.Random(seed)
    msgs: list[str] = []
    rows = df["conversation"].dropna().tolist()
    rng.shuffle(rows)
    for conv in rows:
        for line in conv.split("\n"):
            line = USER_PREFIX.sub("", line).strip()
            if line:
                msgs.append(line)
                if len(msgs) >= n:
                    return msgs
    return msgs


def main():
    size_bytes = VEC_PATH.stat().st_size + MODEL_PATH.stat().st_size
    size_mb = size_bytes / (1024 * 1024)

    # warm the model so first-call load doesn't pollute timings
    _load()

    msgs = sample_messages(N_SAMPLES, SEED)
    latencies: list[float] = []
    for m in msgs:
        # use the wall-clock around classify() rather than the internal
        # latency_ms, so we measure end-to-end and not just the inner call
        t0 = time.perf_counter()
        classify(m)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    avg = statistics.mean(latencies)
    p50 = statistics.median(latencies)
    p99 = sorted(latencies)[max(0, int(round(0.99 * len(latencies))) - 1)]
    mx = max(latencies)

    checks = [
        ("model size",       f"{size_mb:.2f} MB",  f"< {SIZE_BUDGET_MB:.0f} MB", size_mb < SIZE_BUDGET_MB),
        ("avg latency",      f"{avg:.2f} ms",     f"< {LATENCY_BUDGET_MS:.0f} ms", avg < LATENCY_BUDGET_MS),
        ("p99 latency",      f"{p99:.2f} ms",     f"< {LATENCY_BUDGET_MS:.0f} ms", p99 < LATENCY_BUDGET_MS),
    ]

    print(f"benchmark over {len(msgs)} messages")
    print(f"  vectorizer.joblib : {VEC_PATH.stat().st_size/1024:.1f} KB")
    print(f"  intent_model.joblib: {MODEL_PATH.stat().st_size/1024:.1f} KB")
    print(f"  combined size      : {size_mb:.2f} MB")
    print()
    print(f"  avg latency : {avg:.2f} ms")
    print(f"  p50 latency : {p50:.2f} ms")
    print(f"  p99 latency : {p99:.2f} ms")
    print(f"  max latency : {mx:.2f} ms")
    print()
    print(f"  {'check':<14s} {'value':>14s} {'budget':>14s} {'result':>8s}")
    print(f"  {'-'*14:<14s} {'-'*14:>14s} {'-'*14:>14s} {'-'*8:>8s}")
    all_pass = True
    for name, val, budget, ok in checks:
        flag = "PASS" if ok else "FAIL"
        all_pass &= ok
        print(f"  {name:<14s} {val:>14s} {budget:>14s} {flag:>8s}")
    print()
    print("OVERALL:", "PASS" if all_pass else "FAIL")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
