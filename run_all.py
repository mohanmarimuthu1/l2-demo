"""End-to-end entry point.

Checks that each part's artifacts exist; runs the pipeline for any part whose
outputs are missing; then launches the Flask demo on port 5000.

    python run_all.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"

PART1_OUT = ART / "part1" / "drift_timeline.json"
PART2_OUT = [
    ART / "part2" / "intent_model.joblib",
    ART / "part2" / "vectorizer.joblib",
    ART / "part2" / "metrics.json",
]
PART3_CHUNKS = ART / "part3" / "chunks.json"
PART3_INDEX = ART / "part3" / "index" / "chroma.sqlite3"


def _run(mod: str, *args: str) -> None:
    cmd = [sys.executable, "-m", mod, *args]
    print(f"\n>>> {' '.join(cmd)}")
    rc = subprocess.call(cmd, cwd=str(ROOT))
    if rc != 0:
        raise SystemExit(f"step failed: {mod} (exit {rc})")


def ensure_part1() -> None:
    if PART1_OUT.exists():
        print(f"[ok] part1 artifact present: {PART1_OUT.relative_to(ROOT)}")
        return
    print("[run] part1 — building drift timeline")
    _run("src.part1_drift.run")


def ensure_part2() -> None:
    if all(p.exists() for p in PART2_OUT):
        print("[ok] part2 artifacts present (model + vectorizer + metrics)")
        return
    print("[run] part2 — training intent classifier")
    _run("src.part2_intent.train")


def ensure_part3() -> None:
    if not PART3_CHUNKS.exists():
        print("[run] part3 — building chunks (incl. synthetic seeds)")
        _run("src.part3_rag.chunk")
    else:
        print(f"[ok] part3 chunks present: {PART3_CHUNKS.relative_to(ROOT)}")

    if not PART3_INDEX.exists():
        print("[run] part3 — embedding + building chroma index (slow first time)")
        _run("src.part3_rag.embed_index")
    else:
        # check the collection has rows; embed_index is idempotent if not.
        print(f"[ok] part3 index present at {PART3_INDEX.parent.relative_to(ROOT)}")
        _run("src.part3_rag.embed_index")  # no-op when counts match


def launch_app(host: str = "127.0.0.1", port: int = 5000) -> None:
    from app import app
    print()
    print("=" * 60)
    print(f"Ready at http://{host}:{port}")
    print("=" * 60)
    print()
    app.run(host=host, port=port, debug=False, use_reloader=False)


def main() -> None:
    ensure_part1()
    ensure_part2()
    ensure_part3()
    launch_app()


if __name__ == "__main__":
    main()
