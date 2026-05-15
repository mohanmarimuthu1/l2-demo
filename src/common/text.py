"""Light text normalization. Intentionally non-destructive: punctuation is
collapsed, not stripped, so VADER and downstream feature extractors still see
exclamation marks, question marks, and repeated chars that carry signal.
"""

from __future__ import annotations

import re
import unicodedata

_WS = re.compile(r"\s+")
_REPEAT_PUNCT = re.compile(r"([!?.,])\1{2,}")     # !!!! -> !
_REPEAT_CHARS = re.compile(r"(.)\1{3,}")          # sooooo -> soo (keep one duplicate)
_DASHES = re.compile(r"[‐-―−]")    # unicode dashes -> ascii dash
_QUOTES = re.compile(r"[‘’‚‛]")  # curly singles -> '
_DQUOTES = re.compile(r"[“”„‟]")  # curly doubles -> "


def clean(text: str) -> str:
    if text is None:
        return ""
    t = unicodedata.normalize("NFKC", str(text))
    t = _DASHES.sub("-", t)
    t = _QUOTES.sub("'", t)
    t = _DQUOTES.sub('"', t)
    t = _REPEAT_PUNCT.sub(r"\1", t)
    t = _REPEAT_CHARS.sub(r"\1\1", t)
    t = _WS.sub(" ", t).strip()
    return t.lower()


def clean_preserve_case(text: str) -> str:
    """Same as clean() but keeps case. Useful for NER and caps-as-emotion."""
    if text is None:
        return ""
    t = unicodedata.normalize("NFKC", str(text))
    t = _DASHES.sub("-", t)
    t = _QUOTES.sub("'", t)
    t = _DQUOTES.sub('"', t)
    t = _REPEAT_PUNCT.sub(r"\1", t)
    t = _REPEAT_CHARS.sub(r"\1\1", t)
    return _WS.sub(" ", t).strip()
